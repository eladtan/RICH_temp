#ifndef BUSY_WAIT_QUERY_AGENT_HPP
#define BUSY_WAIT_QUERY_AGENT_HPP

#ifdef RICH_MPI

#include <algorithm>
#include <cmath>
#include <set>
#include <vector>
#include <mpi.h>

// set data structure:
#include <boost/container/flat_set.hpp>
#include <unordered_set>

#ifdef TIMING
    #include <chrono>
#endif // TIMING

#include "QueryAgent.hpp"
#include <mpi_utils/serialize/Serializer.hpp>
#include "misc/universal_error.hpp"

#define TAG_REQUEST 200
#define TAG_RESPONSE 201
#define TAG_FINISHED 202

#define UNDEFINED_BUFFER_IDX -1
#define FLUSH_QUERIES_NUM 50

template<typename QueryData, typename AnswerType>
class BusyWaitQueryAgent : public QueryAgent<QueryData, AnswerType>
{
public:
    template<typename T>
    using _set = boost::container::flat_set<T>;

    using _subQueryData = SubQueryData<QueryData>;
    using _queryBatchInfo = QueryBatchInfo<QueryData, AnswerType>;
    using _queryInfo = QueryInfo<QueryData, AnswerType>;

    BusyWaitQueryAgent(const TalkAgent<QueryData> *talkAgent, AnswerAgent<QueryData, AnswerType> *answerAgent, bool sendToSelf = false, const MPI_Comm &comm = MPI_COMM_WORLD);

    virtual ~BusyWaitQueryAgent() = default;
    
    QueryBatchInfo<QueryData, AnswerType> runBatch(const std::vector<QueryData> &queries) override;
    inline std::vector<std::vector<size_t>> &getRecvData() override{return this->recvData;};
    inline std::vector<int> &getRecvProc() override{return this->recvProcessorsRanks;};

private:
    std::vector<MPI_Request> requests;
    std::vector<Serializer> buffers; // send buffers, so that they will not be allocated on the stack
    size_t receivedUntilNow; // number of answers I received until now
    size_t shouldReceiveInTotal; // number of answers I have to receive (to know when to finish)
    bool finishedMyQueries; // if finished to answer my queries
    int finishedReceived; // number of 'finish' messages received

    std::vector<int> recvProcessorsRanks;  // a vector of ranks, that we received data from
    std::vector<std::vector<size_t>> recvData; // a vector of vectors. The vector in index `i` contains the data indices (relatively to my final answer of the batch) that sent to `this->sentProcessorsRanks[i]`.
    std::vector<size_t> ranksBufferIdx; // maps each rank to an index `i`, where this->buffers[i] contains the prepared buffer for sending to the rank (requests are sent in chunks).

    void receiveQueries(QueryBatchInfo<QueryData, AnswerType> &batch);
    void answerQueries();
    void sendQuery(const QueryInfo<QueryData, AnswerType> &query);
    void rearrangeResult(_queryBatchInfo &queriesBatch);

    void sendFinish(_queryBatchInfo &queriesBatch);
    int checkForFinishMessages() const;
    void flushBuffer(int _rank);
    inline void flushAll(){for(int i = 0; i < this->size; i++) this->flushBuffer(i);};
};

template<typename QueryData, typename AnswerType>
BusyWaitQueryAgent<QueryData, AnswerType>::BusyWaitQueryAgent(const TalkAgent<QueryData> *talkAgent, AnswerAgent<QueryData, AnswerType> *answerAgent, bool sendToSelf, const MPI_Comm &comm):
        QueryAgent<QueryData, AnswerType>(talkAgent, answerAgent, sendToSelf, comm)
{
    this->ranksBufferIdx.resize(this->size, UNDEFINED_BUFFER_IDX);
}

template<typename QueryData, typename AnswerType>
void BusyWaitQueryAgent<QueryData, AnswerType>::receiveQueries(_queryBatchInfo &batch)
{
    if(this->receivedUntilNow >= this->shouldReceiveInTotal)
    {
        return;
    }
    MPI_Status status;
    int receivedAnswer = 0;

    std::vector<_queryInfo> &queries = batch.queriesAnswers;
    
    MPI_Message msg;
    MPI_Improbe(MPI_ANY_SOURCE, TAG_RESPONSE, this->comm, &receivedAnswer, &msg, &status);

    while(receivedAnswer)
    {
        // received a message
        ++this->receivedUntilNow;
        
        // prepare the reading buffer for receiving
        int count;
        MPI_Get_count(&status, MPI_BYTE, &count);
        Serializer serializer;
        serializer.resize(count);

        MPI_Mrecv(serializer.getData(), serializer.size(), MPI_BYTE, &msg, MPI_STATUS_IGNORE);
        
        // decode message: first id, then result
        long int id;
        size_t bytes = 0;
        bytes += serializer.extract(id, bytes);
        if(id < 0 or static_cast<size_t>(id) >= queries.size())
        {
            UniversalError eo("BusyWaitQueryAgent::receiveQueries, id of answered query is illegal");
            eo.addEntry("Id", id);
            eo.addEntry("Expected range", "0-" + std::to_string(queries.size()));
            throw eo;
        }
        
        std::vector<AnswerType> result;
        bytes += serializer.extract(result, bytes);
        // std::cout << "Receiving response for id " << id << " (should receive in total is " << this->shouldReceiveInTotal << ", currently received: " << this->receivedUntilNow << ") - serializer size is " << serializer.size() << " bytes, result size is " << result.size() << " answers" << std::endl;

        size_t length = result.size();
        // insert the results to the data received by rank `status.MPI_SOURCE` and to the queries result
        queries[id].finalResults.resize(queries[id].finalResults.size() + length);

        for(size_t i = 0; i < static_cast<size_t>(length); i++)
        {
            queries[id].finalResults[i] = result[i];
            batch.dataByRanks[status.MPI_SOURCE].emplace_back(result[i]);
        }

        MPI_Improbe(MPI_ANY_SOURCE, TAG_RESPONSE, this->comm, &receivedAnswer, &msg, &status);
    }

    if(this->finishedMyQueries and this->shouldReceiveInTotal == this->receivedUntilNow)
    {
        this->sendFinish(batch);
    }
}

template<typename QueryData, typename AnswerType>
void BusyWaitQueryAgent<QueryData, AnswerType>::answerQueries()
{
    static int totalArrived = 0;

    MPI_Status status;
    int arrivedNew = 0;

    MPI_Message msg;
    MPI_Improbe(MPI_ANY_SOURCE, TAG_REQUEST, this->comm, &arrivedNew, &msg, &status);
    
    // while arrived new messages, and we should answer until the end, or answer until a bound we haven't reached to
    while(arrivedNew != 0)
    {
        Serializer queryRecvSerializer;
        int count;
        MPI_Get_count(&status, MPI_BYTE, &count);
        queryRecvSerializer.resize(count);
        
        MPI_Mrecv(queryRecvSerializer.getData(), queryRecvSerializer.size(), MPI_BYTE, &msg, MPI_STATUS_IGNORE);
        
        std::vector<_subQueryData> queries;
        size_t bytes = 0;
        while(bytes < count)
        {
            bytes += queryRecvSerializer.extract(queries.emplace_back(), bytes);
        }
        // std::cout << "Resizing query recv serializer to " << count << " bytes for " << queries.size() << " queries" << std::endl;
        totalArrived += queries.size();
        for(const _subQueryData &query : queries)
        {
            // std::cout << "Answering query " << query.data << std::endl;

            // calculate the result
            std::vector<AnswerType> result = this->answerAgent->answer(query.data, status.MPI_SOURCE);

            Serializer &serializer = this->buffers.emplace_back();
            serializer.insert(query.parent_id);
            serializer.insert(result);
            this->requests.push_back(MPI_REQUEST_NULL);
            // std::cout << "Sending response for id " << query.parent_id << " - serializer size is " << serializer.size() << " bytes, response size is " << result.size() << " answers" << std::endl;
            MPI_Isend(serializer.getData(), serializer.size(), MPI_BYTE, status.MPI_SOURCE, TAG_RESPONSE, this->comm, &this->requests.back());
        }
        MPI_Improbe(MPI_ANY_SOURCE, TAG_REQUEST, this->comm, &arrivedNew, &msg, &status);
    }
}

template<typename QueryData, typename AnswerType>
void BusyWaitQueryAgent<QueryData, AnswerType>::sendQuery(const _queryInfo &query)
{
    typename TalkAgent<QueryData>::RanksSet talkingRanks = this->talkAgent->getTalkList(query.data);
    for(const int &_rank : talkingRanks)
    {
        if((_rank == this->rank) and (!this->sendToSelf))
        {
            continue; // unnecessary to send
        }
        int bufferIdx = this->ranksBufferIdx[_rank];
        // check if a flush is needed

        if((bufferIdx != UNDEFINED_BUFFER_IDX) and ((this->buffers[bufferIdx].size() / sizeof(_subQueryData)) >= FLUSH_QUERIES_NUM))
        {
            // send buffer
            this->flushBuffer(_rank);
        }

        bufferIdx = this->ranksBufferIdx[_rank];
        if(bufferIdx == UNDEFINED_BUFFER_IDX)
        {
            this->ranksBufferIdx[_rank] = this->buffers.size();
            this->buffers.emplace_back();
        }
        bufferIdx = this->ranksBufferIdx[_rank];
        _subQueryData subQuery;
        subQuery.data = query.data;
        subQuery.parent_id = query.id;
        Serializer &serializer = this->buffers[bufferIdx];
        serializer.insert(subQuery);
        ++this->shouldReceiveInTotal;
    }
}

template<typename QueryData, typename AnswerType>
void BusyWaitQueryAgent<QueryData, AnswerType>::sendFinish(_queryBatchInfo &queriesBatch)
{
    #ifdef TIMING
        std::chrono::_V2::system_clock::time_point now = std::chrono::system_clock::now();
        queriesBatch.receivedAllTime = std::chrono::duration_cast<std::chrono::duration<double>>(now - queriesBatch.beginClockTime).count();
    #endif // TIMING

    for(int _rank = 0; _rank < this->size; _rank++)
    {
        this->requests.push_back(MPI_REQUEST_NULL);
        MPI_Isend(NULL, 0, MPI_BYTE, _rank, TAG_FINISHED, this->comm, &this->requests.back());
    }
}

template<typename QueryData, typename AnswerType>
int BusyWaitQueryAgent<QueryData, AnswerType>::checkForFinishMessages() const
{
    int arrived = 0;
    MPI_Status status;
    MPI_Iprobe(MPI_ANY_SOURCE, TAG_FINISHED, this->comm, &arrived, &status);
    if(arrived)
    {
        MPI_Recv(NULL, 0, MPI_BYTE, MPI_ANY_SOURCE, TAG_FINISHED, this->comm, MPI_STATUS_IGNORE);
        return 1;
    }
    return 0;
}

template<typename QueryData, typename AnswerType>
void BusyWaitQueryAgent<QueryData, AnswerType>::flushBuffer(int _rank)
{
    if((_rank < 0) or (_rank >= this->size))
    {
        UniversalError eo("BusyWaitQueryAgent::flushBuffer: invalid rank buffer flush");
        eo.addEntry("Rank", _rank);
        throw eo;
    }
    int bufferIdx = this->ranksBufferIdx[_rank];
    if(bufferIdx == UNDEFINED_BUFFER_IDX)
    {
        return;
    }

    Serializer &serializer = this->buffers[bufferIdx];
    if(serializer.size() > 0)
    {
        this->requests.push_back(MPI_REQUEST_NULL);
        // std::cout << "Sending serializer of size " << serializer.size() << " bytes to rank " << _rank << " for " << serializer.size() / sizeof(_subQueryData) << " queries (currently should receive: " << this->shouldReceiveInTotal << ")" << std::endl;
        MPI_Isend(serializer.getData(), serializer.size(), MPI_BYTE, _rank, TAG_REQUEST, this->comm, &this->requests.back());
    }
    this->ranksBufferIdx[_rank] = UNDEFINED_BUFFER_IDX;
}

template<typename QueryData, typename AnswerType>
void BusyWaitQueryAgent<QueryData, AnswerType>::rearrangeResult(_queryBatchInfo &queriesBatch)
{
     // make receive, by the order of the recv array
    for(size_t i = 0; i < this->recvProcessorsRanks.size(); i++)
    {
        int _rank = this->recvProcessorsRanks[i];
        std::vector<size_t> &rankRecvData = this->recvData[i]; // indices vector
        const std::vector<AnswerType> &newDataFromRank = queriesBatch.dataByRanks[_rank]; // the data itself
        rankRecvData.reserve(newDataFromRank.size());

        queriesBatch.result.reserve(queriesBatch.result.size() + rankRecvData.size());
        for(const AnswerType &_data : newDataFromRank)
        {
            rankRecvData.push_back(queriesBatch.result.size()); // the index of the data received by this rank
            queriesBatch.result.emplace_back(_data);
        }
    }
}

template<typename QueryData, typename AnswerType>
QueryBatchInfo<QueryData, AnswerType> BusyWaitQueryAgent<QueryData, AnswerType>::runBatch(const std::vector<QueryData> &queries)
{
    this->receivedUntilNow = 0; // reset the receive counter
    this->shouldReceiveInTotal = 0; // reset the should-be-received counter
    for(std::vector<size_t> &_receivedDataFromRank : this->recvData)
    {
        _receivedDataFromRank.clear();
    }

    this->buffers.clear();
    size_t originalQueriesNum = queries.size();
    this->buffers.reserve(10 * originalQueriesNum); // heuristic
    this->requests.reserve(10 * originalQueriesNum); // heuristic
    this->requests.clear();
    _queryBatchInfo queriesBatch;
    #ifdef TIMING
        queriesBatch.beginClockTime = std::chrono::system_clock::now();
    #endif // TIMING

    queriesBatch.queriesAnswers.reserve(originalQueriesNum);
    std::vector<_queryInfo> &queriesInfo = queriesBatch.queriesAnswers;
    queriesBatch.dataByRanks.resize(this->size);
    this->finishedMyQueries = queries.empty();
    this->finishedReceived = 0;
    size_t i = 0;

    // if doesn't have any queries, send a finish message
    if(this->finishedMyQueries)
    {
        this->sendFinish(queriesBatch);
        #ifdef TIMING
            std::chrono::_V2::system_clock::time_point now = std::chrono::system_clock::now();
            queriesBatch.finishSubmittingTime = std::chrono::duration_cast<std::chrono::duration<double>>(now - queriesBatch.beginClockTime).count();
        #endif // TIMING
    }
    while((!this->finishedMyQueries) or (this->finishedReceived < this->size))
    {
        if(!this->finishedMyQueries)
        {
            const QueryData &queryData = queries[i];
            queriesInfo.push_back({queryData, i, std::vector<AnswerType>()});
            _queryInfo &query = queriesInfo.back();

            this->sendQuery(query);
            if(i == (originalQueriesNum - 1))
            {
                // send the rest of the waiting (buffered) requests
                this->flushAll();
                #ifdef TIMING
                    std::chrono::_V2::system_clock::time_point now = std::chrono::system_clock::now();
                    queriesBatch.finishSubmittingTime = std::chrono::duration_cast<std::chrono::duration<double>>(now - queriesBatch.beginClockTime).count();
                #endif // TIMING

                this->finishedMyQueries = true;
                // if had several queries, but no communication was needed, send a finish message
                if(this->shouldReceiveInTotal == this->receivedUntilNow)
                {
                    this->sendFinish(queriesBatch);
                }
            }
        }

        MPI_Status status;
        int arrived;
        MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, this->comm, &arrived, &status);
        
        if(arrived)
        {
            switch(status.MPI_TAG)
            {
                case TAG_RESPONSE:
                    this->receiveQueries(queriesBatch);
                    break;
                case TAG_REQUEST:
                    this->answerQueries();
                    break;
                case TAG_FINISHED:
                    this->finishedReceived += this->checkForFinishMessages();
                    break;
                default:
                    UniversalError eo("Received unrecognized tag in BusyWaitQueryAgent");
                    eo.addEntry("Tag", status.MPI_TAG);
                    eo.addEntry("My rank", this->rank);
                    eo.addEntry("From whom", status.MPI_SOURCE);
                    throw eo;
            }
        }
        ++i;
    }

    if(this->requests.size() > 0)
    {
        MPI_Waitall(this->requests.size(), &(*(this->requests.begin())), MPI_STATUSES_IGNORE); // make sure any query was indeed received
    }

    #ifdef TIMING
        std::chrono::_V2::system_clock::time_point now = std::chrono::system_clock::now();
        queriesBatch.finishTime = std::chrono::duration_cast<std::chrono::duration<double>>(now - queriesBatch.beginClockTime).count();
    #endif // TIMING

    // add to the list the processors that sent us a message for the first time
    for(int _rank = 0; _rank < this->size; _rank++)
    {
        if(_rank == this->rank)
        {
            continue; // shouldn't be relevant
        }
        if(queriesBatch.dataByRanks[_rank].empty())
        {
            continue; // the rank `_rank` did not send us any message
        }
        size_t rankIndex = std::find(this->recvProcessorsRanks.begin(), this->recvProcessorsRanks.end(), _rank) - this->recvProcessorsRanks.begin();
        if(rankIndex == this->recvProcessorsRanks.size())
        {
            // rank is not inside the recvProcessors rank, add it
            this->recvProcessorsRanks.push_back(_rank);
            this->recvData.emplace_back(std::vector<size_t>());
        }
    }

    MPI_Barrier(this->comm);

    this->finishedReceived -= this->size;
    this->rearrangeResult(queriesBatch);

    return queriesBatch;
}

#endif // RICH_MPI

#endif // BUSY_WAIT_QUERY_AGENT_HPP
