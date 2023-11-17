#ifndef _RICH_QUERY_AGENT_H_
#define _RICH_QUERY_AGENT_H_

#ifdef RICH_MPI

#include <algorithm>
#include <cmath>
#include <set>
#include <queue>
#include <vector>
#include <mpi.h>

// set data structure:
#include <boost/container/flat_set.hpp>
#include <unordered_set>

#include "3D/environment/EnvironmentAgent.h"
#include "AnswerAgent.hpp"
#include "TalkAgent.hpp"

#define TAG_REQUEST 200
#define TAG_RESPONSE 201
#define TAG_FINISHED 202

#define UNDEFINED_BUFFER_IDX -1
#define FLUSH_QUERIES_NUM 50

namespace
{
    template<typename QueryData>
    struct SubQueryData
    {
        QueryData data;
        size_t parent_id;
    };
};

template<typename QueryData, typename AnswerType>
struct QueryInfo
{
    QueryData data;
    size_t id;
    // int subQueriesNum;
    std::vector<AnswerType> finalResults;
};

template<typename QueryData, typename AnswerType>
struct QueryBatchInfo
{
    std::vector<QueryInfo<QueryData, AnswerType>> queriesAnswers;
    std::vector<AnswerType> result;
    std::vector<std::vector<AnswerType>> dataByRanks;
};

template<typename QueryData, typename AnswerType>
class QueryAgent
{
public:
    template<typename T>
    using _set = boost::container::flat_set<T>;

    using _subQueryData = SubQueryData<QueryData>;
    using _queryBatchInfo = QueryBatchInfo<QueryData, AnswerType>;
    using _queryInfo = QueryInfo<QueryData, AnswerType>;

    QueryAgent(const TalkAgent<QueryData> *talkAgent, AnswerAgent<QueryData, AnswerType> *answerAgent, bool sendToSelf = false, const MPI_Comm &comm = MPI_COMM_WORLD);

    virtual ~QueryAgent() = default;
    
    void receiveQueries(QueryBatchInfo<QueryData, AnswerType> &batch);
    void answerQueries();
    void sendQuery(const QueryInfo<QueryData, AnswerType> &query);
    QueryBatchInfo<QueryData, AnswerType> runBatch(std::queue<QueryData> &queries);

    inline std::vector<std::vector<size_t>> &getSentData(){return this->answerAgent->getSentData();};
    inline std::vector<int> &getSentProc(){return this->answerAgent->getSentProc();};
    inline std::vector<std::vector<size_t>> &getRecvData(){return this->recvData;};
    inline std::vector<int> &getRecvProc(){return this->recvProcessorsRanks;};

private:
    MPI_Comm comm;
    int rank, size;
    std::vector<MPI_Request> requests;
    std::vector<std::vector<char>> buffers; // send buffers, so that they will not be allocated on the stack
    size_t receivedUntilNow; // number of answers I received until now
    size_t shouldReceiveInTotal; // number of answers I have to receive (to know when to finish)
    AnswerAgent<QueryData, AnswerType> *answerAgent; // an answer agent
    const TalkAgent<QueryData> *talkAgent; // answers to whom we should talk to, given a query
    bool sendToSelf; // should send queries to self
    bool finishedMyQueries; // if finished to answer my queries
    int finishedReceived; // number of 'finish' messages received

    std::vector<int> recvProcessorsRanks;  // a vector of ranks, that we received data from
    std::vector<std::vector<size_t>> recvData; // a vector of vectors. The vector in index `i` contains the data indices (relatively to my final answer of the batch) that sent to `this->sentProcessorsRanks[i]`.
    std::vector<size_t> ranksBufferIdx; // maps each rank to an index `i`, where this->buffers[i] contains the prepared buffer for sending to the rank (requests are sent in chunks).

    void rearrangeResult(_queryBatchInfo &queriesBatch);

    void sendFinish();
    int checkForFinishMessages() const;
    void flushBuffer(int _rank);
    inline void flushAll(){for(int i = 0; i < this->size; i++) this->flushBuffer(i);};
};

template<typename QueryData, typename AnswerType>
QueryAgent<QueryData, AnswerType>::QueryAgent(const TalkAgent<QueryData> *talkAgent, AnswerAgent<QueryData, AnswerType> *answerAgent, bool sendToSelf, const MPI_Comm &comm):
        comm(comm), talkAgent(talkAgent), answerAgent(answerAgent), sendToSelf(sendToSelf), finishedReceived(0)
{
    MPI_Comm_rank(this->comm, &this->rank);
    MPI_Comm_size(this->comm, &this->size);
    this->ranksBufferIdx.resize(this->size, UNDEFINED_BUFFER_IDX);
}

template<typename QueryData, typename AnswerType>
void QueryAgent<QueryData, AnswerType>::receiveQueries(_queryBatchInfo &batch)
{
    if(this->receivedUntilNow >= this->shouldReceiveInTotal)
    {
        return;
    }
    MPI_Status status;
    int receivedAnswer = 0;

    std::vector<_queryInfo> &queries = batch.queriesAnswers;
    MPI_Iprobe(MPI_ANY_SOURCE, TAG_RESPONSE, this->comm, &receivedAnswer, &status);

    std::vector<char> buffer;

    while(receivedAnswer)
    {
        // received a message
        ++this->receivedUntilNow;

        // prepare the reading buffer for receiving
        int count;
        MPI_Get_count(&status, MPI_BYTE, &count);
        if(buffer.size() < static_cast<size_t>(count))
        {
            buffer.resize(count);
        }

        // receive
        MPI_Recv(&buffer[0], count, MPI_BYTE, status.MPI_SOURCE, TAG_RESPONSE, this->comm, MPI_STATUS_IGNORE);

        // decode the message - id first, then length, then the data itself
        long int id;
        long int length;
        int pos = 0;

        id = *reinterpret_cast<long int*>(buffer.data()); // decode id
        length = *reinterpret_cast<long int*>(buffer.data() + sizeof(long int)); // decode length

        if(id < 0 or static_cast<size_t>(id) >= queries.size())
        {
            throw UniversalError("In QueryAgent::receiveQueries, id of answered query " + std::to_string(id) + " is illegal (expected 0-"+ std::to_string(queries.size()) + ")");
        }
        if(length > 0)
        {
            // insert the results to the data received by rank `status.MPI_SOURCE` and to the queries result
            queries[id].finalResults.resize(queries[id].finalResults.size() + length);

            // base pointers to data
            AnswerType* base = reinterpret_cast<AnswerType*>(buffer.data() + 2 * sizeof(long int));
            for(size_t i = 0; i < static_cast<size_t>(length); i++)
            {
                queries[id].finalResults[i] = base[i];
                batch.dataByRanks[status.MPI_SOURCE].emplace_back(base[i]);
            }
        }
        else
        {
            if(length < 0)
            {
                throw UniversalError("In QueryAgent::receiveQueries, length of query is negative");
            }
        }
        MPI_Iprobe(MPI_ANY_SOURCE, TAG_RESPONSE, this->comm, &receivedAnswer, &status);
    }

    if(this->finishedMyQueries and this->shouldReceiveInTotal == this->receivedUntilNow)
    {
        this->sendFinish();
    }
}

template<typename QueryData, typename AnswerType>
void QueryAgent<QueryData, AnswerType>::answerQueries()
{
    static int totalArrived = 0;

    MPI_Status status;
    int arrivedNew = 0;

    MPI_Iprobe(MPI_ANY_SOURCE, TAG_REQUEST, this->comm, &arrivedNew, &status);
    
    std::vector<char> arriveBuffer;

    // while arrived new messages, and we should answer until the end, or answer until a bound we haven't reached to
    while(arrivedNew != 0)
    {
        int count;
        MPI_Get_count(&status, MPI_BYTE, &count);
        if(arriveBuffer.size() < static_cast<size_t>(count))
        {
            arriveBuffer.resize(count);
        }
        MPI_Recv(&arriveBuffer[0], count, MPI_BYTE, status.MPI_SOURCE, TAG_REQUEST, this->comm, MPI_STATUS_IGNORE);
        int subQueries = count / sizeof(_subQueryData);
        totalArrived += subQueries;
        for(int i = 0; i < subQueries; i++)
        {
            const _subQueryData &query = *reinterpret_cast<_subQueryData*>(&arriveBuffer[i * sizeof(_subQueryData)]);
            // calculate the result
            std::vector<AnswerType> result = this->answerAgent->answer(query.data, status.MPI_SOURCE);
            long int resultSize = static_cast<long int>(result.size());

            this->buffers.push_back(std::vector<char>());
            std::vector<char> &to_send = this->buffers.back();
            size_t msg_size = 2 * sizeof(long int) + resultSize * sizeof(AnswerType);
            to_send.resize(msg_size);

            long int id = query.parent_id;

           *reinterpret_cast<long int*>(to_send.data()) = id;
           *reinterpret_cast<long int*>(to_send.data() + sizeof(long int)) = resultSize;

            if(resultSize > 0)
            {
                AnswerType *toSendData = reinterpret_cast<AnswerType*>(to_send.data() + sizeof(id) + sizeof(resultSize));
                std::memcpy(toSendData, result.data(), resultSize * sizeof(AnswerType));
            }
            this->requests.push_back(MPI_REQUEST_NULL);
            MPI_Isend(&to_send[0], msg_size, MPI_BYTE, status.MPI_SOURCE, TAG_RESPONSE, this->comm, &this->requests.back());
        }
        MPI_Iprobe(MPI_ANY_SOURCE, TAG_REQUEST, this->comm, &arrivedNew, &status);
    }
}

template<typename QueryData, typename AnswerType>
void QueryAgent<QueryData, AnswerType>::sendQuery(const _queryInfo &query)
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
            this->buffers.emplace_back(std::vector<char>());
            this->buffers.back().reserve(sizeof(_subQueryData) * FLUSH_QUERIES_NUM);
        }
        bufferIdx = this->ranksBufferIdx[_rank];
        this->buffers[bufferIdx].resize(this->buffers[bufferIdx].size() + sizeof(_subQueryData));
        _subQueryData &subQuery = *reinterpret_cast<_subQueryData*>(&(*(this->buffers[bufferIdx].end() - sizeof(_subQueryData))));
        subQuery.data = query.data;
        subQuery.parent_id = query.id;
        ++this->shouldReceiveInTotal;
    }
}

template<typename QueryData, typename AnswerType>
void QueryAgent<QueryData, AnswerType>::sendFinish()
{
    int dummy;
    for(int _rank = 0; _rank < this->size; _rank++)
    {
        this->requests.push_back(MPI_REQUEST_NULL);
        MPI_Isend(&dummy, 1, MPI_BYTE, _rank, TAG_FINISHED, this->comm, &this->requests.back());
    }
}

template<typename QueryData, typename AnswerType>
int QueryAgent<QueryData, AnswerType>::checkForFinishMessages() const
{
    int arrived = 0;
    MPI_Status status;
    MPI_Iprobe(MPI_ANY_SOURCE, TAG_FINISHED, this->comm, &arrived, &status);
    if(arrived)
    {
        int dummy = 0;
        MPI_Recv(&dummy, 1, MPI_BYTE, MPI_ANY_SOURCE, TAG_FINISHED, this->comm, MPI_STATUS_IGNORE);
        return 1;
    }
    return 0;
}

template<typename QueryData, typename AnswerType>
void QueryAgent<QueryData, AnswerType>::flushBuffer(int _rank)
{
    if((_rank < 0) or (_rank >= this->size))
    {
        throw UniversalError("Invalid rank (" + std::to_string(_rank) + "), in QueryAgent::flushBuffer");
    }
    int bufferIdx = this->ranksBufferIdx[_rank];
    if(bufferIdx == UNDEFINED_BUFFER_IDX)
    {
        return;
    }

    std::vector<char> &buffer = this->buffers[bufferIdx];
    if(buffer.size() > 0)
    {
        this->requests.push_back(MPI_REQUEST_NULL);
        MPI_Isend(buffer.data(), buffer.size(), MPI_BYTE, _rank, TAG_REQUEST, this->comm, &this->requests.back());
    }
    this->ranksBufferIdx[_rank] = UNDEFINED_BUFFER_IDX;
}

template<typename QueryData, typename AnswerType>
void QueryAgent<QueryData, AnswerType>::rearrangeResult(_queryBatchInfo &queriesBatch)
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
QueryBatchInfo<QueryData, AnswerType> QueryAgent<QueryData, AnswerType>::runBatch(std::queue<QueryData> &queries)
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
    queriesBatch.queriesAnswers.reserve(originalQueriesNum);
    std::vector<_queryInfo> &queriesInfo = queriesBatch.queriesAnswers;
    queriesBatch.dataByRanks.resize(this->size);
    this->finishedMyQueries = queries.empty();
    this->finishedReceived = 0;
    size_t i = 0;
    bool notEmpty = false;
    
    // if doesn't have any queries, send a finish message
    if(this->finishedMyQueries)
    {
        this->sendFinish();
    }
    while((!this->finishedMyQueries) or (this->finishedReceived < this->size))
    {
        if(!this->finishedMyQueries)
        {
            QueryData queryData = queries.front();
            queries.pop();
            queriesInfo.push_back({queryData, i, std::vector<AnswerType>()});
            _queryInfo &query = queriesInfo.back();

            this->sendQuery(query);
            if(i == (originalQueriesNum - 1))
            {
                // send the rest of the waiting (buffered) requests
                this->flushAll();
                this->finishedMyQueries = true;
                // if had several queries, but no communication was needed, send a finish message
                if(this->shouldReceiveInTotal == 0)
                {
                    this->sendFinish();
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
                    throw UniversalError("Rank " + std::to_string(this->rank) + " received unrecognized tag: " + std::to_string(status.MPI_TAG) + ", from rank " + std::to_string(status.MPI_SOURCE));
            }
        }
        ++i;
    }

    if(this->requests.size() > 0)
    {
        MPI_Waitall(this->requests.size(), &(*(this->requests.begin())), MPI_STATUSES_IGNORE); // make sure any query was indeed received
    }

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

#endif // _RICH_QUERY_AGENT_H_
