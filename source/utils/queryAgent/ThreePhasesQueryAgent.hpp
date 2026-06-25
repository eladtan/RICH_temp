#ifndef THREE_PHASES_QUERY_AGENT_HPP
#define THREE_PHASES_QUERY_AGENT_HPP

#ifdef RICH_MPI
#include "mpi/mpi_commands.hpp"
#include <mpi_utils/serialize/Serializer.hpp>
#endif // RICH_MPI
#include "QueryAgent.hpp"

template<typename QueryData, typename AnswerType>
struct SubQueryAnswer 
                    #ifdef RICH_MPI
                        : public Serializable
                    #endif // RICH_MPI
{
public:
    SubQueryData<QueryData> query;
    size_t resultSize;
    std::vector<AnswerType> result;

    SubQueryAnswer(const SubQueryData<QueryData> &query, size_t resultSize, const std::vector<AnswerType> &result):
        query(query), resultSize(resultSize), result(result)
    {};

    SubQueryAnswer(): SubQueryAnswer(SubQueryData<QueryData>(), 0, std::vector<AnswerType>()){};

    #ifdef RICH_MPI
        size_t dump(Serializer *serializer) const override
        {
            size_t bytes = 0;
            bytes += this->query.dump(serializer);
            bytes += serializer->insert(this->result);
            return bytes;
        }
        
        size_t load(const Serializer *serializer, size_t byteOffset) override
        {
            size_t bytes = 0;
            bytes += serializer->extract(this->query, byteOffset);
            bytes += serializer->extract(this->result, byteOffset + bytes);
            return bytes;
        }
    #endif // RICH_MPI
};

template<typename QueryData, typename AnswerType>
class ThreePhasesQueryAgent : public QueryAgent<QueryData, AnswerType>
{
public:
    ThreePhasesQueryAgent(const TalkAgent<QueryData> *talkAgent, AnswerAgent<QueryData, AnswerType> *answerAgent, bool sendToSelf = false, const MPI_Comm &comm = MPI_COMM_WORLD):
        QueryAgent<QueryData, AnswerType>(talkAgent, answerAgent, sendToSelf, comm)
    {};

    QueryBatchInfo<QueryData, AnswerType> runBatch(const std::vector<QueryData> &queries) override;
    
    inline std::vector<std::vector<size_t>> &getRecvData() override
    {
        return this->recvData;
    }
    
    inline std::vector<int> &getRecvProc() override
    {
        return this->recvProcessorsRanks;
    }

private:
    std::vector<int> recvProcessorsRanks;
    std::vector<std::vector<size_t>> recvData;
};

template<typename QueryData, typename AnswerType>
QueryBatchInfo<QueryData, AnswerType> ThreePhasesQueryAgent<QueryData, AnswerType>::runBatch(const std::vector<QueryData> &queries)
{
    using QI = QueryInfo<QueryData, AnswerType>;
    using SQD = SubQueryData<QueryData>;
    using SQA = SubQueryAnswer<QueryData, AnswerType>;
    
    std::vector<QI> queriesAnswers;
    std::vector<AnswerType> result;
    std::vector<std::vector<AnswerType>> dataByRanks(this->size);

    // phase 1 - send all queries
    std::vector<std::vector<SQD>> to(this->size);
    size_t resultsShouldGet = 0;
    size_t queryID = 0; // equals to `queriesAnswers.size()` when used
    for(const QueryData &query : queries)
    {
        QI info = {query, queryID, std::vector<AnswerType>()};
        queriesAnswers.emplace_back(info);
        for(int _rank : this->talkAgent->getTalkList(query))
        {
            if(_rank == this->rank and not this->sendToSelf)
            {
                continue;
            }
            to[_rank].push_back({query, queryID});
            resultsShouldGet++;
        }
        queryID++;
    }
    
    std::vector<std::vector<SQD>> from = MPI_Exchange_all_to_all(to, this->comm);

    std::vector<std::vector<SQA>> answers(this->size);
    for(int _rank = 0; _rank < this->size; _rank++)
    {
        std::vector<SQA> &answersFromRank = answers[_rank];
        std::cout << "from rank " << _rank << ", got " << from[_rank].size() << " queries" << std::endl;
        for(const SQD &subQuery : from[_rank])
        {
            answersFromRank.emplace_back();
            SQA &ans = answersFromRank.back();
            ans.query = subQuery;
            ans.result = this->answerAgent->answer(subQuery.data, _rank);     
            ans.resultSize = ans.result.size();
            std::cout << "[rank " << this->rank << "] subQuery is " << ans.query.data << " (ID " << ans.query.parent_id << "), answer is " << ans.result << std::endl;
        }
    }

    // TODO: the problem - the size of the sent variables is not fixed
    
    std::vector<std::vector<SQA>> answersFromRanks = MPI_Exchange_all_to_all(answers, this->comm);
    for(int _rank = 0; _rank < this->size; _rank++)
    {
        std::vector<SQA> &answersFromRank = answersFromRanks[_rank];
        std::cout << "rank " << this->rank << " inspects rank " << _rank << ", answers from rank size is " << answersFromRank.size() << std::endl;
        if(answersFromRank.empty())
        {
            continue;
        }
        this->recvProcessorsRanks.push_back(_rank);
        size_t receivedBeforeThisRank = result.size();
        for(SQA &ans : answersFromRank)
        {
            size_t queryID = ans.query.parent_id;
            std::cout << "ans.query is " << ans.query.data << " (ID " << queryID << "), ans.resultSize is " << ans.resultSize << ", ans.resukt is " << ans.result << std::endl;
            queriesAnswers[queryID].finalResults.insert(queriesAnswers[queryID].finalResults.cend(), ans.result.cbegin(), ans.result.cend());
            dataByRanks[_rank].insert(dataByRanks[_rank].cend(), ans.result.cbegin(), ans.result.cend());
            result.insert(result.cend(), ans.result.cbegin(), ans.result.cend());
        }
        this->recvData.emplace_back();
        std::vector<size_t> &receivedFromRank = this->recvData.back();
        receivedFromRank.resize(result.size() - receivedBeforeThisRank);
        std::iota(receivedFromRank.begin(), receivedFromRank.end(), 0);
    }

    #ifdef TIMING
        std::chrono::_V2::system_clock::time_point beginClockTime; 
        double finishSubmittingTime;
        double receivedAllTime;
    #endif // TIMING
    return {queriesAnswers, result, dataByRanks};
}
#endif // THREE_PHASES_QUERY_AGENT_HPP