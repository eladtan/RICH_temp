#ifndef QUERY_DATA_HPP
#define QUERY_DATA_HPP

#include "misc/serializable.hpp"

template<typename QueryData>
struct SubQueryData : public Serializable
{
    size_t parent_id;
    QueryData data;

    SubQueryData(const QueryData &data, size_t parent_id): data(data), parent_id(parent_id)
    {};

    SubQueryData(): data(QueryData()), parent_id(0){};

    inline size_t getChunkSize(void) const override
    {
        return 1 + this->data.getChunkSize();
    }

    std::vector<double> serialize(void) const override
    {
        std::vector<double> data;
        data.push_back(static_cast<double>(this->parent_id));
        std::vector<double> queryData = this->data.serialize();
        data.insert(data.end(), queryData.begin(), queryData.end());
        return data;
    }

    void unserialize(const std::vector<double> &data) override
    {
        size_t index = 0;
        this->parent_id = static_cast<size_t>(data[index]);
        index++;
        this->data.unserialize(std::vector<double>(data.begin() + index, data.begin() + index + this->data.getChunkSize()));
    }
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
    #ifdef TIMING
        std::chrono::_V2::system_clock::time_point beginClockTime; 
        double finishSubmittingTime;
        double receivedAllTime;
    #endif // TIMING
};

#endif // QUERY_DATA_HPP