#ifndef QUERY_DATA_HPP
#define QUERY_DATA_HPP

#include <cstdint>
#include <vector>

#ifdef TIMING
#include <chrono>
#endif // TIMING

#ifdef RICH_MPI
#include <mpi_utils/serialize/Serializer.hpp>
#endif // RICH_MPI

template<typename QueryData>
struct SubQueryData
                #ifdef RICH_MPI
                    : public Serializable
                #endif // RICH_MPI
{
    size_t parent_id;
    QueryData data;

    SubQueryData(const QueryData &data, size_t parent_id): data(data), parent_id(parent_id)
    {};

    virtual ~SubQueryData() = default;

    SubQueryData(): data(QueryData()), parent_id(0){};

    #ifdef RICH_MPI
        virtual size_t load(const Serializer *serializer, size_t byteOffset) override
        {
            size_t bytes = 0;
            bytes += serializer->extract(this->parent_id, byteOffset);
            bytes += this->data.load(serializer, byteOffset + bytes);
            return bytes;
        }

        virtual size_t dump(Serializer *serializer) const override
        {
            size_t bytes = 0;
            bytes += serializer->insert(this->parent_id);
            bytes += this->data.dump(serializer);
            return bytes;
        }
    #endif // RICH_MPI
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
        std::chrono::system_clock::time_point beginClockTime;
        double finishSubmittingTime = 0;
        double receivedAllTime = 0;
        double finishTime = 0;
    #endif // TIMING
};

#endif // QUERY_DATA_HPP
