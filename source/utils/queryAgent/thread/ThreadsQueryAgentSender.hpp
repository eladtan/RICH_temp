#ifndef THREADS_QUERY_AGENT_SENDER_HPP
#define THREADS_QUERY_AGENT_SENDER_HPP

#include <mpi.h>
#include <vector>
#include "../TalkAgent.hpp"
#include "TQASubQuery.hpp"

#define SEND_TAG 1205

namespace TQA
{
    template<typename QueryData>
    class Sender
    {
        using SubQueryData = TQA::SubQueryData<QueryData>;

    public:
        Sender(const MPI_Comm &comm, const TalkAgent<QueryData> *talkAgent, bool sendToSelf = false);

        ~Sender() = default;

        void run(const std::vector<QueryData> &queries);

        const bool &getFinishedSentFlag() const{return this->finishedSentFlag;};

        const std::vector<size_t> &getAmountSent() const{return this->amountSent;};
        
    private:        
        MPI_Comm comm;
        int size;
        int rank;
        bool finishedSentFlag;
        const TalkAgent<QueryData> *talkAgent;
        std::vector<MPI_Request> requests;
        std::vector<SubQueryData> subQueries;
        std::vector<size_t> amountSent;
        bool sendToSelf;
    };
};

template<typename QueryData>
TQA::Sender<QueryData>::Sender(const MPI_Comm &comm, const TalkAgent<QueryData> *talkAgent, bool sendToSelf):
    comm(comm), talkAgent(talkAgent), sendToSelf(sendToSelf)
{
    MPI_Comm_size(this->comm, &this->size);
    MPI_Comm_rank(this->comm, &this->rank);
    this->amountSent.resize(this->size, 0);
    this->finishedSentFlag = false;
    std::cout << "Initializing sender" << std::endl;
}

template<typename QueryData>
void TQA::Sender<QueryData>::run(const std::vector<QueryData> &queries)
{
    this->requests.reserve(queries.size() * 10); // heuristic
    this->subQueries.reserve(queries.size() * 10); // heuristic
    std::fill(this->amountSent.begin(), this->amountSent.end(), 0);

    size_t subQueriesSize = sizeof(SubQueryData);
    size_t N = queries.size();
    for(size_t i = 0; i < N; i++)
    {
        const QueryData &query = queries[i];
        auto ranks = this->talkAgent->getTalkList(query);
        if(ranks.size() == 1)
        {
            continue;
        }
        this->subQueries.emplace_back();
        SubQueryData &subQuery = this->subQueries.back();
        subQuery.queryIndex = i;
        subQuery.data = query;
        for(int _rank : ranks)
        {
            if(_rank == this->rank and this->sendToSelf)
            {
                continue;
            }
            this->requests.emplace_back(MPI_REQUEST_NULL);
            MPI_Isend(&subQuery, subQueriesSize, MPI_BYTE, _rank, SEND_TAG, this->comm, &this->requests.back());
            this->amountSent[_rank]++;
        }
    }

    this->finishedSentFlag = true;

    if(not this->requests.empty())
    {
        MPI_Waitall(this->requests.size(), this->requests.data(), MPI_STATUSES_IGNORE);
    }
}

#endif // THREADS_QUERY_AGENT_SENDER_HPP