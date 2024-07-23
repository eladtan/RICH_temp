#ifndef THREADS_QUERY_AGENT_ANSWERER_HPP
#define THREADS_QUERY_AGENT_ANSWERER_HPP

#include <mpi.h>
#include <vector>
#include "../AnswerAgent.hpp"

namespace TQA
{
    template<typename QueryData, typename AnswerType>
    class Receiver
    {
    public:
        Receiver(const MPI_Comm &comm, const std::vector<size_t> &amountsSent, const bool &finishedTheSends);

        ~Receiver() = default;

        void run();

    private:
        MPI_Comm comm;
        int size;
        int rank;
        AnswerAgent<QueryData, AnswerType> *answerAgent;
        std::vector<MPI_Request> requests;
        std::vector<std::vector<std::byte>> sentData;
    };
}

template<typename QueryData, typename AnswerType>
TQA::Receiver<QueryData, AnswerType>::Receiver(const MPI_Comm &comm, AnswerAgent<QueryData, AnswerType> *answerAgent, const std::vector<size_t> &amountsSent, const bool &finishedTheSends): 
    comm(comm), answerAgent(answerAgent), amountsSent(amountsSent), finishedTheSends(finishedTheSends)
{
    std::cout << "Initializing answerer" << std::endl;
    MPI_Comm_size(comm, &this->size);
    MPI_Comm_rank(comm, &this->rank);
    this->currentlyAnswered = 0;
    this->shouldAnswer = std::numeric_limits<size_t>::max();
    this->requestShouldAnswered = MPI_REQUEST_NULL;
}

#endif // THREADS_QUERY_AGENT_ANSWERER_HPP