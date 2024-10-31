#ifndef THREADS_QUERY_AGENT_ANSWERER_HPP
#define THREADS_QUERY_AGENT_ANSWERER_HPP

#include <mpi.h>
#include <vector>
#include "../AnswerAgent.hpp"
#include "TQASubQuery.hpp"

#define ANSWER_TAG 1206
#define SEND_TAG 1205

namespace TQA
{
    template<typename QueryData, typename AnswerType>
    class Answerer
    {
        using SubQueryData = TQA::SubQueryData<QueryData>;

    public:
        Answerer(const MPI_Comm &comm, AnswerAgent<QueryData, AnswerType> *answerAgent, const std::vector<size_t> &amountsSent, const bool &finishedTheSends);

        ~Answerer() = default;

        void run();

    private:
        void initializeShouldAnswer();

        template<typename T>
        std::vector<std::byte> &addToSentData(const std::vector<T> &data);

        void handleAnswering();

        const bool &finishedTheSends;
        const std::vector<size_t> &amountsSent;

        size_t shouldAnswer;
        size_t currentlyAnswered;
        bool launchedShouldAnswer;
        MPI_Request requestShouldAnswered;

        MPI_Comm comm;
        int size;
        int rank;
        AnswerAgent<QueryData, AnswerType> *answerAgent;
        std::vector<MPI_Request> requests;
        std::vector<std::vector<std::byte>> sentData;
    };
};

template<typename QueryData, typename AnswerType>
TQA::Answerer<QueryData, AnswerType>::Answerer(const MPI_Comm &comm, AnswerAgent<QueryData, AnswerType> *answerAgent, const std::vector<size_t> &amountsSent, const bool &finishedTheSends): 
    comm(comm), answerAgent(answerAgent), amountsSent(amountsSent), finishedTheSends(finishedTheSends)
{
    std::cout << "Initializing answerer" << std::endl;
    MPI_Comm_size(comm, &this->size);
    MPI_Comm_rank(comm, &this->rank);
    this->currentlyAnswered = 0;
    this->shouldAnswer = std::numeric_limits<size_t>::max();
    this->requestShouldAnswered = MPI_REQUEST_NULL;
}

template<typename QueryData, typename AnswerType>
template<typename T>
std::vector<std::byte> &TQA::Answerer<QueryData, AnswerType>::addToSentData(const std::vector<T> &data)
{
    this->sentData.emplace_back();
    std::vector<std::byte> &toReturn = this->sentData.back();
    size_t _size = data.size() * sizeof(T);
    toReturn.resize(_size);
    std::memcpy(toReturn.data(), data.data(), _size);
    return toReturn;
}

template<typename QueryData, typename AnswerType>
void TQA::Answerer<QueryData, AnswerType>::initializeShouldAnswer()
{
    std::vector<std::byte> &ones = this->addToSentData(std::vector<int>(this->size, 1));
    MPI_Ireduce_scatter(this->amountsSent.data(), &this->shouldAnswer, reinterpret_cast<int*>(ones.data()), MPI_UNSIGNED_LONG_LONG, MPI_SUM, this->comm, &this->requestShouldAnswered);
}

template<typename QueryData, typename AnswerType>
void TQA::Answerer<QueryData, AnswerType>::handleAnswering()
{
    if(this->shouldAnswer == std::numeric_limits<size_t>::max())
    {
        // not received yet the number of queries that should be answered, so check if there is any (prevents deadlock)
        int arrived;
        MPI_Iprobe(MPI_ANY_SOURCE, ANSWER_TAG, this->comm, &arrived, MPI_STATUS_IGNORE);
        if(not arrived)
        {
            return;
        }
    }
    else
    {
        if(this->currentlyAnswered >= this->shouldAnswer)
        {
            return; // nothing to answer
        }
    }
    
    MPI_Status status;
    SubQueryData subQuery;
    MPI_Recv(&subQuery, sizeof(SubQueryData), MPI_BYTE, MPI_ANY_SOURCE, SEND_TAG, this->comm, &status);
    int _rank = status.MPI_SOURCE;
    std::vector<AnswerType> result = this->answerAgent->answer(subQuery.data, _rank);

    this->sentData.emplace_back(std::vector<std::byte>(sizeof(subQuery) + result.size() * sizeof(AnswerType)));
    std::vector<std::byte> &answer = this->sentData.back();
    std::memcpy(answer.data(), &subQuery, sizeof(subQuery));
    std::memcpy(answer.data() + sizeof(subQuery), result.data(), result.size() * sizeof(AnswerType));
    this->requests.emplace_back(MPI_REQUEST_NULL);
    MPI_Isend(answer.data(), answer.size(), MPI_BYTE, _rank, ANSWER_TAG, this->comm, &this->requests.back());
    this->currentlyAnswered++;
}

template<typename QueryData, typename AnswerType>
void TQA::Answerer<QueryData, AnswerType>::run()
{
    bool initializedReduceScatterCall = false;
    bool finishedReduceScatterCall = false;
    while((not finishedReduceScatterCall) or (this->currentlyAnswered < this->shouldAnswer))
    {
        // handle variables
        if(this->finishedTheSends)
        {
            if(not initializedReduceScatterCall)
            {
                this->initializeShouldAnswer();
                initializedReduceScatterCall = true;
            }
            else
            {
                if(not finishedReduceScatterCall)
                {
                    int initialized;
                    MPI_Test(&this->requestShouldAnswered, &initialized, MPI_STATUS_IGNORE);
                    finishedReduceScatterCall = initialized;
                    if(finishedReduceScatterCall)
                    {
                        std::cout << "here, this->shouldAnswer: " << this->shouldAnswer << std::endl;
                    }
                }
            }
        
            this->handleAnswering();
        }
    }
    
    if(not this->requests.empty())
    {
        MPI_Waitall(this->requests.size(), this->requests.data(), MPI_STATUSES_IGNORE);
    }

    std::cout << "Finished! finishedReduceScatterCall: " << finishedReduceScatterCall << ", this->currentlyAnswered: " << this->currentlyAnswered << ", this->shouldAnswer: " << this->shouldAnswer << "this->currentlyAnswered < this->shouldAnswer: " << (this->currentlyAnswered < this->shouldAnswer) << std::endl;
    std::cout << "keep: ((not finishedReduceScatterCall) or (this->currentlyAnswered < this->shouldAnswer)) is " << ((not finishedReduceScatterCall) or (this->currentlyAnswered < this->shouldAnswer)) << std::endl;
}

#endif // THREADS_QUERY_AGENT_ANSWERER_HPP