#ifndef THREADS_QUERY_AGENT
#define THREADS_QUERY_AGENT

#ifdef RICH_MPI

#include <mpi.h>
#include <algorithm>
#include <cmath>
#include <set>
#include <vector>
#include <mpi.h>
#include <thread>

// set data structure:
#include <boost/container/flat_set.hpp>
#include <unordered_set>


#ifdef TIMING
    #include <chrono>
#endif // TIMING

#include "../QueryAgent.hpp"
#include "utils/printing/print.hpp" // todo: remove

#include "ThreadsQueryAgentAnswerer.hpp"
#include "ThreadsQueryAgentSender.hpp"
#include "ThreadsQueryAgentReceiver.hpp"

template<typename QueryData, typename AnswerType>
class ThreadsQueryAgent : public QueryAgent<QueryData, AnswerType>
{
public:
    ThreadsQueryAgent(const TalkAgent<QueryData> *talkAgent, AnswerAgent<QueryData, AnswerType> *answerAgent, bool sendToSelf = false, const MPI_Comm &comm = MPI_COMM_WORLD);

    virtual ~ThreadsQueryAgent() = default;
    
    QueryBatchInfo<QueryData, AnswerType> runBatch(const std::vector<QueryData> &queries) override;
    
    inline std::vector<std::vector<size_t>> &getRecvData() override{return this->recvData;};
    
    inline std::vector<int> &getRecvProc() override{return this->recvProcessorsRanks;};

private:
    std::vector<std::vector<size_t>> recvData;
    std::vector<int> recvProcessorsRanks;
};

template<typename QueryData, typename AnswerType>
ThreadsQueryAgent<QueryData, AnswerType>::ThreadsQueryAgent(const TalkAgent<QueryData> *talkAgent, AnswerAgent<QueryData, AnswerType> *answerAgent, bool sendToSelf, const MPI_Comm &comm):
        QueryAgent<QueryData, AnswerType>(talkAgent, answerAgent, sendToSelf, comm)
{
    int provided;
    MPI_Query_thread(&provided);
    if(provided != MPI_THREAD_MULTIPLE)
    {
        throw UniversalError("ThreadsQueryAgent:: MPI_THREAD_MULTIPLE not provided. You should have used 'MPI_Init_thread' correctly");
    }
}

template<typename QueryData, typename AnswerType>
QueryBatchInfo<QueryData, AnswerType> ThreadsQueryAgent<QueryData, AnswerType>::runBatch(const std::vector<QueryData> &queries)
{
    TQA::Sender<QueryData> sender(this->comm, this->talkAgent, this->sendToSelf);
    std::thread senderThread([&sender, &queries](){return sender.run(queries);});
    
    TQA::Answerer<QueryData, AnswerType> answerer(this->comm, this->answerAgent, sender.getAmountSent(), sender.getFinishedSentFlag());
    std::thread answererThread([&answerer](){return answerer.run();});

    // TQA::Receiver receiver;
    // std::thread receiverThread([&receiver](){return receiver.run();});

    senderThread.join();
    answererThread.join();
    // receiverThread.join();
    
    MPI_Barrier(this->comm);

    std::cout << "done" << std::endl;

    QueryBatchInfo<QueryData, AnswerType> info;
    // return receiver.getFinalResult();
    return info;
}

// template<typename QueryData, typename AnswerType>
// class ThreadsQueryAgent<QueryData, AnswerType>::Receiver
// {
//     friend class ThreadsQueryAgent<QueryData, AnswerType>;

//     using SubQueryData = TQA<QueryData, AnswerType>::SubQueryData;

// private:
//     Receiver(const MPI_Comm &comm, const std::vector<size_t> &amountsSent, const bool &finishedTheSends);

//     ~Receiver() = default;

//     void run();

//     void initializeShouldAnswer();

//     template<typename T>
//     std::vector<std::byte> &addToSentData(const std::vector<T> &data);

//     void handleAnswering();

//     const bool &finishedTheSends;
//     const std::vector<size_t> &amountsSent;

//     size_t shouldAnswer;
//     size_t currentlyAnswered;
//     bool launchedShouldAnswer;
//     MPI_Request requestShouldAnswered;

//     MPI_Comm comm;
//     int size;
//     int rank;
//     AnswerAgent<QueryData, AnswerType> *answerAgent;
//     std::vector<MPI_Request> requests;
//     std::vector<std::vector<std::byte>> sentData;
// };


#endif // RICH_MPI

#endif // THREADS_QUERY_AGENT