#ifndef RMA_FACTORY_HPP
#define RMA_FACTORY_HPP

#ifdef RICH_MPI

#include "RemoteMemoryAgent.hpp"
#include "MPIRemoteMemoryAgent.hpp"
#ifdef RICH_IBV
#include "IBVRemoteMemoryAgent.hpp"
#endif

#include <memory>
#include <stdexcept>

enum class RDMA_Type
{
    MPI_RMA,
    IBV_RDMA
};

class RMAFactory
{
public:
    RMAFactory() = delete;

    template<typename T>
    static std::unique_ptr<RemoteMemoryAgent<T>> Create(RDMA_Type type, size_t count, MPI_Comm comm)
    {
        switch(type)
        {
            case RDMA_Type::MPI_RMA:
                return MPIRemoteMemoryAgent<T>::CreateWithDefaultInfo(count, comm);
            case RDMA_Type::IBV_RDMA:
#ifdef RICH_IBV
                return CreateIBV<T>(count, comm);
#else
                throw std::runtime_error("RMAFactory: IBV_RDMA selected but RICH_IBV is not enabled");
#endif
        }
        throw std::runtime_error("RMAFactory: unknown RDMA type");
    }

private:
#ifdef RICH_IBV
    static std::shared_ptr<IBVContext> &GetSharedContext()
    {
        static std::shared_ptr<IBVContext> context;
        return context;
    }

    template<typename T>
    static std::unique_ptr<RemoteMemoryAgent<T>> CreateIBV(size_t count, MPI_Comm agent_comm)
    {
        auto &ctx = GetSharedContext();
        if(not ctx)
        {
            ctx = std::make_shared<IBVContext>(MPI_COMM_WORLD);
        }
        return IBVRemoteMemoryAgent<T>::Create(count, *ctx, agent_comm);
    }
#endif
};

#endif // RICH_MPI

#endif // RMA_FACTORY_HPP
