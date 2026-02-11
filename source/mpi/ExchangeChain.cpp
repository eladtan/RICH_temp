#ifdef RICH_MPI

#include "ExchangeChain.hpp"

ExchangeChain::ExchangeChain(const MPI_Comm &comm) : comm(comm)
{
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
}

void ExchangeChain::Reset(size_t num)
{
    this->globalTransfer.clear();
    this->globalTransferOrigins.clear();
    this->lastTransfer.clear();
    this->origins.clear();

    for(size_t i = 0; i < num; i++)
    {
        this->lastTransfer[i] = {this->rank, i};
        this->globalTransfer[i] = {this->rank, i};
        this->globalTransferOrigins[i] = {this->rank, i};
        this->origins[i] = {this->rank, i};
    }
}

void ExchangeChain::UpdateTransferMap(const std::vector<rank_t> &ranks, const std::vector<std::vector<size_t>> &indices, const std::vector<size_t> &localIndices)
{
    std::vector<size_t> sendAmounts(this->size, 0);
    for(size_t i = 0; i < ranks.size(); i++)
    {
        rank_t _rank = ranks[i];
        sendAmounts[_rank] = indices[i].size();
    }

    std::vector<size_t> recvAmounts(this->size);
    MPI_Alltoall(sendAmounts.data(), 1, MPI_UNSIGNED_LONG_LONG, recvAmounts.data(), 1, MPI_UNSIGNED_LONG_LONG, comm);

    std::vector<size_t> offsets(this->size, 0);
    size_t currentOffset = localIndices.size();
    for(size_t i = 0; i < ranks.size(); i++)
    {
        rank_t _rank = ranks[i];
        offsets[_rank] = currentOffset;
        currentOffset += recvAmounts[_rank];
    }

    std::vector<size_t> otherRanksOffsets(this->size, 0);
    MPI_Alltoall(offsets.data(), 1, MPI_UNSIGNED_LONG_LONG, otherRanksOffsets.data(), 1, MPI_UNSIGNED_LONG_LONG, comm);

    this->lastTransfer.clear();
    for(size_t i = 0; i < localIndices.size(); i++)
    {
        size_t index = localIndices[i];
        this->lastTransfer[index] = std::make_pair(this->rank, i);
    }
    for(size_t i = 0; i < ranks.size(); i++)
    {
        rank_t _rank = ranks[i];
        size_t rankOffset = otherRanksOffsets[_rank];
        for(size_t j = 0; j < indices[i].size(); j++)
        {
            size_t index = indices[i][j];
            this->lastTransfer[index] = std::make_pair(_rank, rankOffset + j);
        }
    }
}

ExchangeChain ExchangeChain::Reverse(void) const
{
    ExchangeChain reversed(this->comm);
    reversed.globalTransfer = this->globalTransferOrigins;
    reversed.globalTransferOrigins = this->globalTransfer;
    reversed.lastTransfer = this->origins;
    reversed.origins = this->lastTransfer;
    return reversed;
}

void ExchangeChain::Exchange(const std::vector<rank_t> &ranks, const std::vector<std::vector<size_t>> &indices, const std::vector<size_t> &localIndices)
{
    this->UpdateTransferMap(ranks, indices, localIndices);
    {
        struct ForwardInformation
        {
            size_t newIndex;
            size_t previousIndexInSender;
            rank_t globalOriginalRank;
            size_t globalOriginalIndex;
        };

        std::vector<std::vector<ForwardInformation>> toSend(this->size);
        for(const std::pair<size_t, std::pair<rank_t, size_t>> &entry : this->lastTransfer)
        {
            size_t previousIndex = entry.first;
            rank_t currentRank = entry.second.first;
            size_t currentIndex = entry.second.second;

            auto [globalOriginalRank, globalOriginalIndex] = this->globalTransferOrigins.at(previousIndex);
            ForwardInformation &info = toSend[currentRank].emplace_back();
            info.newIndex = currentIndex;
            info.previousIndexInSender = previousIndex;
            info.globalOriginalRank = globalOriginalRank;
            info.globalOriginalIndex = globalOriginalIndex;
        }

        std::vector<std::vector<ForwardInformation>> received = MPI_Exchange_all_to_all(toSend, this->comm);
        this->origins.clear();
        this->globalTransferOrigins.clear();
        for(rank_t _rank = 0; _rank < this->size; _rank++)
        {
            const std::vector<ForwardInformation> &vec = received[_rank];
            for(const ForwardInformation &info : vec)
            {
                this->globalTransferOrigins[info.newIndex] = std::make_pair(info.globalOriginalRank, info.globalOriginalIndex);
                this->origins[info.newIndex] = std::make_pair(_rank, info.previousIndexInSender);
            }
        }
    }

    {
        this->globalTransfer.clear();
        struct UpdateOriginalInformation
        {
            size_t originalIndex;
            size_t newIndex;
        };
    
        std::vector<std::vector<UpdateOriginalInformation>> toSend(this->size);
        for(const std::pair<rank_t, std::pair<rank_t, size_t>> &entry : this->globalTransferOrigins)
        {
            size_t currentIndex = entry.first;
            rank_t originalRank = entry.second.first;
            size_t originalIndex = entry.second.second;

            UpdateOriginalInformation &info = toSend[originalRank].emplace_back();
            info.originalIndex = originalIndex;
            info.newIndex = currentIndex;
        }

        std::vector<std::vector<UpdateOriginalInformation>> received = MPI_Exchange_all_to_all(toSend, this->comm);
        for(rank_t _rank = 0; _rank < this->size; _rank++)
        {
            const std::vector<UpdateOriginalInformation> &vec = received[_rank];
            for(const UpdateOriginalInformation &info : vec)
            {
                this->globalTransfer[info.originalIndex] = std::make_pair(_rank, info.newIndex);
            }
        }
    }
}

#endif // RICH_MPI