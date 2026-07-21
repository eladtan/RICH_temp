#ifndef EXCHANGE_CHAIN_HPP
#define EXCHANGE_CHAIN_HPP

#ifdef RICH_MPI

#include "mpi_commands.hpp"
#include "serialize/mpi_commands.hpp"

class ExchangeChain
{
public:
    using RankTransferMap = boost::container::flat_map<size_t, std::pair<rank_t, size_t>>;

    ExchangeChain(const MPI_Comm &comm = MPI_COMM_WORLD);

    void Reset(size_t num);

    void Exchange(const std::vector<rank_t> &ranks, const std::vector<std::vector<size_t>> &indices, const std::vector<size_t> &localIndices);

    inline const std::pair<rank_t, size_t> &GetOrigin(size_t currentIndex) const{return this->globalTransferOrigins.at(currentIndex);};

    inline const std::pair<rank_t, size_t> &GetTarget(size_t oldIndex) const{return this->globalTransfer.at(oldIndex);};

    inline size_t GetNorg(void) const{return this->globalTransfer.size();};

    inline const RankTransferMap &GetTranslationMap(void) const{return this->globalTransfer;};

    inline const RankTransferMap &GetReversedTranslationMap(void) const{return this->globalTransferOrigins;};

    ExchangeChain Reverse(void) const;
    
private:
    MPI_Comm comm;
    rank_t rank, size;
    RankTransferMap globalTransfer; // original index -> (now rank, now index in rank)
    RankTransferMap globalTransferOrigins; // index now [after last transfer] -> (origin rank, origin index in rank)
    RankTransferMap lastTransfer; // index [before last transfer] -> (now rank, now index in rank)
    RankTransferMap origins; // index now [after last transfer] -> (origin rank, origin index in rank)

    void UpdateTransferMap(const std::vector<rank_t> &ranks, const std::vector<std::vector<size_t>> &indices, const std::vector<size_t> &localIndices);
};

#endif // RICH_MPI

#endif // EXCHANGE_CHAIN_HPP