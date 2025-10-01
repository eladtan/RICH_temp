#ifndef RANK_SYNC_HPP
#define RANK_SYNC_HPP

#ifdef RICH_MPI

#include <mpi.h>
#include <vector>
#include <numeric>
#include <cassert>
#include <functional>

using rank_t = int;

std::vector<rank_t> GetRanksOrder(const MPI_Comm &comm);

void ForEachRankSync(const MPI_Comm &comm, const std::vector<rank_t> &order, const std::function<void(rank_t)> &func);

#endif // RICH_MPI

#endif // RANK_SYNC_HPP