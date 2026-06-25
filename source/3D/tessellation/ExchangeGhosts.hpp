#ifndef EXCHANGE_GHOSTS_HPP
#define EXCHANGE_GHOSTS_HPP

#ifdef RICH_MPI

#include <boost/container/flat_map.hpp>
#include "3D/tessellation/Tessellation3D.hpp"
#include "mpi/mpi_commands.hpp"
#include "mpi/mpi_commands.hpp"

boost::container::flat_map<size_t, std::pair<rank_t, size_t>> ExchangeGhosts(const Tessellation3D &tess);

#endif // RICH_MPI

#endif // EXCHANGE_GHOSTS_HPP