#ifndef MPI_COMMANDS_HPP
#define MPI_COMMANDS_HPP 1

#include <mpi_utils/types.h>

#ifdef RICH_MPI

#include <vector>
#include <chrono>
#include <mpi.h>
#include <functional>
#include <mpi_utils/mpi_commands.hpp>
#include "misc/utils.hpp"
#include "3D/tessellation/Tessellation3D.hpp"

#define MPI_TIMED_BARRIER_TAG 110503

#include "mpi_commands_2d.hpp"
#include "mpi_commands_3d.hpp"

void MPI_Timed_barrier(const MPI_Comm &comm, double seconds, std::string const &place);

#endif //RICH_MPI

#endif // MPI_COMMANDS_HPP
