#ifndef CLEAN_NODE_HPP
#define CLEAN_NODE_HPP

#ifdef RICH_MPI
#include <mpi.h>

//! Check that no foreign processes are using significant CPU on this node.
//! Each node elects a master (lowest global rank), collects PIDs from all
//! local ranks, then the master repeatedly polls `ps` for rogue processes
//! (CPU > 20%, excluding known PIDs).  Throws UniversalError if any found.
void ensureCleanNode(MPI_Comm comm = MPI_COMM_WORLD, int checkCount = 3, int sleepSeconds = 2);

#endif // RICH_MPI

#endif // CLEAN_NODE_HPP
