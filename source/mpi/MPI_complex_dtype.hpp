#ifndef MPI_COMPLEX_DTYPE_HPP
#define MPI_COMPLEX_DTYPE_HPP

#ifdef RICH_MPI

#include <mpi.h>
#include <type_traits>

template<typename T>
struct MPI_has_complex_dtype : std::false_type {};

#endif // RICH_MPI

#endif // MPI_COMPLEX_DTYPE_HPP
