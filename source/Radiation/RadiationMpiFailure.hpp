#ifndef RADIATION_MPI_FAILURE_HPP
#define RADIATION_MPI_FAILURE_HPP

#ifdef RICH_MPI

#include <cstdio>
#include <cstdlib>
#include <mpi.h>

namespace RadiationMpi
{
[[noreturn]] inline void AbortFailure(
    char const* const Marker,
    char const* const Operation,
    int const MpiError,
    char const* const Detail = nullptr) noexcept
{
    int Rank = -1;
    (void)MPI_Comm_rank(MPI_COMM_WORLD, &Rank);
    int const EffectiveError = MpiError == MPI_SUCCESS ?
        MPI_ERR_OTHER : MpiError;
    char ErrorText[MPI_MAX_ERROR_STRING] = {};
    int ErrorTextLength = 0;
    if(MPI_Error_string(EffectiveError, ErrorText,
                        &ErrorTextLength) != MPI_SUCCESS)
        ErrorTextLength = 0;
    std::fprintf(stderr,
                 "%s rank=%d operation=%s mpi_error=%d "
                 "mpi_message=%.*s detail=%s\n",
                 Marker == nullptr ? "MG_RADIATION_MPI_FATAL" : Marker,
                 Rank, Operation == nullptr ? "unknown" : Operation,
                 EffectiveError, ErrorTextLength, ErrorText,
                 Detail == nullptr ? "none" : Detail);
    std::fflush(stderr);
    (void)MPI_Abort(MPI_COMM_WORLD, EffectiveError);
    std::abort();
}

inline void RequireSuccess(
    int const MpiError,
    char const* const Marker,
    char const* const Operation) noexcept
{
    if(MpiError != MPI_SUCCESS)
        AbortFailure(Marker, Operation, MpiError);
}
}

#endif

#endif
