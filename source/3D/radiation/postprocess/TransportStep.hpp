#ifndef IMC_POSTPROCESS_TRANSPORT_STEP_HPP
#define IMC_POSTPROCESS_TRANSPORT_STEP_HPP

#include <exception>
#include <iostream>
#include <vector>

#include "source/3D/monte/MonteCarloManager3D.hpp"
#include "source/misc/universal_error.hpp"

#ifdef RICH_MPI
#include <mpi.h>
#endif

namespace imc_postprocess_tde {

// Runs one Monte Carlo transport step. A transport error is usually raised on
// a single rank (one packet), while the other ranks keep transporting. Letting
// it unwind would destroy the Monte Carlo manager on that rank only, and the
// manager frees its MPI windows collectively, so the failing rank would wait
// for the others forever and the job would hang until its time limit. The
// error is therefore reported here and the whole job aborted.
//
// This applies in embedded RunPostProcess mode too: its failure relay is an
// MPI_Allreduce that the ranks still transporting never reach, so a one-rank
// transport error cannot be turned into an exception on every rank there
// either. Errors raised on every rank outside the transport step (settings,
// restart checks) still propagate as exceptions. Without MPI the exception is
// rethrown unchanged.
//
// Only the step is guarded, not the manager's whole lifetime: the forward
// pass's manager lives in the runtime across checks that throw on every rank
// at once, where unwinding frees the windows collectively and must not abort.
inline void StepOrAbort(MonteCarloManager3D& manager,
                        std::vector<ComputationalCell3D> const& cells,
                        dt_t duration, char const* what, int rank)
{
    try {
        manager.step(cells, duration);
    } catch (...) {
        std::cerr << "Transport error in " << what << " on rank " << rank
                  << "; aborting the job." << std::endl;
        try {
            throw;
        } catch (UniversalError const& eo) {
            // The full report: message, entries and stack trace.
            reportError(eo, std::cerr);
        } catch (std::exception const& e) {
            // STORM errors: their entries are printed to the standard
            // output where the transport catches them.
            std::cerr << e.what() << std::endl;
        } catch (...) {
            std::cerr << "unknown exception" << std::endl;
        }
#ifdef RICH_MPI
        MPI_Abort(MPI_COMM_WORLD, 2);
#endif
        throw;
    }
}

} // namespace imc_postprocess_tde

#endif // IMC_POSTPROCESS_TRANSPORT_STEP_HPP
