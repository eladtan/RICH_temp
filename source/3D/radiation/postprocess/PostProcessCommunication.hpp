#ifndef POST_PROCESS_COMMUNICATION_HPP
#define POST_PROCESS_COMMUNICATION_HPP

// Selection of the Monte Carlo particle-exchange engine for the post-process
// calculations.
//
// STORM already provides a runtime selector (STORM::CreateCommunicationEngine,
// with ManagerType Auto/RDMA/P2P). The post-process calculations used to bypass
// it and hard-code an RDMA engine at six separate sites, which left no way to
// request two-sided MPI exchange short of editing the sources. This header routes
// every one of those sites through the STORM factory instead, so the choice is a
// runtime flag: --transport.communication auto|rdma|p2p.
//
// Two-sided P2P has the smaller memory footprint; RDMA is generally faster where
// the fabric supports it. "auto" tries RDMA+OFI and falls back to P2P, logging
// which one it ended up using.

#include "PostProcessConfig.hpp"

namespace imc_postprocess_tde
{

//! \brief Human-readable name of a communication mode (available in every build).
inline char const* CommunicationModeName(CommunicationMode mode)
{
    switch (mode)
    {
    case CommunicationMode::TwoSided:
        return "p2p (two-sided MPI)";
    case CommunicationMode::Rdma:
        return "rdma";
    case CommunicationMode::Auto:
    default:
        return "auto (rdma, falling back to p2p)";
    }
}

} // namespace imc_postprocess_tde

#ifdef RICH_MPI

#include <memory>
#include <mpi.h>

#include "source/3D/elementary/Vector3D.hpp"
#include "source/3D/tessellation/Tessellation3D.hpp"
#include "source/monte/manager/MonteCarloConfig.hpp"
#include "source/monte/manager/MonteCarloManagerFactory.hpp"
#include "source/monte/manager/communication/CommunicationEngine.hpp"

namespace imc_postprocess_tde
{

inline STORM::ManagerType ToManagerType(CommunicationMode mode)
{
    switch (mode)
    {
    case CommunicationMode::TwoSided:
        return STORM::ManagerType::P2P;
    case CommunicationMode::Rdma:
        return STORM::ManagerType::RDMA;
    case CommunicationMode::Auto:
    default:
        return STORM::ManagerType::Auto;
    }
}

//! \brief Build the particle-exchange engine requested by the configuration.
//! \param tess Tessellation the engine exchanges over
//! \param monteCarloConfig Manager configuration
//! \param mode Requested communication mode
//! \param comm MPI communicator
//! \return Communication engine
inline std::unique_ptr<STORM::CommunicationEngine<Vector3D>>
MakeCommunicationEngine(Tessellation3D const& tess,
                        MonteCarloConfig const& monteCarloConfig,
                        CommunicationMode mode,
                        MPI_Comm comm)
{
    return STORM::CreateCommunicationEngine<Vector3D, Tessellation3D>(
        tess, ToManagerType(mode), STORM::RDMAEngine::Auto, monteCarloConfig, comm);
}

} // namespace imc_postprocess_tde

#endif // RICH_MPI

#endif // POST_PROCESS_COMMUNICATION_HPP
