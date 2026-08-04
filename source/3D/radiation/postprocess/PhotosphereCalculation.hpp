#ifndef IMC_POSTPROCESS_TDE_PHOTOSPHERE_CALCULATION_HPP
#define IMC_POSTPROCESS_TDE_PHOTOSPHERE_CALCULATION_HPP

#include "PostProcessRuntime.hpp"

namespace imc_postprocess_tde {

SphericalObserver::PhotosphereData ComputeObserverPhotospheres(
    Config const& cfg,
    PostprocessRuntime& runtime);

} // namespace imc_postprocess_tde

#endif // IMC_POSTPROCESS_TDE_PHOTOSPHERE_CALCULATION_HPP
