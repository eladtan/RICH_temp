#ifndef IMC_POSTPROCESS_TDE_FORWARD_CALCULATION_HPP
#define IMC_POSTPROCESS_TDE_FORWARD_CALCULATION_HPP

#include "postprocess_config.hpp"
#include "postprocess_runtime.hpp"

namespace imc_postprocess_tde {

ForwardPostprocessResult RunForwardPostprocess(Config const& cfg, PostprocessRuntime& runtime);

} // namespace imc_postprocess_tde

#endif // IMC_POSTPROCESS_TDE_FORWARD_CALCULATION_HPP
