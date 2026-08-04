#ifndef IMC_POSTPROCESS_TDE_GREY_CALCULATION_HPP
#define IMC_POSTPROCESS_TDE_GREY_CALCULATION_HPP

#include "postprocess_config.hpp"
#include "postprocess_runtime.hpp"

namespace imc_postprocess_tde {

void RunGreyPostprocess(
    Config const& cfg,
    PostprocessRuntime& runtime,
    ForwardPostprocessResult const& forwardResult);

} // namespace imc_postprocess_tde

#endif // IMC_POSTPROCESS_TDE_GREY_CALCULATION_HPP
