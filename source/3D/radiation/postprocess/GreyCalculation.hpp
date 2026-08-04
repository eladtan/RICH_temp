#ifndef IMC_POSTPROCESS_TDE_GREY_CALCULATION_HPP
#define IMC_POSTPROCESS_TDE_GREY_CALCULATION_HPP

#include "PostProcessConfig.hpp"
#include "PostProcessRuntime.hpp"

namespace imc_postprocess_tde {

ForwardPostprocessResult RunGreyPostprocess(
    Config const& cfg,
    PostprocessRuntime& runtime,
    ForwardPostprocessResult const& forwardResult);

} // namespace imc_postprocess_tde

#endif // IMC_POSTPROCESS_TDE_GREY_CALCULATION_HPP
