#ifndef RICH_IMC_POSTPROCESS_RESULT_WRITER_HPP
#define RICH_IMC_POSTPROCESS_RESULT_WRITER_HPP

#include "IMCPostProcess.hpp"
#include "PostProcessConfig.hpp"
#include "PostProcessRuntime.hpp"

namespace imc_postprocess_tde
{

Config MakePassOutputConfig(Config const& finalConfig);

void WriteGreyPassScratch(
    Config const& config,
    SphericalObserver const& observer,
    SphericalObserver::Diagnostics const& diagnostics);

void FinalizeResultBundle(
    Config const& finalConfig,
    Config const& passConfig,
    PostProcessIMC::PostProcessConfig const& effectiveConfig,
    std::string const& scenarioName,
    PostprocessRuntime const& runtime,
    ForwardPostprocessResult const& forward,
    ForwardPostprocessResult const& grey);

} // namespace imc_postprocess_tde

#endif
