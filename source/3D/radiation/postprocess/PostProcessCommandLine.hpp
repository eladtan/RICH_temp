#ifndef RICH_IMC_POSTPROCESS_COMMAND_LINE_HPP
#define RICH_IMC_POSTPROCESS_COMMAND_LINE_HPP

#include "IMCPostProcess.hpp"

#include <string>
#include <utility>
#include <vector>

namespace PostProcessIMC
{

enum class CommandLineAction
{
    Run,
    Help,
    PrintEffectiveConfig,
    Error
};

CommandLineAction ParseCommandLine(
    int argc, char* argv[], PostProcessConfig& config, int rank);

void PrintCommandLineHelp(PostProcessConfig const& defaults, int rank);
void PrintEffectiveConfig(PostProcessConfig const& config, int rank);
std::vector<std::pair<std::string, std::string>> EffectiveConfigEntries(
    PostProcessConfig const& config);

} // namespace PostProcessIMC

#endif
