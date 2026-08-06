#include "source/3D/radiation/postprocess/IMCPostProcess.hpp"

#include <utility>

int main(int argc, char* argv[])
{
    PostProcessIMC::PostProcessScenario scenario =
        PostProcessIMC::MakeStaSnapshotScenario("tde");

    scenario.defaults.output.stem = "tde_postprocess_output";
    scenario.defaults.observer.radius = 5e14;
    scenario.defaults.observer.count = 256;

    return PostProcessIMC::RunPostProcessMain(argc, argv, std::move(scenario));
}
