#include "source/3D/radiation/postprocess/IMCPostProcess.hpp"

#include <utility>

int main(int argc, char* argv[])
{
    PostProcessIMC::PostProcessScenario scenario =
        PostProcessIMC::MakeStaSnapshotScenario("tde");

    scenario.defaults.input.snapshot =
        "/home/elads/TDEMG/R0.47M0.5BH1e+06beta1S50n1.5Compton/snap_full_136.h5";
    scenario.defaults.input.multigroupOpacityDirectory =
        "/home/elads/RICH/data/STA/MG/";
    scenario.defaults.input.eosDirectory = "/home/elads/RICH/data/EOS/";
    scenario.defaults.output.stem = "tde_postprocess_output";
    scenario.defaults.observer.radius = 5e14;
    scenario.defaults.observer.count = 256;

    return PostProcessIMC::RunPostProcessMain(argc, argv, std::move(scenario));
}
