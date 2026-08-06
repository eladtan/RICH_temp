#include <filesystem>
#include "../densmore2012_interface_test.hpp"

int main(int argc, char **argv)
{
    densmore2012_interface_test::RunOptions options;
    options.ddmcInterfaceDiagnostics = true;
    return densmore2012_interface_test::Run<true>(
        argc, argv,
        std::filesystem::path(__FILE__).parent_path().string(),
        "desmore2012_interface_ddmc",
        options);
}
