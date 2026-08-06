#include <filesystem>
#include "../densmore2012_interface_test.hpp"

int main(int argc, char **argv)
{
    densmore2012_interface_test::RunOptions options;
    options.useCenteredInterfaceMesh = true;
    return densmore2012_interface_test::Run<false>(
        argc, argv,
        std::filesystem::path(__FILE__).parent_path().string(),
        "desmore2012_interface_mc_centered",
        options);
}
