#include "snapshot.hpp"

#include <catch2/internal/catch_context.hpp>
#include <catch2/interfaces/catch_interfaces_capture.hpp>
#include <catch2/catch_test_macros.hpp>
#include "utils_for_tests.hpp"
#include <json/json.h>
#include "source/mpi/serialize/mpi_commands.hpp"
#include <algorithm>
#include <json/json.h>

namespace {
    // Catch2 documentation says it is depracated!!!!!!
    // Might break when changing versions
    std::string get_current_test_name() {
        return Catch::getResultCapture().getCurrentTestName();
    }
}

namespace snapshot {

namespace fs = std::filesystem;
using namespace utils_for_tests;

SnapShot::SnapShot(std::optional<std::string> test_name_)
: test_name{test_name_? test_name_.value() : get_current_test_name()}
{
    INFO("Test Name: " + test_name);    
}

fs::path SnapShot::test_dir() const{
    return tests_config::TestsConfig::golder_dir() / test_name;
}

fs::path SnapShot::test_data_path(
    std::optional<std::string> const& data_file_name
) const {

    if(data_file_name) return test_dir() / (*data_file_name + ".json");

    return test_dir() / (test_name + ".json");
}

} // namespace snapshot
