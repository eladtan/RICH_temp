#ifndef CONFIG_TESTS_HPP_
#define CONFIG_TESTS_HPP_

#include <string>
#include <limits>
#include <filesystem>

namespace tests_config {

enum class SnapshotDataConfig {
    Regenerate,
    Compare
};

struct TestsConfig {
    SnapshotDataConfig _mode = SnapshotDataConfig::Compare;
    double _relative_tolerance_compare = 100.0*std::numeric_limits<double>::epsilon();
    
    std::filesystem::path _golder_dir = std::filesystem::path{__FILE__}.parent_path() / "golder_dir";

    std::string repr() const;
    // Scott Meyers' singleton pattern
    static TestsConfig& instance();
    
    static SnapshotDataConfig mode();
    static double relative_tolerance_compare();
    static std::filesystem::path golder_dir(); 
};

void parseTestsConfigArguments(int& argc, char**& argv);

}; // namespace tests config
#endif