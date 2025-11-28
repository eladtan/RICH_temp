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

/**
 * @brief Configuration for test infrastructure
 */
struct TestsConfig {
    SnapshotDataConfig _mode = SnapshotDataConfig::Compare;
    
    double _relative_tolerance_compare = 100.0*std::numeric_limits<double>::epsilon();
    
    std::filesystem::path _golden_dir = std::filesystem::path{__FILE__}.parent_path() / "golden_dir";

    std::string repr() const;

    /**
     * @brief Singleton instance accessor (Scott Meyers' pattern).
     * @return Reference to the global TestsConfig instance.
     */
    static TestsConfig& instance();

    /**
     * @brief Returns the current snapshot data mode (regenerate or compare).
     * @return The configured SnapshotDataConfig mode.
     */
    static SnapshotDataConfig mode();

    /**
     * @brief Returns the relative tolerance used for floating-point comparisons.
     * @return The relative tolerance value.
     */
    static double relative_tolerance_compare();

    /**
     * @brief Returns the path to the directory containing golden state data.
     * @return The configured golden directory as a std::filesystem::path.
     */
    static std::filesystem::path golden_dir();
};

/**
 * @brief Parses command-line arguments to adjust test configuration settings (e.g., mode/tolerance/golden dir).
 */
void parseTestsConfigArguments(int& argc, char**& argv);

/**
 * @brief Asserts that the required --order argument is present in the provided command-line arguments.
 */
void assert_order_arg_is_given(int const& argc, char** const& argv);

}; // namespace tests config
#endif