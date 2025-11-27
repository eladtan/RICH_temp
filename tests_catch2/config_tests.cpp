#include "config_tests.hpp"
#include <vector>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <cstdlib>
#include <charconv>
#include <system_error>

namespace {
    [[nodiscard]] bool parse_double(std::string_view sv, double& out){
        const char* first = sv.data();
        const char* last  = sv.data() + sv.size();

        auto res = std::from_chars(first, last, out, std::chars_format::general);
        
        // success only if no error and we consumed the whole string
        return res.ec == std::errc{} && res.ptr == last;
    }
}

namespace tests_config {
    TestsConfig& TestsConfig::instance() {
        static TestsConfig config{};
        
        return config;
    }

    SnapshotDataConfig TestsConfig::mode(){
        return TestsConfig::instance()._mode;
    }

    double TestsConfig::relative_tolerance_compare(){
        return TestsConfig::instance()._relative_tolerance_compare;
    }

    std::filesystem::path TestsConfig::golden_dir(){
        return TestsConfig::instance()._golden_dir;
    }

    std::string TestsConfig::repr() const {
        std::ostringstream oss{"TestsConfig{"};

        oss << (_mode == SnapshotDataConfig::Regenerate ? "regenerate" : "compare")
            << ", rel_tol = " << _relative_tolerance_compare
            << ", golden_dir = " << golden_dir 
            << "}";

        return oss.str();
    }

    void parseTestsConfigArguments(int& argc, char**& argv) {
        TestsConfig& config = TestsConfig::instance();

        std::vector<char*> newArgs;
        newArgs.push_back(argv[0]); // Program name

        for(int i=1; i < argc; ++i){
            std::string arg = argv[i];

            if(arg == "--regen") {
                config._mode = SnapshotDataConfig::Regenerate;
                continue;
            }

            if(arg == "--rel_tol"){
                if(i+1 >= argc){
                    std::cerr << "Missing Value after --rel_tol" << std::endl;
                    exit(1);
                }

                std::string_view value{argv[++i]};
                if(not parse_double(value, config._relative_tolerance_compare)){
                    std::cerr << "Invalid value for --rel_tol: " << value << std::endl;
                    std::exit(1);
                }

                continue;
            }

            if(arg == "--golden_dir"){
                if (i+1 >= argc) {
                    std::cerr << "Missing value after --golden_dir" << std::endl;
                    std::exit(1);
                }

                config._golden_dir = std::filesystem::path(argv[++i]);
                continue;
            }

            // not our args save them for catch2
            newArgs.push_back(argv[i]);
        }

        argc = static_cast<int>(newArgs.size());
        for(int i=0; i<argc; ++i){
            argv[i] = newArgs[i];
        }
    }

    void assert_order_arg_is_given(int const& argc, char** const& argv){
        for(int i=1; i < argc; ++i){
            std::string arg = argv[i];

            if(arg == "--order"){
                if(i+1 < argc){
                    std::string value = argv[++i];
                    
                    if(value == "decl"){
                        return;
                    }
                }
            }
        }

        // RATIONAL: The default for Catch2 is to run the tests in a random order each time.
        // For MPI runs this means that the different processes run different tests thus different sections of the code which fucks up the tests
        throw std::invalid_argument("rich_tests run must ran with the command line option `--order decl`.");
    }

} // namespace tests_config