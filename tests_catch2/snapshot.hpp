#ifndef SNAPSHOT_HPP_
#define SNAPSHOT_HPP_

#include <string>
#include <fstream>
#include <iostream>
#include <sstream>
#include <filesystem>
#include <cmath>
#include "config_tests.hpp"
#include <optional>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/catch_test_macros.hpp>
#include "utils_for_tests.hpp"

namespace snapshot {

using utils_for_tests::named_vector;
namespace mpi = utils_for_tests::mpi;
namespace tc = tests_config;

/**
 * @brief The SnapShot class manages saving and comparing simulation data "snapshots" for unit/integration tests.
 * 
 * Usage pattern:
 *   - In the test, construct a SnapShot 
 *   - Use CompareOrSaveGather() to either:
 *      - Compare generated results to a stored snapshot (fail if the output is not sufficiently close).
 *      - Or save a new snapshot if one does not exist yet.
 * 
 * Typically, in most test modes, this will be run with a regression snapshot mode that controls whether to check or save snapshots.
 * 
 * @see CompareOrSaveGather()
 */
class SnapShot {
public:
    SnapShot(std::optional<std::string> test_name_ = std::nullopt);

    /**
     * @brief Gathers and compares (or saves) a collection of named_vectors representing test data against a saved snapshot.
     * 
     * Data is gathered across MPI ranks if relevant. Returning true means data matched or was saved; false indicates a mismatch.
     *
     * @tparam Ts The types of data vectors to compare/save (typically arithmetic types).
     * @param data_file_name Optional alternative filename to save the snapshot under this test.
     * @param ID A named_vector holding the unique IDs for the data rows.
     * @param data One or more named_vectors holding measured test data.
     * @return True if comparison succeeded (or data was saved), false on failure.
     */
    template <typename... Ts>
    [[nodiscard]]
    bool CompareOrSaveGather(
        std::optional<std::string> const& data_file_name,
        named_vector<std::size_t> ID,
        named_vector<Ts>... data
    ) const;

    std::string const test_name;

    private:
        std::filesystem::path test_dir() const;
        std::filesystem::path test_data_path(
            std::optional<std::string> const& data_file_name = std::nullopt
        ) const;
        
};

/**
 * @brief Gathers a vector from all MPI ranks onto the root process.
 * @tparam T Type of vector elements.
 * @param vector The vector to be gathered (modified in place on root).
 */
template<typename T>
void Gather_single(std::vector<T>& vector){
    auto const tmp_vector = mpi::mpi_gather_vector(vector);

    if(mpi::get_mpi_rank() == mpi::rank_root){
        vector = tmp_vector;
    }
}

/**
 * @brief Gathers one or more named_vectors from all MPI ranks onto the root.
 * @tparam Ts Types of vectors inside named_vectors. Must provide at least one.
 * @param vectors The named_vectors to be gathered (modified in place on root).
 */
template <typename... Ts>
void Gather(named_vector<Ts>&... vectors){
    static_assert(sizeof...(Ts) > 0, "Gather: requires at least one vector");
    
    (Gather_single(vectors.vec), ...);
}

/**
 * @brief Checks if two values are close enough within a relative tolerance.
 * @tparam T Type of values to compare.
 * @param v1 The first value.
 * @param v2 The second value.
 * @param rel_tol The relative tolerance.
 * @return True if the values are close enough, false otherwise.
 */
template<typename T>
[[nodiscard]] 
bool close_enough(T const v1, T const v2, double const rel_tol){
    return std::abs(v1 - v2) <= rel_tol * 0.5 * (std::abs(v1) + std::abs(v2));
}

/**
 * @brief Compares two vectors element-wise within a relative tolerance.
 * @tparam T Type of vector elements.
 * @param description A description string for error reporting.
 * @param v1 The first vector.
 * @param v2 The second vector.
 * @param rel_tol The relative tolerance.
 * @return True if all elements are close enough, false otherwise.
 */
template <typename T>
[[nodiscard]] 
bool compare_vectors_check(
    std::string const& description, 
    std::vector<T> const& v1, 
    std::vector<T> const& v2, 
    double const rel_tol){

    if(v1.size() != v2.size()){
        throw std::runtime_error("compare_vectors_check: vectors have different sizes");
    }
    
    auto const size = v1.size();
    auto number_of_close_enough = 0;

    for(std::size_t i=0; i < v1.size(); ++i){
        if(close_enough(v1[i], v2[i], rel_tol)) ++number_of_close_enough;
    }

    std::ostringstream oss;
    oss << description << ": number of elements that are close up to relative tolerance " << rel_tol
        << " is " << number_of_close_enough << "/" << size;

    INFO(oss.str());
    CHECK(number_of_close_enough == size);

    return number_of_close_enough == size;
}


template<typename... Ts>
bool SnapShot::CompareOrSaveGather(
    std::optional<std::string> const& data_file_name,
    named_vector<std::size_t> ID, 
    named_vector<Ts>... data
) const {
    Gather(ID, data...);

    if(mpi::get_mpi_rank() == mpi::rank_root){
        auto const data_path = test_data_path(data_file_name);

        bool const regenerate = tc::TestsConfig::mode() == tc::SnapshotDataConfig::Regenerate;

        if(regenerate || not std::filesystem::exists(data_path)){
            utils_for_tests::json::save_data_to_json_file(
                data_path,
                ID,
                data...
            );

            if(regenerate){
                SKIP("Regenerate test \"" + test_name + " \" data at: " + data_path.string());
                return true;
            } else {
                FAIL("Created new golden state data for \"" + test_name + "\" at : " + data_path.string() + ". Re-run tests (or use --regen to update intentionally).");
                return false;
            }
        } else {
            std::vector<std::string> const data_names{data.name...};

            auto const expected = utils_for_tests::json::load_data_from_json<Ts...>(data_path, data_names);

            bool const compare_result = std::apply(
                [&](auto const& ... expected_vectors){
                    return (compare_vectors_check(
                        "Compare " + data.name,
                        data.vec,
                        expected_vectors.vec,
                        tc::TestsConfig::relative_tolerance_compare()
                    ) && ...);
                },
                expected
            );
            
            bool const compare_names = std::apply(
                [&](auto const& ... expected_vectors){
                    return ((data.name == expected_vectors.name) && ...);
                },
                expected
            );
            
            std::ostringstream oss;
            oss << "data names: ";
            (((oss << data.name) << ", "), ...) << "\n";
            oss << "expected names: ";
            std::apply(
                [&oss](auto const&... expected_vectors){
                    (((oss << expected_vectors.name) << ", "), ...);
                },
                expected
            );

            INFO(oss.str());
            CHECK(compare_result);
            CHECK(compare_names);

            return compare_names && compare_result;
        }
    }

    return true;
}

}

#endif // SNAPSHOT_HPP_