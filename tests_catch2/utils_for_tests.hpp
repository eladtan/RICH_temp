#ifndef UTILS_FOR_TESTS_
#define UTILS_FOR_TESTS_

#include <vector>
#include <stdexcept>
#include <sstream>
#include <type_traits>
#include <json/json.h>
#include <filesystem>
#include <memory>
#include <tuple>
#include <utility>
#include <fstream>
#include <algorithm>
#include "source/mpi/serialize/mpi_commands.hpp"
#include "source/mpi/types.h"
#include "source/newtonian/three_dimensional/computational_cell.hpp"
#include "source/3D/tessellation/Tessellation3D.hpp"
#include <type_traits>

#ifdef RICH_MPI
#include <mpi.h>
#endif

namespace utils_for_tests {

/// @brief A structure that associates a name with a vector of data.
/// @tparam T Type of elements in the vector.
template<typename T>
struct named_vector {
    std::string name;  ///< The name/key for this vector.
    std::vector<T> vec;  ///< The vector of data.
};

/// @brief Factory function to create a named_vector from a name and vector.
/// @tparam T Type of elements in the vector.
/// @param name The name to associate with the vector.
/// @param vec The vector of data (will be moved).
/// @return A named_vector containing the provided name and vector.
template<typename T>
[[nodiscard]] 
named_vector<T> make_named_vector(std::string name, std::vector<T> vec) {
    return {std::move(name), std::move(vec)};
}

/// @brief Factory function to create a named_vector from a name and vector.
/// @tparam T Type of elements in the vector.
/// @param name The name to associate with the vector.
/// @param init_list The init_list of data (will be moved).
/// @return A named_vector containing the provided name and vector.
template<typename T>
[[nodiscard]] 
named_vector<T> make_named_vector(std::string name, std::initializer_list<T> init_list){
    return {std::move(name), std::vector<T>(init_list)};
}


// only for arithmetic types
/**
 * @brief Extracts a named_vector of arithmetic data members from a vector of ComputationalCell3D objects.
 * 
 * @tparam T The arithmetic type of the data member to extract (e.g., double, int).
 * @param name The name to be associated with the resultant named_vector.
 * @param cells The vector of ComputationalCell3D from which to extract data.
 * @param ptr_to_cell_data Pointer-to-member specifying which field to extract from each cell.
 * @return named_vector<T> A named_vector with the provided name and the extracted values.
 * 
 * @note Example usage:
 *   auto densities = extract_data_from_cells("density", cells, &ComputationalCell3D::density);
 */
template<
    typename T,
    std::enable_if_t<std::is_arithmetic_v<T>, int> = 0
>
[[nodiscard]]
named_vector<T> extract_data_from_cells(
    std::string const& name,
    std::vector<ComputationalCell3D> const& cells,
    T ComputationalCell3D::* const ptr_to_cell_data
){
    std::vector<T> vec(cells.size(), T{});

    for(std::size_t i = 0; i < cells.size(); ++i){
        vec[i] = cells[i].*ptr_to_cell_data;
    }

    return make_named_vector(name, vec);
}

/**
 * @brief Extracts named_vectors of the x, y, and z components of a Vector3D data member from a vector of ComputationalCell3D objects.
 * 
 * @param name The base name to associate with each component vector ("_x", "_y", "_z" will be appended).
 * @param cells The vector of ComputationalCell3D objects from which to extract vector components.
 * @param ptr_to_cell_data Pointer-to-member indicating which Vector3D field to extract (e.g., &ComputationalCell3D::velocity).
 * @return std::tuple<named_vector<double>, named_vector<double>, named_vector<double>> 
 *         Tuple of named_vectors holding the x, y, and z components respectively.
 * 
 * @note Example usage:
 *   auto [vx, vy, vz] = extract_data_from_cells("velocity", cells, &ComputationalCell3D::velocity);
 *   // vx.name == "velocity_x", vx.vec holds all x components
 */
[[nodiscard]]
std::tuple<
    named_vector<double>,
    named_vector<double>,
    named_vector<double>
> extract_data_from_cells(
    std::string const& name,
    std::vector<ComputationalCell3D> const& cells,
    Vector3D ComputationalCell3D::* const ptr_to_cell_data
);

/**
 * @brief Extract x, y, z cell center coordinates from a Tessellation3D.
 * @param tess The tessellation to extract from.
 * @return Tuple of named_vectors for x, y, and z.
 */
[[nodiscard]]
std::tuple<
    named_vector<double>,
    named_vector<double>,
    named_vector<double>
>
extract_center_of_mass(Tessellation3D const& tess);

namespace mpi {

/// @brief The MPI root rank 
static constexpr rank_t rank_root = 0;

/// @brief Gets the MPI rank of the current process.
/// @return The MPI rank (0-based) of the current process in MPI_COMM_WORLD.
///         Returns 0 if MPI is not enabled (RICH_MPI not defined).
int get_mpi_rank();

/// @brief Gets the total number of MPI processes in the communicator.
/// @return The size of MPI_COMM_WORLD (total number of processes).
///         Returns 1 if MPI is not enabled (RICH_MPI not defined).
int get_mpi_world_size();

/**
 * @brief Collects elements from all MPI ranks into a single vector on the root process. 
 * Falls back to returning the input vector if MPI is not enabled.
 * 
 * @note T must be serializable (std::string is not supported).
 */
template<typename T>
std::vector<T> mpi_gather_vector(std::vector<T> const& vector){
    static_assert(not std::is_same_v<T, std::string>, "MPI_Gatherv_serializable does not know how to serialize strings!");

    std::vector result{vector};

    #ifdef RICH_MPI
    result = MPI_Gatherv_serializable(vector, rank_root, MPI_COMM_WORLD);
    #endif

    return result;
}

/**
 * @brief Synchronizes all processes in the MPI communicator.
 */
void mpi_barrier();

/// @brief Base test fixture that initializes MPI rank and size.
struct RichBasicTestFixture {
    int const comm_size; ///< Total MPI processes
    int const rank;      ///< Current MPI rank
    
    protected:
    RichBasicTestFixture();
};

/// @brief Test fixture for single-process (non-MPI) tests; skips tests on non-root ranks.
struct RichNoMpiTestFixture : public RichBasicTestFixture {
    RichNoMpiTestFixture();
};

/// @brief Test fixture for multi-process (MPI) tests.
struct RichMpiFixture : public RichBasicTestFixture {
    RichMpiFixture();
};

} // namespace mpi

/// @brief Extracts the size of the first object in a variadic pack.
/// @tparam First Type of the first object (must have a size() method).
/// @tparam Rest Types of the remaining objects (ignored).
/// @param vec The first object in the pack.
/// @param ... Remaining objects (unused).
/// @return The size of the first object (vec.size()).
template<typename First, typename... Rest>
std::size_t size_of_first_in_pack(First const& vec, Rest const& ...){
    return vec.size();
}

/// @brief Checks if all provided objects have the same size.
/// @tparam Sizeables Types of objects to check (must have a size() method).
/// @param objs Variadic pack of objects to compare sizes.
/// @return true if all objects have the same size, false otherwise.
/// @note Requires at least one argument (enforced by static_assert).
template<typename... Sizeables>
bool has_same_size(Sizeables const&... objs){
    static_assert(sizeof...(Sizeables) > 0, "has_same_size requires at least one argument");

    auto const size = size_of_first_in_pack(objs...);
    return ((size == objs.size()) && ...);
}

/// @brief Zips multiple vectors into a vector of tuples.
/// @tparam Ts Types of elements in each vector (variadic template).
/// @param vectors Variadic pack of vectors to zip together.
/// @return A vector of tuples, where each tuple contains one element from each input vector at the same index.
/// @throws std::runtime_error If the input vectors have different sizes.
/// @note Requires at least 2 vectors (enforced by static_assert).
template<typename... Ts>
std::vector<std::tuple<Ts...>> zip(std::vector<Ts> const&... vectors){
    static_assert(sizeof...(Ts) > 1, "zip requires at least 2 vectors");

    if(not has_same_size(vectors...)){
        throw std::runtime_error("zip: vector sizes are not equal");
    }

    std::vector<std::tuple<Ts...>> result;
    
    auto const size = size_of_first_in_pack(vectors...);
    result.reserve(size);

    for(std::size_t i=0; i < size; ++i){
        result.emplace_back(vectors[i] ...);
    }

    return result;
}

/// @brief Reserves capacity for all vectors in a tuple of vectors.
/// @tparam Ts Types of elements in each vector (variadic template).
/// @param size The capacity to reserve for each vector.
/// @param vectors Tuple of vectors to reserve capacity for (modified in place).
template<typename... Ts>
void reserve_vector_pack(
    std::size_t size, 
    std::tuple<std::vector<Ts>...>& vectors
){    
    std::apply(
        [size](auto&... vecs){
            (vecs.reserve(size), ...);
        },
        vectors
    );
}

/// @brief Pushes back elements from a tuple into corresponding vectors in a tuple of vectors.
/// @tparam Ts Types of elements in each vector (variadic template).
/// @tparam Is Index sequence for expanding the variadic template.
/// @param tuple_of_vectors Tuple of vectors to push elements into (modified in place).
/// @param tuple Tuple containing elements to push back (one element per vector).
/// @param std::index_sequence<Is...> Index sequence for template expansion.
/// @note This is an internal helper function used by unzip. Expands to push_back operations
///       for each corresponding element-vector pair.
template <typename... Ts, std::size_t... Is>
void push_back_tuple(
    std::tuple<std::vector<Ts>...>& tuple_of_vectors,
    std::tuple<Ts...> const& tuple,
    std::index_sequence<Is...>
) {
    // expands to:
    // get<0>(tuple_of_vectors).push_back(get<0>(tuple));
    // get<1>(tuple_of_vectors).push_back(get<1>(tuple)); ...
    (std::get<Is>(tuple_of_vectors).push_back(std::get<Is>(tuple)), ...);
}

/// @brief Unzips a vector of tuples into a tuple of vectors (inverse of zip).
/// @tparam Ts Types of elements in each tuple (variadic template).
/// @param vec Vector of tuples to unzip.
/// @return A tuple of vectors, where each vector contains elements from the corresponding position
///         in all input tuples.
/// @note The i-th vector in the result contains all i-th elements from the input tuples.
template<typename... Ts>
std::tuple<std::vector<Ts>...> unzip(std::vector<std::tuple<Ts...>> const& vec){

    std::tuple<std::vector<Ts>...> result{
        std::vector<Ts>{}...
    };
    
    auto const size = vec.size();
    reserve_vector_pack(size, result);
    for(auto const& tup : vec){
        push_back_tuple(result, tup, std::index_sequence_for<Ts...>{});
    }
    
    return result;
}

/// @brief Sorts a tuple of vectors by the element at the specified index.
/// @tparam index The index of the element to sort by.
/// @tparam Ts Types of elements in the vectors (variadic template).
/// @param vectors Tuple of vectors to sort.
/// @return A tuple of sorted vectors.
template<std::size_t index, typename... Ts>
std::tuple<std::vector<Ts>...> sort_vectors_by_index( 
    std::vector<Ts> const&... vectors
){
    auto vector_of_tuples = zip(vectors...);
    
    std::sort(
        std::begin(vector_of_tuples),
        std::end(vector_of_tuples),
        [](auto const& tup1, auto const tup2){
            return std::get<index>(tup1) < std::get<index>(tup2);
        }
    );

    return unzip(vector_of_tuples);
}

namespace json {

/// @brief Validates that a file path has a .json extension.
/// @param path_to_json The file path to validate.
/// @throws std::runtime_error If the file extension is not .json.
inline void assert_json_extension(std::filesystem::path const& path_to_json){
    if(path_to_json.extension() != ".json"){
        std::ostringstream oss{"assert_json_extension: file extension is not .json!. path_to_file: "};
        
        oss << path_to_json
            << ", extension: " << path_to_json.extension();
        
        throw std::runtime_error(oss.str());
    }
}

/// @brief Converts a C++ vector to a JSON array value.
/// @tparam T Type of elements in the vector (must be convertible to Json::Value).
/// @param vec Input vector to convert.
/// @return A Json::Value containing a JSON array with elements from the input vector.
template<typename T>
Json::Value vector_to_json_array(std::vector<T> const& vec){
    Json::Value result(Json::arrayValue);

    for(auto const& element : vec){
        result.append(Json::Value(element));
    }

    return result;
}

/// @brief Converts a JSON array value to a C++ vector.
/// @tparam T Type to convert each JSON array element to.
/// @param arr JSON array value to convert.
/// @return A vector containing elements converted from the JSON array.
/// @throws std::runtime_error If the input Json::Value is not an array.
template<typename T>
std::vector<T> json_array_to_vector(Json::Value const& arr){
    if(!arr.isArray()){
        throw std::runtime_error("json_array_to_vector: Value is not an array");
    }

    std::vector<T> result{};
    result.reserve(arr.size());
    for(int i=0; i < arr.size(); ++i){
        auto const& element = arr[i];
        result.push_back(element.as<T>());
    } 

    return result;
}

/// @brief Adds a named vector to a JSON root object as a key-value pair.
/// @tparam T Type of elements in the vector.
/// @param root The JSON root object to add the vector to (modified in place).
/// @param vec The named vector to add to the JSON root.
template<typename T>
void add_vector_to_json_root(Json::Value& root, named_vector<T> const& vec){
    Json::Value arr = vector_to_json_array(vec.vec);
    root[vec.name] = arr;
}

/// @brief Saves multiple named vectors to a JSON file.
/// @tparam Ts Types of elements in the vectors to save (variadic template).
/// @param path_to_json Path to the output JSON file.
/// @param vectors_to_save One or more named_vector objects to save to the file.
/// @throws std::runtime_error If the file extension is not .json, if the file cannot be opened,
///         or if directory creation fails.
/// @note Creates parent directories if they don't exist. The JSON file is formatted with 2-space indentation.
template<typename... Ts>
void save_data_to_json_file(std::filesystem::path const& path_to_json, named_vector<Ts> const& ... vectors_to_save){
    assert_json_extension(path_to_json);

    std::filesystem::create_directories(path_to_json.parent_path());

    std::ofstream ofs{path_to_json};
    if(!ofs){
        throw std::runtime_error("save_data_to_json_file: Cannot open file for writing: " + path_to_json.string());
    }

    Json::Value root;
    (add_vector_to_json_root(root, vectors_to_save), ...);

    Json::StreamWriterBuilder builder;
    builder["indentation"] = "  ";
    std::unique_ptr<Json::StreamWriter> writer(builder.newStreamWriter());
    writer->write(root, &ofs);
    ofs << std::endl;
}

/// @brief Internal helper function that constructs a tuple of named_vector objects from JSON data.
/// @tparam Ts Types of elements in the vectors to extract (variadic template).
/// @tparam Is Index sequence for expanding the variadic template.
/// @param root The JSON root object containing the data.
/// @param data_names Vector of names/keys to look up in the JSON root.
/// @param std::index_sequence<Is...> Index sequence for template expansion.
/// @return A tuple of named_vector objects, one for each type in Ts.
/// @note This is an internal implementation detail used by load_data_from_json.
template<typename... Ts, std::size_t... Is>
std::tuple<named_vector<Ts>...>
make_tuple_from_json(
    Json::Value const& root,
    std::vector<std::string> const& data_names,
    std::index_sequence<Is...>){

    // Expands to something like:
    // named_vector<T0>{ data_names[0], json_array_to_vector<T0>(root[data_names[0]]) },
    // named_vector<T1>{ data_names[1], json_array_to_vector<T1>(root[data_names[1]]) }, ...
    return std::tuple<named_vector<Ts>...>{
        named_vector<Ts>{
            data_names[Is],
            json_array_to_vector<Ts>(root[data_names[Is]])
        }...
    };
}

/// @brief Loads multiple named vectors from a JSON file.
/// @tparam Ts Types of elements in the vectors to load (variadic template).
/// @param path_to_json Path to the input JSON file.
/// @param data_names Vector of names/keys to look up in the JSON file.
///                   Must have the same size as the number of types in Ts.
/// @return A tuple of named_vector objects, one for each type in Ts.
/// @throws std::runtime_error If the file extension is not .json, if the file doesn't exist,
///         if the file cannot be opened, if JSON parsing fails, or if the number of data_names
///         doesn't match the number of template parameters.
template<typename... Ts>
std::tuple<named_vector<Ts>...> 
load_data_from_json(
    std::filesystem::path const& path_to_json, 
    std::vector<std::string> const& data_names) {
    
    assert_json_extension(path_to_json);
    if(not std::filesystem::exists(path_to_json)){
        throw std::runtime_error("load_data_from_json: file doesn't exist: " + path_to_json.string());
    }

    if(sizeof...(Ts) != data_names.size()){
        std::ostringstream oss{"load_data_from_json: expected: "};
        oss << sizeof...(Ts) << "data_names but got: " << data_names.size() << " instead.";
        

        throw std::runtime_error(oss.str());
    }

    std::ifstream ifs(path_to_json);
    if(!ifs){
        throw std::runtime_error(
            "load_data_from_json: Cannot open file for reading: " +
            path_to_json.string()
        );
    }
    Json::CharReaderBuilder rbuilder;
    rbuilder["collectComments"] = false;

    std::string errs;
    Json::Value root;
    if(!Json::parseFromStream(rbuilder, ifs, &root, &errs)){
        throw std::runtime_error("load_data_from_json: parse error: " + errs);
    }

    return make_tuple_from_json<Ts...>(root, data_names, std::index_sequence_for<Ts...>{});

}

} // namespace json

} // namespace utils_for_tests
#endif