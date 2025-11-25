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

namespace utils_for_tests {
    
int get_mpi_rank();

int get_mpi_world_size();

template<typename First, typename... Rest>
std::size_t size_of_first_in_pack(First const& vec, Rest const& ...){
    return vec.size();
}

template<typename... Sizeables>
bool has_same_size(Sizeables const&... objs){
    static_assert(sizeof...(Sizeables) > 0, "has_same_size requires at least one argument");

    auto const size = size_of_first_in_pack(objs...);
    return ((size == objs.size()) && ...);
}

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

namespace json {

    for(auto const& [elem_T, elem_U] : vec){
        vec_1.push_back(elem_T);
        vec_2.push_back(elem_U);
    }

    return {vec_1, vec_2};
}

}
#endif