#ifndef UTILS_FOR_TESTS_
#define UTILS_FOR_TESTS_

#include <vector>
#include <stdexcept>
#include <sstream>

namespace utils_for_tests {
    
int get_mpi_rank();

int get_mpi_world_size();

template<typename T, typename U>
std::vector<std::pair<T, U>> zip(std::vector<T> const& vec_1, std::vector<U> const& vec_2){
    
    if(vec_1.size() != vec_2.size()){
        std::ostringstream oss{"zip: vector sizes are not equal, vec_1.size() = "};
        oss << vec_1.size()
            << ", vec_2.size() = " << vec_2.size();
        throw std::runtime_error(oss.str());
    }
    
    std::vector<std::pair<T, U>> result;
    auto const size = vec_1.size();
    result.reserve(size);

    for(std::size_t i=0; i < size; ++i){
        result.push_back(std::pair(vec_1[i], vec_2[i]));
    }

    return result;
}

template<typename T, typename U>
std::pair<std::vector<T>, std::vector<U>> unzip(std::vector<std::pair<T, U>> const& vec){
    
    std::vector<T> vec_1;
    std::vector<U> vec_2;

    auto const size = vec.size();
    vec_1.reserve(size);
    vec_2.reserve(size);

    for(auto const& [elem_T, elem_U] : vec){
        vec_1.push_back(elem_T);
        vec_2.push_back(elem_U);
    }

    return {vec_1, vec_2};
}

}
#endif