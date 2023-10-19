#ifndef UNSORTED_UNIQUE
#define UNSORTED_UNIQUE

#include <vector>

template <class T>
std::vector<T> unsorted_unique(std::vector<T> const& vec) {
    std::size_t const vec_size = vec.size();

    std::vector<bool> isDeleted(vec_size, false);

    std::size_t size_deleted = 0;
    for(std::size_t i=0; i<vec_size; ++i){
        for(std::size_t j=i+1; j<vec_size; ++j){
            if(isDeleted[j]) continue;

            bool const res = (vec[i] == vec[j]);
            
            isDeleted[j] = res;
            if(res) ++size_deleted;
        }
    }

    std::vector<T> new_vec(vec_size - size_deleted, T());

    for(std::size_t i=0, j=0; i<vec_size; ++i){
        if(isDeleted[i]) continue;

        new_vec[j] = vec[i];
        ++j;
    }

    return new_vec;
}

#endif // UNSORTED_UNIQUE