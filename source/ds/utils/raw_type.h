#ifndef RAW_TYPE_H
#define RAW_TYPE_H

// see here: https://stackoverflow.com/a/27567052
template<class ...>
using void_t = void;

template<class T, class = void>
struct is_raw_type_defined { 
    using type = T;
};

template<class T>
struct is_raw_type_defined<T, void_t<typename T::Raw_type>> { 
    using type = typename T::Raw_type;
};

#endif // RAW_TYPE_H