#ifndef RAW_TYPE_H
#define RAW_TYPE_H

#include <type_traits> // defined `void_t`

// see here: https://stackoverflow.com/a/27567052

template<typename T, typename = void>
struct is_raw_type_defined 
{ 
    using type = T;
};

template<typename T>
struct is_raw_type_defined<T, std::void_t<typename T::Raw_type>>
{ 
    using type = typename T::Raw_type;
};

#endif // RAW_TYPE_H