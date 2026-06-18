#ifndef SERIALIZER_HPP
#define SERIALIZER_HPP

#ifdef RICH_MPI

#include <iostream>
#include <array>
#include <vector>
#include <memory>
#include <limits>
#include <cstring>
#include <cassert>
#include <boost/container/flat_set.hpp>

#include "Serializable.hpp"
#include "misc/universal_error.hpp"

#include "utils/compiler.h"

class Serializer
{
public:
    Serializer(void) = default;

    void reset(){this->internal.clear();};
    void release(){std::vector<char>().swap(this->internal);};

    inline size_t capacity() const{return this->internal.capacity();};

    char *resize(size_t size);

    template<typename T>
    size_t insert(const T &data);

    template<typename T>
    size_t insert(const std::vector<T> &data);

    template<typename U, typename V>
    size_t insert(const std::pair<U, V> &data);

    template<typename T, std::size_t N>
    inline size_t insert(const std::array<T, N> &data){return this->insert_array(data.data(), N);};

    template<typename T, template<typename...> class VectorContainer, typename... Ts>
    size_t insert(const VectorContainer<T, Ts...> &data, size_t startIndex, size_t writeSize);

    template<typename T, template<typename...> class VectorContainer, typename... Ts>
    inline size_t insert_all(const VectorContainer<T, Ts...> &data, size_t offset = 0){return this->insert(data, offset, std::numeric_limits<size_t>::max());};

    template<typename T>
    size_t insert_array(const T *data, size_t arraySize);

    template<typename T, size_t N>
    inline size_t insert_array(const std::array<T, N> &data){return this->insert_array(data.data(), N);};
    
    template<typename T, typename Index_T = size_t>
    size_t insert_all_indexed(const std::vector<T> &data, const std::vector<Index_T> &indices, size_t extent = 1);
    
    template<typename T>
    size_t insert_elements(const std::vector<T> &data, size_t startIndex, size_t numElements);

    template<typename T>
    size_t extract(T &data, size_t idx) const;

    template<typename T, std::size_t N>
    size_t extract(std::array<T, N> &data, size_t idx) const;

    template<typename U, typename V>
    size_t extract(std::pair<U, V> &data, size_t idx) const;

    template<typename T>
    size_t extract(std::vector<T> &data, size_t idx) const;

    template<typename T, template<typename...> class VectorContainer, typename... Ts>
    size_t extract(VectorContainer<T, Ts...> &values, size_t startIndex, size_t endIndex) const;

    template<typename T, template<typename...> class VectorContainer, typename... Ts>
    inline size_t extract_all(VectorContainer<T, Ts...> &values, size_t offset = 0) const{return this->extract(values, offset, this->internal.size());};

    template<typename T>
    size_t extract_array(T *values, size_t arraySize, size_t offset = 0) const;

    template<typename T, size_t N>
    inline size_t extract_array(std::array<T, N> &values, size_t offset = 0) const{return this->extract_array(values.data(), N, offset);};

    inline const char *getData() const{return this->internal.data();};

    inline char *getData(){return this->internal.data();};

    inline size_t size() const{return this->internal.size();};

    template<typename T>
    size_t hardSet(size_t index, const T &data);

private:
    std::vector<char> internal; // internal buffer
};

template<>
size_t Serializer::extract(std::string &data, size_t idx) const;

template<>
size_t Serializer::insert(const std::string &data);

inline char *Serializer::resize(size_t size)
{
    size_t oldSize = this->internal.size();
    this->internal.resize(oldSize + size);
    return this->internal.data() + oldSize;
}

template<typename T>
size_t Serializer::insert(const T &data)
{
    if constexpr(is_serializable<T>::value)
    {
        // serializable
        return data.dump(this);
    }
    else
    {
        static_assert(std::is_trivially_copyable_v<T>, "T must be trivially copyable");
        // not serializable
        constexpr size_t size = sizeof(T);
        const char *ptr = reinterpret_cast<const char*>(&data);
        this->internal.insert(this->internal.end(), ptr, ptr + size);
        return size;
    }
}

template<typename U, typename V>
size_t Serializer::insert(const std::pair<U, V> &data)
{
    size_t bytes = 0;
    bytes += this->insert(data.first);
    bytes += this->insert(data.second);
    return bytes;
}

template<typename U, typename V>
size_t Serializer::extract(std::pair<U, V> &data, size_t idx) const
{
    size_t bytes = 0;
    bytes += this->extract(data.first, idx);
    bytes += this->extract(data.second, idx + bytes);
    return bytes;
}

template<typename T>
size_t Serializer::insert(const std::vector<T> &data)
{
    // vector
    size_t bytes = 0;
    size_t putSizeIdx = this->internal.size();
    bytes += this->insert<size_t>(static_cast<size_t>(0));
    size_t vectorBytes = 0;
    vectorBytes = this->insert_all(data);
    bytes += vectorBytes;
    *(reinterpret_cast<size_t*>(this->internal.data() + putSizeIdx)) = vectorBytes;
    // std::cout << "Inserted a vector of of type " << typeid(T).name() << ", size " << vectorBytes << " bytes" << std::endl;
    return bytes;
}

template<typename T, template<typename...> class VectorContainer, typename... Ts>
force_inline size_t Serializer::insert(const VectorContainer<T, Ts...> &data, size_t startIndex, size_t writeSize)
{
    size_t index = startIndex;
    size_t bytes = 0;
    while((bytes < writeSize) and (index < data.size()))
    {
        bytes += this->insert(data[index]);
        index++;
    }
    return bytes;
}

template<typename T>
force_inline size_t Serializer::insert_elements(const std::vector<T> &data, size_t startIndex, size_t numElements)
{
    size_t index = startIndex;
    size_t inserted = 0;
    size_t bytes = 0;
    while((inserted < numElements) and (index < data.size()))
    {
        bytes += this->insert(data[index]);
        index++;
        inserted++;
    }
    return bytes;
}

template<typename T, typename Index_T>
force_inline size_t Serializer::insert_all_indexed(const std::vector<T> &data, const std::vector<Index_T> &indices, size_t extent)
{
    size_t bytes = 0;
    size_t N = indices.size();
    for(size_t i = 0; i < N; ++i)
    {
        for(size_t j = 0; j < extent; ++j)
        {
            assert(indices[i] * extent + j < data.size());
            bytes += this->insert(data[indices[i] * extent + j]);
        }
    }
    return bytes;
}

template<typename T>
force_inline size_t Serializer::insert_array(const T *data, size_t arraySize)
{
    size_t bytes = 0;
    for(size_t i = 0; i < arraySize; ++i)
    {
        bytes += this->insert(data[i]);
    }
    return bytes;
}

template<typename T>
size_t Serializer::extract(T &data, size_t idx) const
{
    assert(idx < this->internal.size());
    if constexpr(is_serializable<T>::value)
    {
        // serializable
        return data.load(this, idx);
    }
    else
    {
        static_assert(std::is_trivially_copyable_v<T>, "T must be trivially copyable");
        // not serializable
        constexpr size_t size = sizeof(T);
        if(idx + size > this->internal.size())
        {
            UniversalError eo("Serializer::extract, index out of bounds");
            eo.addEntry("Index", idx);
            eo.addEntry("Size", this->internal.size());
            throw eo;
        }
        char *ptr = reinterpret_cast<char*>(&data);
        std::memcpy(ptr, this->internal.data() + idx, size);
        return size;
    }
}

template<typename T, std::size_t N>
size_t Serializer::extract(std::array<T, N> &data, size_t idx) const
{
    return this->extract_array(data.data(), data.size(), idx);
}

template<typename T>
size_t Serializer::extract(std::vector<T> &data, size_t idx) const
{
    // vector
    size_t bytes = 0;
    data.clear();
    size_t size;
    bytes += this->extract(size, idx);
    // std::cout << "Reading a vector of type " << typeid(T).name() << ", of size " << size << " bytes" << std::endl;
    bytes += this->extract(data, idx + bytes, size);
    return bytes;
}


template<typename T, template<typename...> class VectorContainer, typename... Ts>
force_inline size_t Serializer::extract(VectorContainer<T, Ts...> &values, size_t startOffset, size_t readSize) const
{
    size_t bytesRead = 0;
    while((bytesRead < readSize) and ((startOffset + bytesRead) < this->internal.size()))
    {
        values.emplace_back();
        size_t bytesJustRead = this->extract(values.back(), startOffset + bytesRead);
        // std::cout << "Read an object of type " << typeid(T).name() << ", with size " << bytesJustRead << " bytes (total required read size " << readSize << ", internal size " << this->internal.size() << ")" << std::endl;
        bytesRead += bytesJustRead;
    }
    return bytesRead;
}

template<typename T>
force_inline size_t Serializer::extract_array(T *values, size_t arraySize, size_t offset) const
{
    size_t bytesRead = 0;
    size_t index = 0;
    while(((offset + bytesRead) < this->internal.size()) and (index < arraySize))
    {
        bytesRead += this->extract(values[index], offset + bytesRead);
        index++;
    }
    return bytesRead;
}

template<typename T>
size_t Serializer::hardSet(size_t index, const T &data)
{
    static_assert(std::is_trivially_copyable_v<T>, "Serializer::hardSet: T must be trivially copyable");
    constexpr size_t size = sizeof(T);
    std::memcpy(this->internal.data() + index, reinterpret_cast<const char*>(&data), size);
    return size;
}

#endif // RICH_MPI

#endif // SERIALIZER_HPP