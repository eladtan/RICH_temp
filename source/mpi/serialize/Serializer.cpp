#ifdef RICH_MPI

#include "Serializer.hpp"

template<>
size_t Serializer::extract(std::string &data, size_t idx) const
{
    // string 
    size_t bytes = 0;
    data = "";
    size_t size;
    bytes += this->extract<size_t>(size, idx);
    for(size_t i = 0; i < size; i++)
    {
        char c;
        bytes += this->extract<char>(c, idx + bytes);
        data += c;
    }
    return bytes;
}

template<>
size_t Serializer::insert(const std::string &data)
{
    // string
    size_t bytes = 0;
    bytes += this->insert<size_t>(data.size());
    for(char c : data)
    {
        bytes += this->insert<char>(c);
    }
    return bytes;
}

#endif // RICH_MPI