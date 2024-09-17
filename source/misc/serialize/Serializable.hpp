#ifndef SERIALIZABLE2_HPP
#define SERIALIZABLE2_HPP

class Serializer;

class Serializable2
{
public:
    virtual ~Serializable2(void) = default;

    virtual size_t dump(Serializer *serializer) const = 0;

    virtual size_t load(const Serializer *serializer, std::size_t byteOffset) = 0;
};

template<typename T>
using is_serializable2 = std::is_convertible<T*, Serializable2*>;

#endif // SERIALIZABLE2_HPP