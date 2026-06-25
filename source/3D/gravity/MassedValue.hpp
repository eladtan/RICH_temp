#ifndef MASSED_VALUE_HPP
#define MASSED_VALUE_HPP

#include <array>
#include <cstring>
#include "GravityTypes.h"
#include <mpi_utils/serialize/Serializer.hpp>

template<typename T>
struct MassedValue 
                #ifdef RICH_MPI
                    : public Serializable
                #endif // RICH_MPI
{
    using coord_type = typename T::coord_type;
    using Raw_type = typename is_raw_type_defined<T>::type;

    T value;
    T CM; // center of mass
    gravity_result_t mass;
    std::array<coord_type, 6> Q; // for quadropole calculations

    typename T::coord_type operator[](size_t idx) const{return this->value[idx];};
    
    typename T::coord_type &operator[](size_t idx){return this->value[idx];};

    inline MassedValue operator+(const MassedValue &other) const{return MassedValue(this->value + other.value);};

    inline MassedValue operator-(const MassedValue &other) const{return MassedValue(this->value - other.value);};

    inline MassedValue operator*(typename T::coord_type scalar) const{return MassedValue(this->value * scalar);};

    inline MassedValue operator/(typename T::coord_type scalar) const{return this->operator*(1 / scalar);};

    inline bool operator==(const T &other) const{return this->value == other;};

    inline bool operator==(const MassedValue &other) const{return this->operator==(other.value);};

    inline bool operator!=(const MassedValue &other) const{return !this->operator==(other);};

    inline friend std::ostream &operator<<(std::ostream &stream, const MassedValue &value)
    {
        stream << "[Point: " << value.value << ", Mass: " << value.mass << ", CM: " << value.CM << ", Q: (" << value.Q[0] << ", " << value.Q[1] << ", " << value.Q[2] << ", " << value.Q[3] << ", " << value.Q[4] << ", " << value.Q[5] << ")]";
        return stream;
    };

    explicit inline MassedValue(const T &value, gravity_result_t mass, const std::array<coord_type, 6> &Q): value(value), CM(value), mass(mass), Q(Q)
    {}

    explicit inline MassedValue(const T &value, gravity_result_t mass): MassedValue(value, mass, std::array<coord_type, 6>({0, 0, 0, 0, 0, 0})){};

    explicit inline MassedValue(const T &value): MassedValue(value, 0){};

    explicit inline MassedValue(): MassedValue(T(), 0){};

    #ifdef RICH_MPI

    inline size_t dump(Serializer *serializer) const override
    {
        size_t bytes = 0;
        bytes += this->value.dump(serializer);
        bytes += this->CM.dump(serializer);
        bytes += serializer->insert(this->mass);
        bytes += serializer->insert_array(this->Q);
        return bytes;
    }

    inline size_t load(const Serializer *serializer, size_t byteOffset) override
    {
        size_t bytes = 0;
        bytes += this->value.load(serializer, byteOffset);
        bytes += this->CM.load(serializer, byteOffset + bytes);
        bytes += serializer->extract(this->mass, byteOffset + bytes);
        bytes += serializer->extract_array(this->Q, byteOffset + bytes);
        return bytes;
    }

    static constexpr size_t FLAT_BYTE_SIZE = 13 * sizeof(double);

    inline void dumpFlat(char *dst) const
    {
        double buf[13];
        buf[0] = this->value.x; buf[1] = this->value.y; buf[2] = this->value.z;
        buf[3] = this->CM.x;    buf[4] = this->CM.y;    buf[5] = this->CM.z;
        buf[6] = this->mass;
        buf[7] = this->Q[0]; buf[8] = this->Q[1]; buf[9]  = this->Q[2];
        buf[10] = this->Q[3]; buf[11] = this->Q[4]; buf[12] = this->Q[5];
        std::memcpy(dst, buf, FLAT_BYTE_SIZE);
    }

    inline void loadFlat(const char *src)
    {
        double buf[13];
        std::memcpy(buf, src, FLAT_BYTE_SIZE);
        this->value.x = buf[0]; this->value.y = buf[1]; this->value.z = buf[2];
        this->CM.x    = buf[3]; this->CM.y    = buf[4]; this->CM.z    = buf[5];
        this->mass     = buf[6];
        this->Q[0] = buf[7]; this->Q[1] = buf[8]; this->Q[2]  = buf[9];
        this->Q[3] = buf[10]; this->Q[4] = buf[11]; this->Q[5] = buf[12];
    }

    #endif // RICH_MPI
};

#endif // MASSED_VALUE_HPP