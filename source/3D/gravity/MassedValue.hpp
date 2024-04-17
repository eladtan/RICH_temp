#ifndef MASSED_VALUE_HPP
#define MASSED_VALUE_HPP

#include <array>
#include "GravityTypes.h"
#include "misc/serializable.hpp"

template<typename T>
struct MassedValue : public Serializable
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

    inline size_t getChunkSize() const override
    {
        return 3 + 3 + 1 + 6; // 3, 3 for value and CM, 1 for mass, 6 for Q
    }

    inline std::vector<double> serialize() const override
    {
        std::vector<double> valueSerialized = this->value.serialize();
        std::vector<double> CMSerialized = this->CM.serialize();
        valueSerialized.insert(valueSerialized.end(), CMSerialized.begin(), CMSerialized.end());
        valueSerialized.push_back(this->mass);
        valueSerialized.insert(valueSerialized.end(), this->Q.begin(), this->Q.end());
        return valueSerialized;
    }

    inline void unserialize(const std::vector<double> &data) override
    {
        this->value.unserialize(std::vector<double>(data.cbegin(), data.cbegin() + 3));
        this->CM.unserialize(std::vector<double>(data.cbegin() + 3, data.cbegin() + 6));
        this->mass = data[6];
        std::copy(data.cbegin() + 7, data.cbegin() + 13, this->Q.begin());
    }
};

#endif // MASSED_VALUE_HPP