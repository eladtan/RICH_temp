#ifndef _RANKED_INDEXED_VECTOR_HPP
#define _RANKED_INDEXED_VECTOR_HPP

#include <iostream>
#include "3D/elementary/Vector3D.hpp"
#include "3D/hilbert/hilbertTypes.h"
#ifdef RICH_MPI
    #include "mpi/serialize/Serializer.hpp"
#endif // RICH_MPI

#define ILLEGAL_IDX std::numeric_limits<size_t>::max()
#define ILLEGAL_RANK -1

#ifdef RICH_MPI

typedef struct RankedIndexedVector3D
                    #ifdef RICH_MPI
                        : public Serializable
                    #endif // RICH_MPI
{
    using coord_type = coord_t;
    using Raw_type = Vector3D;
    
    coord_t values[3];
    size_t index;
    rank_t rank;

    inline RankedIndexedVector3D(const coord_t *values, size_t index, rank_t rank): index(index), rank(rank){this->values[0] = values[0]; this->values[1] = values[1]; this->values[2] = values[2];};
    inline RankedIndexedVector3D(coord_t x, coord_t y, coord_t z, size_t index, rank_t rank): index(index), rank(rank){this->values[0] = x; this->values[1] = y; this->values[2] = z;};
    inline RankedIndexedVector3D(coord_t x = 0, coord_t y = 0, coord_t z = 0): RankedIndexedVector3D(x, y, z, ILLEGAL_IDX, -1){};
    inline RankedIndexedVector3D(const Vector3D &other): RankedIndexedVector3D(other.x, other.y, other.z){};
    inline RankedIndexedVector3D(const RankedIndexedVector3D &other): RankedIndexedVector3D(other.values[0], other.values[1], other.values[2], other.index, other.rank){};
    inline RankedIndexedVector3D(const Vector3D &point, size_t index, rank_t rank): RankedIndexedVector3D(point.x, point.y, point.z, index, rank){};
    inline RankedIndexedVector3D &operator=(const RankedIndexedVector3D &other)
    {
        this->values[0] = other.values[0];
        this->values[1] = other.values[1];
        this->values[2] = other.values[2];
        this->index = other.index;
        this->rank = other.rank;
        return *this;
    };
    inline bool operator==(const RankedIndexedVector3D &other) const{return (std::abs(this->values[0] - other.values[0]) < EPSILON) and (std::abs(this->values[1] - other.values[1]) < EPSILON) and (std::abs(this->values[2] - other.values[2]) < EPSILON);};
    inline bool operator<=(const RankedIndexedVector3D &other) const{
        if(this->values[0] < other.values[0]) return true;
        if(this->values[0] == other.values[0])
        {
            if(this->values[1] < other.values[1]) return true;
            if(this->values[1] == other.values[1]) return (this->values[2] <= other.values[2]);
        }
        return false;
    }
    inline bool operator<(const RankedIndexedVector3D &other) const{return (*this) <= other;};
    inline coord_t &operator[](size_t idx){return this->values[idx];};
    inline const coord_t &operator[](size_t idx) const{return this->values[idx];};
    inline RankedIndexedVector3D operator+(const RankedIndexedVector3D &other) const{return RankedIndexedVector3D(this->values[0] + other.values[0], this->values[1] + other.values[1], this->values[2] + other.values[2], ILLEGAL_IDX, ILLEGAL_RANK);};
    inline RankedIndexedVector3D operator*(coord_t scalar) const{return RankedIndexedVector3D(this->values[0] * scalar, this->values[1] * scalar, this->values[2] * scalar, ILLEGAL_IDX, ILLEGAL_RANK);};
    inline RankedIndexedVector3D operator/(coord_t scalar) const{return this->operator*(1/scalar);};
    friend inline std::ostream &operator<<(std::ostream &stream, const RankedIndexedVector3D &vec)
    {
        stream << "(" << vec.values[0] << ", " << vec.values[1] << ", " << vec.values[2] << ")";
        return stream;
    }

    friend inline std::istream &operator>>(std::istream &stream, RankedIndexedVector3D &point)
    {
        std::string str;
        std::getline(stream, str, '(');
        std::getline(stream, str, ',');
        point.values[0] = std::stod(str);
        std::getline(stream, str, ',');
        point.values[1] = std::stod(str);
        std::getline(stream, str, ')');
        point.values[2] = std::stod(str);
        return stream;
    }

    inline size_t getIndex() const{return this->index;};
    inline rank_t getRank() const{return this->rank;};
    inline Vector3D getVector() const{return Vector3D(values[0], values[1], values[2]);};

    #ifdef RICH_MPI
        force_inline size_t dump(Serializer *serializer) const override
        {
            size_t bytes = 0;
            bytes += serializer->insert(this->values[0]);
            bytes += serializer->insert(this->values[1]);
            bytes += serializer->insert(this->values[2]);
            bytes += serializer->insert(this->index);
            bytes += serializer->insert(this->rank);
            return bytes;
        }

        force_inline size_t load(const Serializer *serializer, size_t byteOffset) override
        {
            size_t bytesRead = 0;
            bytesRead += serializer->extract(this->values[0], byteOffset);
            bytesRead += serializer->extract(this->values[1], byteOffset + bytesRead);
            bytesRead += serializer->extract(this->values[2], byteOffset + bytesRead);
            bytesRead += serializer->extract(this->index, byteOffset + bytesRead);
            bytesRead += serializer->extract(this->rank, byteOffset + bytesRead);
            return bytesRead;
        }
    #endif // RICH_MPI
    
} RankedIndexedVector3D;

#endif // RICH_MPI

#endif // _RANKED_INDEXED_VECTOR_HPP
