#ifndef _INDEXED_VECTOR_HPP
#define _INDEXED_VECTOR_HPP

#include <iostream>
#include "3D/elementary/Vector3D.hpp"
#include <MeshDecomposer3D/hilbert/hilbertTypes.h>
#ifdef RICH_MPI
    #include <mpi_utils/serialize/Serializer.hpp>
#endif // RICH_MPI

#define ILLEGAL_IDX -1

typedef struct IndexedVector3D
                    #ifdef RICH_MPI
                        : public Serializable
                    #endif // RICH_MPI
{
    using coord_type = coord_t;
    using Raw_type = Vector3D;
    
    coord_t values[3];
    size_t index;

    inline IndexedVector3D(const coord_t *values, size_t index): index(index){this->values[0] = values[0]; this->values[1] = values[1]; this->values[2] = values[2];};
    inline IndexedVector3D(coord_t x, coord_t y, coord_t z, size_t index): index(index){this->values[0] = x; this->values[1] = y; this->values[2] = z;};
    inline IndexedVector3D(coord_t x, coord_t y, coord_t z): IndexedVector3D(x, y, z, 0){};
    inline explicit IndexedVector3D(): IndexedVector3D(0, 0, 0){};
    inline IndexedVector3D(const Vector3D &other): IndexedVector3D(other.x, other.y, other.z){};
    inline IndexedVector3D(const IndexedVector3D &other): IndexedVector3D(&other.values[0], other.index){};
    inline IndexedVector3D(const Vector3D &point, size_t index): IndexedVector3D(point.x, point.y, point.z, index){};
    inline IndexedVector3D &operator=(const IndexedVector3D &other){this->values[0] = other.values[0]; this->values[1] = other.values[1]; this->values[2] = other.values[2]; this->index = other.index; return *this;};
    inline bool operator==(const IndexedVector3D &other) const{return (std::abs(this->values[0] - other.values[0]) < EPSILON) and (std::abs(this->values[1] - other.values[1]) < EPSILON) and (std::abs(this->values[2] - other.values[2]) < EPSILON);};
    inline bool operator<=(const IndexedVector3D &other) const{
        if(this->values[0] < other.values[0]) return true;
        if(this->values[0] == other.values[0])
        {
            if(this->values[1] < other.values[1]) return true;
            if(this->values[1] == other.values[1]) return (this->values[2] <= other.values[2]);
        }
        return false;
    }
    inline bool operator<(const IndexedVector3D &other) const{return (*this) <= other;};
    inline coord_t &operator[](size_t idx){return this->values[idx];};
    inline const coord_t &operator[](size_t idx) const{return this->values[idx];};
    inline IndexedVector3D operator+(const IndexedVector3D &other) const{return IndexedVector3D(this->values[0] + other.values[0], this->values[1] + other.values[1], this->values[2] + other.values[2], ILLEGAL_IDX);};
    inline IndexedVector3D operator*(coord_t scalar) const{return IndexedVector3D(this->values[0] * scalar, this->values[1] * scalar, this->values[2] * scalar, ILLEGAL_IDX);};
    inline IndexedVector3D operator/(coord_t scalar) const{return this->operator*(1/scalar);};
    friend inline std::ostream &operator<<(std::ostream &stream, const IndexedVector3D &vec)
    {
        stream << "(" << vec.values[0] << ", " << vec.values[1] << ", " << vec.values[2] << ")";
        return stream;
    }

    friend inline std::istream &operator>>(std::istream &stream, IndexedVector3D &point)
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
    inline Vector3D getVector() const{return Vector3D(values[0], values[1], values[2]);};

    #ifdef RICH_MPI
        force_inline size_t dump(Serializer *serializer) const override
        {
            size_t bytes = 0;
            bytes += serializer->insert(this->values[0]);
            bytes += serializer->insert(this->values[1]);
            bytes += serializer->insert(this->values[2]);
            bytes += serializer->insert(this->index);
            return bytes;
        }

        force_inline size_t load(const Serializer *serializer, size_t byteOffset) override
        {
            size_t bytesRead = 0;
            bytesRead += serializer->extract(this->values[0], byteOffset);
            bytesRead += serializer->extract(this->values[1], byteOffset + bytesRead);
            bytesRead += serializer->extract(this->values[2], byteOffset + bytesRead);
            bytesRead += serializer->extract(this->index, byteOffset + bytesRead);
            return bytesRead;
        }
    #endif // RICH_MPI
    
} IndexedVector3D;


#endif // _INDEXED_VECTOR_HPP
