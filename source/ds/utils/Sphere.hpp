#ifndef SPHERE_HPP
#define SPHERE_HPP

#ifdef USE_VCL_VECTORIZATION
    #include <vectorclass.h>
#endif // USE_VCL_VECTORIZATION

#ifdef RICH_MPI
    #include "misc/serializable.hpp"
#endif // RICH_MPI

#define DIM 3

#define TOLERANCE 1e-12

template<typename T>
class Sphere
            #ifdef RICH_MPI
                : public Serializable
            #endif // RICH_MPI
{
public:
    T center;
    typename T::coord_type radius;

    template<typename U> 
    Sphere(const U &center, typename T::coord_type radius): radius(radius)
    {
        this->center[0] = center[0];
        this->center[1] = center[1];
        this->center[2] = center[2];
    }

    Sphere(): Sphere(T(), typename T::coord_type()){};

    template<typename U>
    bool contains(const U &point) const;

    friend inline std::ostream &operator<<(std::ostream &stream, const Sphere<T> &sphere)
    {
        return stream << "Sphere(" << sphere.center << ", " << sphere.radius << ")";
    }

    #ifdef RICH_MPI
        inline size_t getChunkSize() const override
        {
            return 4;
        }

        inline std::vector<double> serialize() const override
        {
            std::vector<double> data(4);
            data[0] = this->center[0];
            data[1] = this->center[1];
            data[2] = this->center[2];
            data[3] = this->radius;
            return data;
        }

        inline void unserialize(const std::vector<double> &data) override
        {
            this->center[0] = data[0];
            this->center[1] = data[1];
            this->center[2] = data[2];
            this->radius = data[3];
        }
    #endif // RICH_MPI
};

template<typename T>
template<typename U>
bool Sphere<T>::contains(const U &point) const
{
    typename T::coord_type distance2 = 0, radius2 = 0;
    #ifdef USE_VCL_VECTORIZATION
        Vec4d diff(point[0] - this->center[0], point[1] - this->center[1], point[2] - this->center[2], this->radius);
        Vec4d distanceSquared = diff * diff;
        distance2 = distanceSquared[0] + distanceSquared[1] + distanceSquared[2];
        radius2 = distanceSquared[3];
    #else // USE_VCL_VECTORIZATION
        for(int i = 0; i < DIM; i++)
        {
            double _distance = (point[i] - this->center[i]);
            distance2 += _distance * _distance;
        }
        radius2 = (this->radius * this->radius);
    #endif // USE_VCL_VECTORIZATION
    return distance2 <= (1 + TOLERANCE) * radius2;
}

#endif // SPHERE_HPP