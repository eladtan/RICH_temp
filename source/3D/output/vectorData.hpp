#ifndef VECTOR_DATA_HPP
#define VECTOR_DATA_HPP

#include "3D/elementary/Vector3D.hpp"
#include "utils/hdf5/HDF5Helper.hpp"

using namespace H5;

namespace HDF5Utils
{
    template<>
    struct HasCompType<Vector3D> : std::true_type {};

    template<>
    struct CompTypeCreator<Vector3D>
    {
        static H5::CompType get()
        {
            static H5::CompType vtype = []()
            {
                H5::CompType vtype(sizeof(Vector3D));
                vtype.insertMember("x", HOFFSET(Vector3D, x), H5::PredType::NATIVE_DOUBLE);
                vtype.insertMember("y", HOFFSET(Vector3D, y), H5::PredType::NATIVE_DOUBLE);
                vtype.insertMember("z", HOFFSET(Vector3D, z), H5::PredType::NATIVE_DOUBLE);
                return vtype;
            }();
            return vtype;
        }
    };
}

#endif // VECTOR_DATA_HPP