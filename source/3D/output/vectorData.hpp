#ifndef VECTOR_DATA_HPP
#define VECTOR_DATA_HPP

#include "3D/elementary/Vector3D.hpp"
#include "utils/hdf5/HDF5Helper.hpp"

namespace HDF5Utils
{
    template<>
    struct HasCompType<Vector3D> : std::true_type {};

    template<>
    struct CompTypeCreator<Vector3D>
    {
        static hid_t get()
        {
            static hid_t vtype = []()
            {
                hid_t t = H5Tcreate(H5T_COMPOUND, sizeof(Vector3D));
                H5Tinsert(t, "x", HOFFSET(Vector3D, x), H5T_NATIVE_DOUBLE);
                H5Tinsert(t, "y", HOFFSET(Vector3D, y), H5T_NATIVE_DOUBLE);
                H5Tinsert(t, "z", HOFFSET(Vector3D, z), H5T_NATIVE_DOUBLE);
                return t;
            }();
            return vtype;
        }
    };
}

#endif // VECTOR_DATA_HPP
