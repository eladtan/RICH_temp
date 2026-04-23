#ifndef MATRIX_DATA_HPP
#define MATRIX_DATA_HPP

#include "3D/elementary/Mat33.hpp"
#include "3D/elementary/Mat44.hpp"
#include "utils/hdf5/HDF5Helper.hpp"

namespace HDF5Utils
{
    template<>
    struct HasCompType<Mat33<double>> : std::true_type {};

    template<>
    struct CompTypeCreator<Mat33<double>>
    {
        static hid_t get()
        {
            static hid_t type = []()
            {
                Mat33<double> dummy;
                size_t dataOffset = reinterpret_cast<const char*>(&dummy(0, 0))
                                  - reinterpret_cast<const char*>(&dummy);

                hid_t t = H5Tcreate(H5T_COMPOUND, sizeof(Mat33<double>));
                hsize_t dims[] = {3, 3};
                hid_t arrType = H5Tarray_create2(H5T_NATIVE_DOUBLE, 2, dims);
                H5Tinsert(t, "data", dataOffset, arrType);
                H5Tclose(arrType);
                return t;
            }();
            return type;
        }
    };

    template<>
    struct HasCompType<Mat44<double>> : std::true_type {};

    template<>
    struct CompTypeCreator<Mat44<double>>
    {
        static hid_t get()
        {
            static hid_t type = []()
            {
                Mat44<double> dummy;
                size_t dataOffset = reinterpret_cast<const char*>(&dummy(0, 0))
                                  - reinterpret_cast<const char*>(&dummy);

                hid_t t = H5Tcreate(H5T_COMPOUND, sizeof(Mat44<double>));
                hsize_t dims[] = {4, 4};
                hid_t arrType = H5Tarray_create2(H5T_NATIVE_DOUBLE, 2, dims);
                H5Tinsert(t, "data", dataOffset, arrType);
                H5Tclose(arrType);
                return t;
            }();
            return type;
        }
    };
}

#endif // MATRIX_DATA_HPP
