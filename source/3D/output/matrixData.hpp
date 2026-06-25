#ifndef MATRIX_DATA_HPP
#define MATRIX_DATA_HPP

#include "3D/elementary/Mat33.hpp"
#include "3D/elementary/Mat44.hpp"
#include <MeshDecomposer3D/kernels/math/Mat33.hpp>
#include <MeshDecomposer3D/kernels/math/Mat44.hpp>
#include "utils/hdf5/HDF5Helper.hpp"

namespace HDF5Utils
{
    template<>
    struct HasCompType<Mat33<double>> : std::true_type {};

    template<>
    struct CompTypeCreator<Mat33<double>>
    {
        static H5::CompType get()
        {
            static H5::CompType type = []()
            {
                Mat33<double> dummy;
                size_t dataOffset = reinterpret_cast<const char*>(&dummy(0, 0))
                                  - reinterpret_cast<const char*>(&dummy);

                H5::CompType t(sizeof(Mat33<double>));
                hsize_t dims[] = {3, 3};
                H5::ArrayType arrType(H5::PredType::NATIVE_DOUBLE, 2, dims);
                t.insertMember("data", dataOffset, arrType);
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
        static H5::CompType get()
        {
            static H5::CompType type = []()
            {
                Mat44<double> dummy;
                size_t dataOffset = reinterpret_cast<const char*>(&dummy(0, 0))
                                  - reinterpret_cast<const char*>(&dummy);

                H5::CompType t(sizeof(Mat44<double>));
                hsize_t dims[] = {4, 4};
                H5::ArrayType arrType(H5::PredType::NATIVE_DOUBLE, 2, dims);
                t.insertMember("data", dataOffset, arrType);
                return t;
            }();
            return type;
        }
    };

    template<>
    struct HasCompType<Kernelization3D::Mat33<double>> : std::true_type {};

    template<>
    struct CompTypeCreator<Kernelization3D::Mat33<double>>
    {
        static H5::CompType get()
        {
            static H5::CompType type = []()
            {
                Kernelization3D::Mat33<double> dummy;
                size_t dataOffset = reinterpret_cast<const char*>(&dummy(0, 0))
                                  - reinterpret_cast<const char*>(&dummy);

                H5::CompType t(sizeof(Kernelization3D::Mat33<double>));
                hsize_t dims[] = {3, 3};
                H5::ArrayType arrType(H5::PredType::NATIVE_DOUBLE, 2, dims);
                t.insertMember("data", dataOffset, arrType);
                return t;
            }();
            return type;
        }
    };

    template<>
    struct HasCompType<Kernelization3D::Mat44<double>> : std::true_type {};

    template<>
    struct CompTypeCreator<Kernelization3D::Mat44<double>>
    {
        static H5::CompType get()
        {
            static H5::CompType type = []()
            {
                Kernelization3D::Mat44<double> dummy;
                size_t dataOffset = reinterpret_cast<const char*>(&dummy(0, 0))
                                  - reinterpret_cast<const char*>(&dummy);

                H5::CompType t(sizeof(Kernelization3D::Mat44<double>));
                hsize_t dims[] = {4, 4};
                H5::ArrayType arrType(H5::PredType::NATIVE_DOUBLE, 2, dims);
                t.insertMember("data", dataOffset, arrType);
                return t;
            }();
            return type;
        }
    };
}

#endif // MATRIX_DATA_HPP
