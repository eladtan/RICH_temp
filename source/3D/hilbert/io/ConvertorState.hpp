#ifndef CONVERTOR_STATE_HPP
#define CONVERTOR_STATE_HPP

#include "3D/elementary/Vector3D.hpp"
#include "3D/output/vectorData.hpp"
#include "utils/hdf5/HDF5Helper.hpp"

struct ConvertorState
{
    Vector3D ll;
    Vector3D ur;
    size_t order;
};

namespace HDF5Utils
{
    template<>
    struct HasCompType<ConvertorState> : std::true_type {};

    template<>
    struct CompTypeCreator<ConvertorState>
    {
        static hid_t get()
        {
            static hid_t type = []()
            {
                hid_t t = H5Tcreate(H5T_COMPOUND, sizeof(ConvertorState));
                H5Tinsert(t, "ll", HOFFSET(ConvertorState, ll), HDF5Utils::CompTypeCreator<Vector3D>::get());
                H5Tinsert(t, "ur", HOFFSET(ConvertorState, ur), HDF5Utils::CompTypeCreator<Vector3D>::get());
                H5Tinsert(t, "order", HOFFSET(ConvertorState, order), H5T_NATIVE_ULLONG);
                return t;
            }();
            return type;
        }
    };
}

#endif // CONVERTOR_STATE_HPP
