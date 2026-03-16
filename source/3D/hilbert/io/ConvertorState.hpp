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
        static H5::CompType get()
        {
            static H5::CompType type = []()
            {
                H5::CompType t(sizeof(ConvertorState));
                t.insertMember("ll", HOFFSET(ConvertorState, ll), HDF5Utils::CompTypeCreator<Vector3D>::get());
                t.insertMember("ur", HOFFSET(ConvertorState, ur), HDF5Utils::CompTypeCreator<Vector3D>::get());
                t.insertMember("order", HOFFSET(ConvertorState, order), H5::PredType::NATIVE_ULLONG);
                return t;
            }();
            return type;
        }
    };
}

#endif // CONVERTOR_STATE_HPP
