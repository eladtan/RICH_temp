#include "newtonian/three_dimensional/computational_cell.hpp"
#include <spatial_ds/utils/BoundingBox.hpp>
#include "utils/hdf5/HDF5Helper.hpp"
#include "vectorData.hpp"

using namespace H5;

namespace HDF5Utils
{
    template<>
    struct HasCompType<BoundingBox<Vector3D>> : std::true_type {};

    template<>
    struct CompTypeCreator<BoundingBox<Vector3D>>
    {
        static H5::CompType get()
        {
            static H5::CompType btype = []()
            {
                H5::CompType btype(sizeof(BoundingBox<Vector3D>));
                const BoundingBox<Vector3D> dummy(Vector3D(0,0,0), Vector3D(1,1,1));
                const size_t ll_offset = reinterpret_cast<const char*>(&dummy.getLL()) - reinterpret_cast<const char*>(&dummy);
                const size_t ur_offset = reinterpret_cast<const char*>(&dummy.getUR()) - reinterpret_cast<const char*>(&dummy);
                btype.insertMember("ll", ll_offset, CompTypeCreator<Vector3D>::get());
                btype.insertMember("ur", ur_offset, CompTypeCreator<Vector3D>::get());
                return btype;
            }();
            return btype;
        }
    };

    template<>
    struct HasCompType<ComputationalCell3D> : std::true_type {};

    template<>
    struct CompTypeCreator<ComputationalCell3D>
    {
        static H5::CompType get()
        {
            #pragma GCC diagnostic push
            #pragma GCC diagnostic ignored "-Winvalid-offsetof"
                    H5::CompType mtype(sizeof(ComputationalCell3D));
                    mtype.insertMember("ID", HOFFSET(ComputationalCell3D, ID), H5::PredType::NATIVE_ULLONG);

                    H5::CompType vecType = CompTypeCreator<Vector3D>::get();
                    mtype.insertMember("velocity", HOFFSET(ComputationalCell3D, velocity), vecType);

                    mtype.insertMember("density", HOFFSET(ComputationalCell3D, density), H5::PredType::NATIVE_DOUBLE);
                    mtype.insertMember("pressure", HOFFSET(ComputationalCell3D, pressure), H5::PredType::NATIVE_DOUBLE);
                    mtype.insertMember("internal_energy", HOFFSET(ComputationalCell3D, internal_energy), H5::PredType::NATIVE_DOUBLE);
                    mtype.insertMember("temperature", HOFFSET(ComputationalCell3D, temperature), H5::PredType::NATIVE_DOUBLE);
                    mtype.insertMember("dt", HOFFSET(ComputationalCell3D, dt), H5::PredType::NATIVE_DOUBLE);
                    mtype.insertMember("Erad", HOFFSET(ComputationalCell3D, Erad), H5::PredType::NATIVE_DOUBLE);
                    mtype.insertMember("Erad_dt", HOFFSET(ComputationalCell3D, Erad_dt), H5::PredType::NATIVE_DOUBLE);
                    mtype.insertMember("Erad_dt_dt", HOFFSET(ComputationalCell3D, Erad_dt_dt), H5::PredType::NATIVE_DOUBLE);
                    mtype.insertMember("cs", HOFFSET(ComputationalCell3D, cs), H5::PredType::NATIVE_DOUBLE);

                    if(!ComputationalCell3D::tracerNames.empty())
                    {
                        hsize_t tracers_size[] = {ComputationalCell3D::tracerNames.size()};
                        H5::ArrayType tracersType(H5::PredType::NATIVE_DOUBLE, 1, tracers_size);
                        mtype.insertMember("tracers", HOFFSET(ComputationalCell3D, tracers), tracersType);
                    }
                    
                    if(!ComputationalCell3D::stickerNames.empty())
                    {
                        hsize_t stickers_size[] = {ComputationalCell3D::stickerNames.size()};
                        H5::ArrayType stickersType(H5::PredType::NATIVE_UINT8, 1, stickers_size);
                        mtype.insertMember("stickers", HOFFSET(ComputationalCell3D, stickers), stickersType);
                    }

                    if constexpr(ENERGY_GROUPS_NUM > 0)
                    {
                        const ComputationalCell3D dummy;
                        const size_t eg_offset = reinterpret_cast<const char*>(dummy.Eg.data()) - reinterpret_cast<const char*>(&dummy);
                        hsize_t energy_groups_size[] = {ENERGY_GROUPS_NUM};
                        H5::ArrayType energy_groupsType(H5::PredType::NATIVE_DOUBLE, 1, energy_groups_size);
                        mtype.insertMember("Eg", eg_offset, energy_groupsType);
                    }

            #pragma GCC diagnostic pop
            return mtype;
        }
    };
} // namespace HDF5Utils