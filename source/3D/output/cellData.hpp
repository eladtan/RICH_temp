#include "newtonian/three_dimensional/computational_cell.hpp"
#include "ds/utils/BoundingBox.hpp"
#include "utils/hdf5/HDF5Helper.hpp"
#include "vectorData.hpp"

namespace HDF5Utils
{
    template<>
    struct HasCompType<BoundingBox<Vector3D>> : std::true_type {};

    template<>
    struct CompTypeCreator<BoundingBox<Vector3D>>
    {
        static hid_t get()
        {
            static hid_t btype = []()
            {
                hid_t t = H5Tcreate(H5T_COMPOUND, sizeof(BoundingBox<Vector3D>));
                const BoundingBox<Vector3D> dummy(Vector3D(0,0,0), Vector3D(1,1,1));
                const size_t ll_offset = reinterpret_cast<const char*>(&dummy.getLL()) - reinterpret_cast<const char*>(&dummy);
                const size_t ur_offset = reinterpret_cast<const char*>(&dummy.getUR()) - reinterpret_cast<const char*>(&dummy);
                H5Tinsert(t, "ll", ll_offset, CompTypeCreator<Vector3D>::get());
                H5Tinsert(t, "ur", ur_offset, CompTypeCreator<Vector3D>::get());
                return t;
            }();
            return btype;
        }
    };

    template<>
    struct HasCompType<ComputationalCell3D> : std::true_type {};

    template<>
    struct CompTypeCreator<ComputationalCell3D>
    {
        static hid_t get()
        {
            #pragma GCC diagnostic push
            #pragma GCC diagnostic ignored "-Winvalid-offsetof"
                    hid_t mtype = H5Tcreate(H5T_COMPOUND, sizeof(ComputationalCell3D));
                    H5Tinsert(mtype, "ID", HOFFSET(ComputationalCell3D, ID), H5T_NATIVE_ULLONG);

                    hid_t vecType = CompTypeCreator<Vector3D>::get();
                    H5Tinsert(mtype, "velocity", HOFFSET(ComputationalCell3D, velocity), vecType);

                    H5Tinsert(mtype, "density", HOFFSET(ComputationalCell3D, density), H5T_NATIVE_DOUBLE);
                    H5Tinsert(mtype, "pressure", HOFFSET(ComputationalCell3D, pressure), H5T_NATIVE_DOUBLE);
                    H5Tinsert(mtype, "internal_energy", HOFFSET(ComputationalCell3D, internal_energy), H5T_NATIVE_DOUBLE);
                    H5Tinsert(mtype, "temperature", HOFFSET(ComputationalCell3D, temperature), H5T_NATIVE_DOUBLE);
                    H5Tinsert(mtype, "dt", HOFFSET(ComputationalCell3D, dt), H5T_NATIVE_DOUBLE);
                    H5Tinsert(mtype, "Erad", HOFFSET(ComputationalCell3D, Erad), H5T_NATIVE_DOUBLE);
                    H5Tinsert(mtype, "Erad_dt", HOFFSET(ComputationalCell3D, Erad_dt), H5T_NATIVE_DOUBLE);
                    H5Tinsert(mtype, "Erad_dt_dt", HOFFSET(ComputationalCell3D, Erad_dt_dt), H5T_NATIVE_DOUBLE);
                    H5Tinsert(mtype, "cs", HOFFSET(ComputationalCell3D, cs), H5T_NATIVE_DOUBLE);

                    if(!ComputationalCell3D::tracerNames.empty())
                    {
                        hsize_t tracers_size[] = {ComputationalCell3D::tracerNames.size()};
                        hid_t tracersType = H5Tarray_create2(H5T_NATIVE_DOUBLE, 1, tracers_size);
                        H5Tinsert(mtype, "tracers", HOFFSET(ComputationalCell3D, tracers), tracersType);
                        H5Tclose(tracersType);
                    }
                    
                    if(!ComputationalCell3D::stickerNames.empty())
                    {
                        hsize_t stickers_size[] = {ComputationalCell3D::stickerNames.size()};
                        hid_t stickersType = H5Tarray_create2(H5T_NATIVE_UINT8, 1, stickers_size);
                        H5Tinsert(mtype, "stickers", HOFFSET(ComputationalCell3D, stickers), stickersType);
                        H5Tclose(stickersType);
                    }

                    if constexpr(ENERGY_GROUPS_NUM > 0)
                    {
                        const ComputationalCell3D dummy;
                        const size_t eg_offset = reinterpret_cast<const char*>(dummy.Eg.data()) - reinterpret_cast<const char*>(&dummy);
                        hsize_t energy_groups_size[] = {ENERGY_GROUPS_NUM};
                        hid_t energy_groupsType = H5Tarray_create2(H5T_NATIVE_DOUBLE, 1, energy_groups_size);
                        H5Tinsert(mtype, "Eg", eg_offset, energy_groupsType);
                        H5Tclose(energy_groupsType);
                    }

            #pragma GCC diagnostic pop
            return mtype;
        }
    };
} // namespace HDF5Utils
