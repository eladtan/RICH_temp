#ifndef READ_WRITE_PARTICLES_HPP
#define READ_WRITE_PARTICLES_HPP

#include <filesystem>
#include "monte/MonteCarloParticle.hpp"
#include "utils/hdf5/HDF5Helper.hpp"
#include "utils/hdf5/HDF5Reader.hpp"
#include "utils/hdf5/HDF5Writer.hpp"
#include "../vectorData.hpp"
#include "3D/tessellation/Tessellation3D.hpp"

using Particle3D = MonteCarloParticle<Vector3D, Tessellation3D>;

#define PARTICLES_DATASET_NAME "particles"

namespace fs = std::filesystem;

#ifdef RICH_MPI
    #include <mpi.h>
#endif // RICH_MPI

namespace HDF5Utils
{
    template<>
    struct HasCompType<Particle3D> : std::true_type {};

    template<>
    struct CompTypeCreator<Particle3D>
    {
        static hid_t get()
        {
            static hid_t mtype = []()
            {
                #pragma GCC diagnostic push
                #pragma GCC diagnostic ignored "-Winvalid-offsetof"
                        hid_t t = H5Tcreate(H5T_COMPOUND, sizeof(Particle3D));
                #ifdef RICH_MPI
                        H5Tinsert(t, "rank", HOFFSET(Particle3D, rank), H5T_NATIVE_INT);
                #endif // RICH_MPI
                        H5Tinsert(t, "id", HOFFSET(Particle3D, id), H5T_NATIVE_ULLONG);
                        H5Tinsert(t, "cellID", HOFFSET(Particle3D, cellID), H5T_NATIVE_ULLONG);
                        hid_t vecType = HDF5Utils::CompTypeCreator<Vector3D>::get();
                        Particle3D dummy;
                        const size_t location_offset = reinterpret_cast<const char*>(&dummy.location) - reinterpret_cast<const char*>(&dummy);
                        const size_t velocity_offset = reinterpret_cast<const char*>(&dummy.velocity) - reinterpret_cast<const char*>(&dummy);
                        H5Tinsert(t, "location", location_offset, vecType);
                        H5Tinsert(t, "velocity", velocity_offset, vecType);
                        H5Tinsert(t, "cellIndex", HOFFSET(Particle3D, cellIndex), H5T_NATIVE_ULLONG);
                        H5Tinsert(t, "timeLeft", HOFFSET(Particle3D, timeLeft), H5T_NATIVE_DOUBLE);
                        H5Tinsert(t, "frequency", HOFFSET(Particle3D, frequency), H5T_NATIVE_DOUBLE);
                        H5Tinsert(t, "weight", HOFFSET(Particle3D, weight), H5T_NATIVE_DOUBLE);
                        H5Tinsert(t, "initialWeight", HOFFSET(Particle3D, initialWeight), H5T_NATIVE_DOUBLE);
                        H5Tinsert(t, "steps", HOFFSET(Particle3D, steps), H5T_NATIVE_ULLONG);
                #pragma GCC diagnostic pop
                return t;
            }();
            return mtype;
        }
    };
} // namespace HDF5Utils

void WriteParticles(HDF5Writer &writer, const std::vector<Particle3D> &particles);

std::vector<Particle3D> ReadParticles(const HDF5Reader &reader);

std::vector<Particle3D> ReadParticles(const std::string &fname
                                        #ifdef RICH_MPI
                                            , bool mpi_read = false, int fake_rank = -1
                                        #endif // RICH_MPI
                                    );

void WriteParticlesSerial(const std::string &filename, const std::vector<Particle3D> &particles);

#ifdef RICH_MPI
    std::vector<Particle3D> ReadParticlesParallel(const std::string &filename, int fake_rank = -1);

    void WriteParticlesParallel(const std::string &filename, const std::vector<Particle3D> &particles);
#endif // RICH_MPI

#endif // READ_WRITE_PARTICLES_HPP
