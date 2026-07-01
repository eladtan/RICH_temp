#ifndef READ_WRITE_PARTICLES_HPP
#define READ_WRITE_PARTICLES_HPP

#include <H5Cpp.h>
#include <filesystem>
#include "monte/particle/Particle.hpp"
#include "utils/hdf5/HDF5Helper.hpp"
#include "utils/hdf5/HDF5Reader.hpp"
#include "utils/hdf5/HDF5Writer.hpp"
#include "../vectorData.hpp"
#include "3D/tessellation/Tessellation3D.hpp"

using Particle3D = MonteCarloParticle<Vector3D, Tessellation3D>;

#define PARTICLES_DATASET_NAME "particles"

using namespace H5;
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
        static H5::CompType get()
        {
            static H5::CompType mtype = []()
            {
                #pragma GCC diagnostic push
                #pragma GCC diagnostic ignored "-Winvalid-offsetof"
                        H5::CompType mtype(sizeof(Particle3D));
                #ifdef RICH_MPI
                        mtype.insertMember("rank", HOFFSET(Particle3D, rank), H5::PredType::NATIVE_INT);
                #endif // RICH_MPI
                        mtype.insertMember("id", HOFFSET(Particle3D, id), H5::PredType::NATIVE_ULLONG);
                        mtype.insertMember("cellID", HOFFSET(Particle3D, cellID), H5::PredType::NATIVE_ULLONG);
                        // Vector3D may have a vtable (RICH_MPI), but x,y,z are contiguous after it.
                        // Point to x and use ArrayType(DOUBLE, 3) to cover x,y,z.
                        H5::CompType vecType = HDF5Utils::CompTypeCreator<Vector3D>::get();
                        Particle3D dummy;
                        const size_t location_offset = reinterpret_cast<const char*>(&dummy.location) - reinterpret_cast<const char*>(&dummy);
                        const size_t velocity_offset = reinterpret_cast<const char*>(&dummy.velocity) - reinterpret_cast<const char*>(&dummy);
                        mtype.insertMember("location", location_offset, vecType);
                        mtype.insertMember("velocity", velocity_offset, vecType);
                        mtype.insertMember("cellIndex", HOFFSET(Particle3D, cellIndex), H5::PredType::NATIVE_ULLONG);
                        mtype.insertMember("timeLeft", HOFFSET(Particle3D, timeLeft), H5::PredType::NATIVE_DOUBLE);
                        mtype.insertMember("frequency", HOFFSET(Particle3D, frequency), H5::PredType::NATIVE_DOUBLE);
                        mtype.insertMember("weight", HOFFSET(Particle3D, weight), H5::PredType::NATIVE_DOUBLE);
                        mtype.insertMember("initialWeight", HOFFSET(Particle3D, initialWeight), H5::PredType::NATIVE_DOUBLE);
                        mtype.insertMember("steps", HOFFSET(Particle3D, steps), H5::PredType::NATIVE_ULLONG);
                #pragma GCC diagnostic pop
                return mtype;
            }();
            return mtype;
        }
    };
} // namespace HDF5Utils

void WriteParticles(const std::vector<Particle3D> &particles, const Group &group);

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