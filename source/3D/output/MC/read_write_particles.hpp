#ifndef READ_WRITE_PARTICLES_HPP
#define READ_WRITE_PARTICLES_HPP

#include <H5Cpp.h>
#include <filesystem>
#include "monte/MonteCarloParticle.hpp"
#include "3D/tessellation/voronoi/Voronoi3D.hpp"
#include "utils/hdf5/HDF5Reader.hpp"
#include "utils/hdf5/HDF5Writer.hpp"

using Particle3D = MonteCarloParticle<Vector3D, Tessellation3D>;

#define PARTICLES_DATASET_NAME "particles"

namespace fs = std::filesystem;
using namespace H5;

#ifdef RICH_MPI
    #include <mpi.h>
    #include "mpi/debug.h"
#endif // RICH_MPI

struct ParticleHDF5
{
public:
    #ifdef RICH_MPI
        rank_t rank;
    #endif // RICH_MPI
    size_t id = std::numeric_limits<size_t>::max();
    size_t cellID = std::numeric_limits<size_t>::max();
    double location[3] = {std::numeric_limits<double>::max(), std::numeric_limits<double>::max(), std::numeric_limits<double>::max()};
    double velocity[3] = {std::numeric_limits<double>::max(), std::numeric_limits<double>::max(), std::numeric_limits<double>::max()};
    size_t cellIndex = std::numeric_limits<size_t>::max();
    double timeLeft = std::numeric_limits<double>::max();
    double energy = std::numeric_limits<double>::max();
    double weight = std::numeric_limits<double>::max();
    double initialWeight = std::numeric_limits<double>::max();
    size_t steps = 0;
    
    static CompType CreateHDF5CompType(void)
    {
        CompType mtype(sizeof(ParticleHDF5));
        #ifdef RICH_MPI
            mtype.insertMember("rank", HOFFSET(ParticleHDF5, rank), PredType::NATIVE_INT);
        #endif // RICH_MPI
        mtype.insertMember("id", HOFFSET(ParticleHDF5, id), PredType::NATIVE_ULLONG);
        mtype.insertMember("cellID", HOFFSET(ParticleHDF5, cellID), PredType::NATIVE_ULLONG);
        hsize_t dims[1] = {3};
        ArrayType vecType(PredType::NATIVE_DOUBLE, 1, dims);
        mtype.insertMember("location", HOFFSET(ParticleHDF5, location), vecType);
        mtype.insertMember("velocity", HOFFSET(ParticleHDF5, velocity), vecType);
        mtype.insertMember("cellIndex", HOFFSET(ParticleHDF5, cellIndex), PredType::NATIVE_ULLONG);
        mtype.insertMember("timeLeft", HOFFSET(ParticleHDF5, timeLeft), PredType::NATIVE_DOUBLE);
        mtype.insertMember("energy", HOFFSET(ParticleHDF5, energy), PredType::NATIVE_DOUBLE);
        mtype.insertMember("weight", HOFFSET(ParticleHDF5, weight), PredType::NATIVE_DOUBLE);
        mtype.insertMember("initialWeight", HOFFSET(ParticleHDF5, initialWeight), PredType::NATIVE_DOUBLE);
        mtype.insertMember("steps", HOFFSET(ParticleHDF5, steps), PredType::NATIVE_ULLONG);
        return mtype;
    }
};

std::vector<Particle3D> ReadParticles(const Group &group);

std::vector<Particle3D> ReadParticles(const std::string &fname
                                        #ifdef RICH_MPI
                                            , bool mpi_write = false, int fake_rank = -1
                                        #endif // RICH_MPI
                                    );

void WriteParticlesSerial(const std::string &filename, const std::vector<Particle3D> &particles);

#ifdef RICH_MPI
    std::vector<Particle3D> ReadParticlesParallel(const std::string &filename, int fake_rank = -1);

    void WriteParticlesParallel(const std::string &filename, const std::vector<Particle3D> &particles);
#endif // RICH_MPI

#endif // READ_WRITE_PARTICLES_HPP