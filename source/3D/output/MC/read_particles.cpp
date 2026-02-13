#include "read_write_particles.hpp"

void ReadParticle(const ParticleHDF5 &hdf5Particle, Particle3D &particle)
{
    #ifdef RICH_MPI
        particle.rank = hdf5Particle.rank;
    #endif // RICH_MPI
    particle.id = hdf5Particle.id;
    particle.cellID = hdf5Particle.cellID;
    for(int i = 0; i < 3; i++)
    {
        particle.location[i] = hdf5Particle.location[i];
        particle.velocity[i] = hdf5Particle.velocity[i];
    }
    particle.cellIndex = hdf5Particle.cellIndex;
    particle.timeLeft = hdf5Particle.timeLeft;
    particle.energy = hdf5Particle.energy;
    particle.weight = hdf5Particle.weight;
    particle.initialWeight = hdf5Particle.initialWeight;
    particle.steps = hdf5Particle.steps;
}

std::vector<Particle3D> ReadParticles(const HDF5Reader &reader)
{

    std::vector<ParticleHDF5> particles;
    reader.ReadElement(PARTICLES_DATASET_NAME, particles);
    
    size_t N = particles.size();
    std::vector<Particle3D> particlesToReturn(N);
    for(size_t i = 0; i < N; i++)
    {
        ReadParticle(particles[i], particlesToReturn[i]);
    }

    return particlesToReturn;
}

std::vector<Particle3D> ReadParticles(const std::string &fname
    #ifdef RICH_MPI
        , bool mpi_read, int fake_rank
    #endif // RICH_MPI
    )
{
    // HDF5Reader globalReader(fname);
    std::shared_ptr<HDF5Reader> reader = nullptr;

    #ifdef RICH_MPI
        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

        if(mpi_read)
        {
            std::string read_directory = std::filesystem::path(fname).replace_extension("").string();
            
            int rank_to_read = (fake_rank >= 0)? fake_rank : rank;
            std::string rank_file = read_directory + "/" + std::to_string(rank_to_read) + ".h5";
            if(not std::filesystem::exists(rank_file))
            {
                rank_file = read_directory + "/0.h5";
            }
            reader = std::make_shared<HDF5Reader>(rank_file);

            rank_t rank;
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);
            std::string rank_group_name = "/rank_" + std::to_string(rank);
            if(fake_rank >= 0)
            {
                rank = fake_rank;
            }
        }
        else
        {
            reader = std::make_shared<HDF5Reader>(fname);
        }
    #else // RICH_MPI
        reader = std::make_shared<HDF5Reader>(fname);
    #endif // RICH_MPI

    std::vector<Particle3D> res = ReadParticles(*reader);

    return res;
}

#ifdef RICH_MPI
    std::vector<Particle3D> ReadParticlesParallel(const std::string &fname, int fake_rank)
    {
        return ReadParticles(fname, true, fake_rank);
    }
#endif // RICH_MPI