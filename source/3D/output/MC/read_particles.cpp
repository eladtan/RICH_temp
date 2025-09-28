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

std::vector<Particle3D> ReadParticles(const Group &group)
{
    CompType ptype = ParticleHDF5::CreateParticleType();
    DataSet dataset = group.openDataSet(PARTICLES_DATASET_NAME);
    DataSpace space = dataset.getSpace();

    hsize_t dims[1];
    int ndims = space.getSimpleExtentDims(dims);
    size_t N = dims[0];
    
    std::vector<ParticleHDF5> particles(N);
    dataset.read(particles.data(), ptype);

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
    H5File file(fname, H5F_ACC_RDONLY);
    Group read_location = file.openGroup("/");
    bool good_found = true;
    #ifdef RICH_MPI
        if(mpi_read)
        {
            rank_t rank;
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);
            std::string rank_group_name = "/rank_" + std::to_string(rank);
            if(fake_rank >= 0)
            {
                rank = fake_rank;
            }
            try
            {
                read_location = file.openGroup("/rank" + std::to_string(rank));
            }
            catch(const Exception &notfounderror)
            {
                good_found = false;
                read_location = file.openGroup("/rank" + std::to_string(0));
            }

        }
    #endif // RICH_MPI

    std::vector<Particle3D> res = ReadParticles(read_location);

    read_location.close();
    file.close();

    return res;
}

#ifdef RICH_MPI
    std::vector<Particle3D> ReadParticlesParallel(const std::string &fname, int fake_rank)
    {
        fs::path input_directory = fs::path(fname).replace_extension();
        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        if(fake_rank >= 0)
        {
            rank = fake_rank;
        }

        H5File file(input_directory / std::to_string(rank) / ".h5", H5F_ACC_RDONLY);
        Group read_location = file.openGroup("/");
        H5File globalFile(fname, H5F_ACC_RDONLY);

        std::vector<Particle3D> res = ReadParticles(read_location);

        globalFile.close();
        read_location.close();
        file.close();

        return res;
    }
#endif // RICH_MPI