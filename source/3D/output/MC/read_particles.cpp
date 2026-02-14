#include "read_write_particles.hpp"

std::vector<Particle3D> ReadParticles(const HDF5Reader &reader)
{
    std::vector<Particle3D> particles;
    reader.ReadElement(PARTICLES_DATASET_NAME, particles);
    return particles;
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