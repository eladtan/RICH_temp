#include "read_write_particles.hpp"

void WriteParticles(HDF5Writer &writer, const std::vector<Particle3D> &particles)
{
    writer.WriteElement(PARTICLES_DATASET_NAME, particles);
}

void WriteParticlesSerial(const std::string &filename, const std::vector<Particle3D> &particles)
{
    HDF5Writer writer(filename);
    WriteParticles(writer, particles);
}

#ifdef RICH_MPI
    void WriteParticlesParallel(const std::string &filename, const std::vector<Particle3D> &particles)
    {
        rank_t rank;
        int ws = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &ws);

        fs::path path = fs::absolute(filename).parent_path();
        std::string myFilePath;

        fs::path ranks_files_path = path / fs::path(filename).filename().replace_extension();
        if(not fs::exists(ranks_files_path))
        {
            fs::create_directory(ranks_files_path);
        }

        std::shared_ptr<HDF5Writer> globalWriter = nullptr;

        if(rank == 0)
        {
            globalWriter = std::make_shared<HDF5Writer>(filename);
        }

        myFilePath = (ranks_files_path / std::to_string(rank)).string() + ".h5";

        std::shared_ptr<HDF5Writer> writer = nullptr;
        bool open_ok = false;
        for(size_t i = 0; i < 150; i++)
        {
            try
            {
                writer = std::make_shared<HDF5Writer>(myFilePath);
                open_ok = true;
                break;
            }
            catch(H5::FileIException &error)
            {
                usleep((500 + rank) * 1000);
                std::cerr << "Rank " << rank << " failed to open file " << myFilePath << " after " << i << " attempts" << std::endl;
                std::cout << "Rank " << rank << " failed to open file " << myFilePath << " after " << i << " attempts, directory exists " << fs::exists(ranks_files_path) << std::endl;
            }
        }

        if(not open_ok)
        {
            throw UniversalError("Couldn't open file");
        }

        WriteParticles(*writer, particles);

        MPI_Barrier(MPI_COMM_WORLD);
        // only rank 0 makes the shared file
        if(rank == 0)
        {
            for(int _rank = 0; _rank < ws; _rank++)
            {
                // merge `_rank`'s file
                std::string rankFile((ranks_files_path / std::to_string(_rank)).string() + ".h5");
                std::string rankGroupName("/rank" + std::to_string(_rank));
                globalWriter->AddExternalLink(rankFile, "/", rankGroupName);
            }
        }
    }
#endif // RICH_MPI