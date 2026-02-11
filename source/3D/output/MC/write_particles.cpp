#include "read_write_particles.hpp"

void WriteParticle(const Particle3D &particle, ParticleHDF5 &hdf5Particle)
{
    #ifdef RICH_MPI
        hdf5Particle.rank = particle.rank;
    #endif // RICH_MPI
    hdf5Particle.id = particle.id;
    hdf5Particle.cellID = particle.cellID;
    for(int i = 0; i < 3; i++)
    {
        hdf5Particle.location[i] = particle.location[i];
        hdf5Particle.velocity[i] = particle.velocity[i];
    }
    hdf5Particle.cellIndex = particle.cellIndex;
    hdf5Particle.timeLeft = particle.timeLeft;
    hdf5Particle.energy = particle.energy;
    hdf5Particle.weight = particle.weight;
    hdf5Particle.initialWeight = particle.initialWeight;
    hdf5Particle.steps = particle.steps;
}

void WriteParticles(const std::vector<Particle3D> &particles, const Group &group)
{
    CompType ptype = ParticleHDF5::CreateParticleType();
    size_t N = particles.size();
    std::vector<ParticleHDF5> particlesToWrite(N);
    for(size_t i = 0; i < N; i++)
    {
        WriteParticle(particles[i], particlesToWrite[i]);
    }
    hsize_t dims[1] = {N};
    DataSpace space(1, dims);
    DataSet dataset = group.createDataSet(PARTICLES_DATASET_NAME, ptype, space);
    dataset.write(particlesToWrite.data(), ptype);
}

void WriteParticlesSerial(const std::string &filename, const std::vector<Particle3D> &particles)
{
    H5File file;
    file.openFile(H5std_string(filename), H5F_ACC_RDWR);

    Group writegroup = file.openGroup("/");

    WriteParticles(particles, writegroup);
    writegroup.close();
    file.close();
}

#ifdef RICH_MPI
    void WriteParticlesParallel(const std::vector<Particle3D> &particles, const std::string &filename)
    {
        rank_t rank;
        int ws = 0;

        fs::path path = fs::absolute(filename).parent_path();
        std::string myFilePath;

        fs::path ranks_files_path = path / fs::path(filename).filename().replace_extension();
        if(not fs::exists(ranks_files_path))
        {
            fs::create_directory(ranks_files_path);
        }
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &ws);
        myFilePath = (ranks_files_path / std::to_string(rank)).string() + ".h5";

        // truncate my file and open it
        H5File file2;
        bool open_ok = false;
        for(size_t i = 0; i < 150; i++)
        {
            try
            {
                file2 = H5File(H5std_string(myFilePath), H5F_ACC_TRUNC);
                open_ok = true;
                break;
            }
            catch(H5::FileIException &error)
            {
                usleep((500 + rank) * 1000);
                std::cerr << "Rank " << rank << " failed to open counter " << i << std::endl;
                std::cout << "Rank " << rank << " failed to open counter " << i << ", directory exists " << fs::exists(ranks_files_path) << std::endl;
                // file2 = H5File(H5std_string(myFilePath), H5F_ACC_TRUNC);
            }
        }
        if(not open_ok)
        {
            throw UniversalError("Couldn't open file");
        }
        file2.close();

        WriteParticlesSerial(myFilePath, particles);

        MPI_Barrier(MPI_COMM_WORLD);
        // only rank 0 makes the shared file
        if(rank == 0)
        {
            file2 = H5File(H5std_string(filename), H5F_ACC_TRUNC);
            file2.close();

            hid_t shared_file_id = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

            for(int _rank = 0; _rank < ws; _rank++)
            {
                // merge `_rank`'s file
                std::string rankFile((ranks_files_path / std::to_string(_rank)).string() + ".h5");
                std::string rankGroupName("/rank" + std::to_string(_rank));
                H5Lcreate_external(rankFile.c_str(),
                                            "/",
                                        shared_file_id,
                                        rankGroupName.c_str(),
                                        H5P_DEFAULT,
                                        H5P_DEFAULT);
            }
            H5Fclose(shared_file_id);
        }
    }
#endif // RICH_MPI