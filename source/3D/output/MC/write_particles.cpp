#include "read_write_particles.hpp"

#ifdef RICH_MPI
    #include "utils/hdf5/HDF5WriterParallel.hpp"
#endif

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
        HDF5WriterParallel pwriter(filename, MPI_COMM_WORLD);
        HDF5Writer writer(pwriter.GetFileId());

        std::string rankPrefix = pwriter.GetPrefix();
        writer.WriteElement(rankPrefix + "/" + PARTICLES_DATASET_NAME, particles);
    }
#endif // RICH_MPI
