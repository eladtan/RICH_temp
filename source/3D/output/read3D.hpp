#ifndef OUTPUT_READ_3D_HPP
#define OUTPUT_READ_3D_HPP

#include <filesystem>
#include <set>
#include "Snapshot3D.hpp"
#include "3D/tessellation/voronoi/Voronoi3D.hpp" // includes Tessellation3D as well
#include "utils/hdf5/HDF5Reader.hpp"

#if RICH_MPI
/*! \brief Load snapshot data into memory
\param fname File name
\param mpi_write Flag for providing parallelisation data
\param fake_rank Process id
\return Snapshot data
*/
#else
/*! \brief Load snapshot data into memory
\param fname File name
\param fake_rank Process id
\return Snapshot data
*/
#endif // RICH_MPI
  Snapshot3D ReadSnapshot3D(const string& fname
#ifdef RICH_MPI
	                          , bool mpi_write = false, int fake_rank = -1
#endif
);

#ifdef RICH_MPI
  Snapshot3D ReadSnapshot3DParallel(const string &fname, int fake_rank = -1);
#endif

std::vector<Vector3D> ReadVoronoiPoints(const std::string &filename);

#ifdef RICH_MPI
    std::vector<Vector3D> ReadVoronoiPointsParallel(const std::string &filename);

    rank_t GetNumberOfRanksInHDF(std::string const& fname);
#endif // RICH_MPI


#endif // OUTPUT_READ_3D_HPP