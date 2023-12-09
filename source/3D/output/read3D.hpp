#ifndef OUTPUT_READ_3D_HPP
#define OUTPUT_READ_3D_HPP

#include "read_utils.hpp"
#include "Snapshot3D.hpp"
#include "3D/tesselation/voronoi/Voronoi3D.hpp" // includes Tessellation3D as well

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
	,bool mpi_write = false,int fake_rank=-1
#endif
);

#ifdef RICH_MPI

/*! \brief Redistribute data between the different processes
  \param filename Name of output file
  \param proctess Meta tessellation
  \param snapshot_number Number of snapshot
  \param mpi_write Parallel output flag
  \return Hydrodynamic snapshot
 */
Snapshot3D ReDistributeData3D(string const& filename, Tessellation3D const& proctess, size_t snapshot_number,bool mpi_write=false);

#endif

std::vector<Vector3D> ReadVoronoiPoints(std::string const& filename);

#endif // OUTPUT_READ_3D_HPP