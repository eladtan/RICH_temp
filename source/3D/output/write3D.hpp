#ifndef OUTPUT_WRITE_3D_HPP
#define OUTPUT_WRITE_3D_HPP

#include <H5Cpp.h>
#include <string>
#include <filesystem>
#include "3D/tesselation/voronoi/Voronoi3D.hpp"
#include "misc/hdf5_utils.hpp"
#include "write_vtu_3d.hpp"
#include "DiagnosticAppendix3D.hpp"

namespace fs = std::filesystem;
using namespace H5;

#ifdef RICH_MPI
	#include <mpi.h>
	#define HDF5_WRITE_BLOCK_TAG 9041
#endif // RICH_MPI

/*! \brief Write voronoi data to a file, parallely
  \param tri Voronoit tessellation
  \param filename Name of output file
  \param write_vtu whether to write to vtu as well
 */
void WriteVoronoiParallel(const Voronoi3D &tri, const std::string &filename, const std::vector<std::vector<double>> &data = std::vector<std::vector<double>>(), const std::vector<std::string>& names = std::vector<std::string>(), bool write_vtu = true);

/*! \brief Write voronoi data to a file, of all the ranks, but not parallely (excusively)
  \param tri Voronoit tessellation
  \param filename Name of output file
  \param write_vtu whether to write to vtu as well
 */
void WriteVoronoi(const Voronoi3D &tri, const std::string &filename, const std::vector<std::vector<double>> &data = std::vector<std::vector<double>>(), const std::vector<std::string>& names = std::vector<std::string>(), bool write_vtu = true);

/*! \brief Write voronoi data to a file
  \param tri Voronoit tessellation
  \param filename Name of output file
 */
void WriteVoronoiSerial(const Voronoi3D &tri, const std::string &filename);

#if RICH_MPI
/*! \brief Write snapshot to file
  \param sim Simulation
  \param filename name of output file
  \param appendices Custom fields
  \param mpi_write Determines whether to write parallisation data
 */
#else
/*! \brief Write snapshot to file
  \param sim Simulation
  \param filename name of output file
  \param appendices Custom fields
 */
#endif // RICH_MPI
void WriteSnapshot3D(HDSim3D const& sim, std::string const& filename,
	const vector<DiagnosticAppendix3D*>& appendices = vector<DiagnosticAppendix3D*>()
#ifdef RICH_MPI
	,bool mpi_write = true
#endif
, bool const write_vtu = true
);

void WritePoints(const std::vector<Vector3D> &points, const std::string &filename, const std::vector<std::vector<double>> &data = std::vector<std::vector<double>>(), const std::vector<std::string>& names = std::vector<std::string>());

void WritePointsParallel(const std::vector<Vector3D> &points, const std::string &filename, const std::vector<std::vector<double>> &data = std::vector<std::vector<double>>(), const std::vector<std::string>& names = std::vector<std::string>());


#endif // OUTPUT_WRITE_3D_HPP