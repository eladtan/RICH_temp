#ifndef WRITE_VTU_HPP_
#define WRITE_VTU_HPP_

#include <vector>
#include <string>
#include <filesystem>
#include <cassert>

#if defined(RICH_MPI) && defined(RICH_VTK_MPI)
#include <mpi.h>
#endif

#include <vtkUnstructuredGrid.h>
#include <vtkCellData.h>
#include <vtkPointData.h>
#include <vtkCellArray.h>
#include <vtkDoubleArray.h>
#include <vtkIntArray.h>
#include <vtkXMLUnstructuredGridWriter.h>
#include <vtkXMLPUnstructuredGridWriter.h>
#include <vtkPolyhedron.h>
#include <vtkDataArray.h>
#include <vtkIdList.h>
#include <vtkPoints.h>
#include <vtkProperty.h>
#include <vtkNew.h>

#if defined(RICH_MPI) && defined(RICH_VTK_MPI)
#include <vtkMPI.h>
#include <vtkMPICommunicator.h>
#include <vtkMPIController.h>
#endif

#include "../tesselation/Tessellation3D.hpp"
#include "newtonian/three_dimensional/hdsim_3d.hpp"

namespace write_vtu3d {

/**
 * @brief write a vtu snapshot file (in parallel mode a pvtu file is also written)
 * 
 * @param file_name The file name to write
 * @param cell_variable_names The name of the variables to write
 * @param cell_variables The cell centered data to write
 * @param cell_vectors_names 
 * @param cell_vectors
 * @param time 
 * @param cycle 
 */
void write_vtu_3d(std::filesystem::path const& file_name, std::vector<std::string> const& cell_variable_names,
                  std::vector<std::vector<double>> const& cell_variables,
                  std::vector<std::string> const& cell_vectors_names,
                  std::vector<std::vector<Vector3D>> const& cell_vectors, double const time, std::size_t cycle,
                  Tessellation3D const& tess);

inline void write_vtu_3d(std::filesystem::path const& file_name, std::vector<std::string> const& cell_variable_names,
                         std::vector<std::vector<double>> const& cell_variables,
                         std::vector<std::string> const& cell_vectors_names,
                         std::vector<std::vector<Vector3D>> const& cell_vectors, Tessellation3D const& tess) {
    write_vtu_3d(file_name, cell_variable_names, cell_variables, cell_vectors_names, cell_vectors,
                 std::numeric_limits<double>::max(), std::numeric_limits<size_t>::max(), tess);
}

} // namespace write_vtu3d

#endif /* WRITE_VTU_HPP_ */