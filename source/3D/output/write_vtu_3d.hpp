#ifndef WRITE_VTU_HPP_
#define WRITE_VTU_HPP_

#include <vector>
#include <string>
#include <filesystem>
#include <cassert>

#ifdef RICH_MPI
	#include <mpi.h>
#endif

#include <vtkUnstructuredGrid.h>
#include <vtkCellData.h>
#include <vtkPointData.h>
#include <vtkCellArray.h>
#include <vtkDoubleArray.h>
#include <vtkIntArray.h>
#include <vtkStringArray.h>
#include <vtkXMLUnstructuredGridWriter.h>
#include <vtkXMLPUnstructuredGridWriter.h>
#include <vtkPolyhedron.h>
#include <vtkDataArray.h>
#include <vtkIdList.h>
#include <vtkPoints.h>
#include <vtkProperty.h>
#include <vtkNew.h>

#ifdef RICH_MPI
	#include <vtkMPI.h>
	#include <vtkMPICommunicator.h>
	#include <vtkMPIController.h>
#endif

#include "../tessellation/Tessellation3D.hpp"
#include "newtonian/three_dimensional/hdsim_3d.hpp"

namespace write_vtu3d
{
	void write_vtu_3d(std::filesystem::path const& file_name,
			std::vector<std::string> const& cell_variable_names,
			std::vector<std::vector<double>> const& cell_variables,
			std::vector<std::string> const& cell_strings_names,
			std::vector<std::vector<std::string>> const& cell_strings,
			std::vector<std::string> const& cell_vectors_names,
			std::vector<std::vector<Vector3D> > const& cell_vectors,
			std::vector<std::pair<std::string, double>> const &scalar_values,
			double const time,
			std::size_t cycle,
			Tessellation3D const& tess);

	inline void write_vtu_3d(std::filesystem::path const& file_name,
			std::vector<std::string> const& cell_variable_names,
			std::vector<std::vector<double>> const& cell_variables,
			std::vector<std::string> const& cell_vectors_names,
			std::vector<std::vector<Vector3D> > const& cell_vectors,
			double const time,
			std::size_t cycle,
			Tessellation3D const& tess)
	{
		std::vector<std::string> cell_strings_names;
		std::vector<std::vector<std::string>> cell_strings;
		std::vector<std::pair<std::string, double>> scalar_values;
		write_vtu_3d(file_name, cell_variable_names, cell_variables, cell_strings_names, cell_strings, cell_vectors_names, cell_vectors, scalar_values, time, cycle, tess);
	}

	inline void write_vtu_3d(std::filesystem::path const& file_name,
			std::vector<std::string> const& cell_variable_names,
			std::vector<std::vector<double>> const& cell_variables,
			std::vector<std::string> const& cell_strings_names,
			std::vector<std::vector<std::string>> const& cell_strings,
			std::vector<std::string> const& cell_vectors_names,
			std::vector<std::vector<Vector3D> > const& cell_vectors,
			std::vector<std::pair<std::string, double>> const &scalar_values,
			Tessellation3D const& tess)
	{
		write_vtu_3d(file_name, cell_variable_names, cell_variables, cell_strings_names, cell_strings, cell_vectors_names, cell_vectors, scalar_values, std::numeric_limits<double>::max(), std::numeric_limits<size_t>::max(), tess); 
	}
} //namespace

#endif /* WRITE_VTU_HPP_ */