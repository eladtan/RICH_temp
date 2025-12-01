#include "write_vtu.hpp"

#ifdef RICH_VTK

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
#include <vtkXMLUnstructuredGridWriter.h>
#include <vtkXMLPUnstructuredGridWriter.h>

#ifdef RICH_MPI
	#include <vtkMPI.h>
	#include <vtkMPICommunicator.h>
	#include <vtkMPIController.h>
#endif

namespace write_vtu{

void write_vtu(std::string const& file_name,
			   std::vector<double> const& x_vertices,
			   std::vector<double> const& y_vertices,
			   std::vector<std::size_t> const& cells_num_vertices, //number of vertices each cell has

			   std::vector<std::string> const& cell_variable_names,
			   std::vector<std::vector<double>> const& cell_variables,

			   std::vector<std::string> const& cell_vectors_names,
			   std::vector<std::vector<double>> const& cell_vectors_x,
			   std::vector<std::vector<double>> const& cell_vectors_y,
			   double const time,
			   std::size_t cycle){
	
	std::size_t const num_vertices = x_vertices.size();
	std::size_t const num_cells = cells_num_vertices.size();

	assert(x_vertices.size() == num_vertices);
	assert(y_vertices.size() == num_vertices);
	assert(cell_variable_names.size() == cell_variables.size());
	assert(cell_vectors_names.size() == cell_vectors_x.size());
	assert(cell_vectors_names.size() == cell_vectors_y.size());

	#ifdef RICH_MPI
		int mpi_rank;
		int mpi_size;
		MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
		MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
		auto const file_vtu = file_name + "_" + std::to_string(mpi_rank) + ".vtu";
	#else
		auto const file_vtu = file_name + ".vtu";
	#endif

	printf("writing vtu file '%s'\n", file_vtu.c_str());
	
	vtkNew<vtkUnstructuredGrid> ugrid;

	// ----------- time and cycle
	// see https://www.visitusers.org/index.php?title=Time_and_Cycle_in_VTK_files

	// time
    vtkNew<vtkDoubleArray> td;
    td->SetName("TIME");
    td->SetNumberOfTuples(1);
    td->SetTuple1(0, time);
    ugrid->GetFieldData()->AddArray(td);
	
	// cycle
    vtkNew<vtkIntArray> cd;
    cd->SetName("CYCLE");
    cd->SetNumberOfTuples(1);
    cd->SetTuple1(0, cycle);
    ugrid->GetFieldData()->AddArray(cd);

	// ----------- grid vertices coordinates
	// see https://kitware.github.io/vtk-examples/site/Cxx/GeometricObjects/QuadraticHexahedron/
	vtkNew<vtkPoints> points;
	points->SetNumberOfPoints(num_vertices);
 	for(std::size_t p=0; p<num_vertices; ++p){
		points->SetPoint(p, x_vertices[p], y_vertices[p], 0.);
	}
    ugrid->SetPoints(points);

	// ----------- cells connectivity
	ugrid->Allocate(num_cells);
	std::size_t vertex_index = 0;
 	for(std::size_t cell=0; cell<num_cells; ++cell){
		vtkIdType currect_cell_num_vertices = static_cast<vtkIdType>(cells_num_vertices[cell]);
		vtkIdType ptIds[currect_cell_num_vertices];
		for(vtkIdType ind=0; ind <currect_cell_num_vertices; ++ind){
			ptIds[ind] = vertex_index + ind; //RICH topology assumes cell vertices are ordered in this way
		}
		vertex_index += currect_cell_num_vertices;
		ugrid->InsertNextCell(VTK_POLYGON, currect_cell_num_vertices, ptIds);
	}

	// ------------cell centered data
	// see here:
	// https://paraview.paraview.narkive.com/GUqqKIK1/assign-scalars-vectors-to-mesh-points-in-vtkrectilineargrid-c

	// cell scalars data
	for(std::size_t var_index=0; var_index<cell_variable_names.size(); ++var_index){
		vtkNew<vtkDoubleArray> var_data;
		var_data->SetName(cell_variable_names[var_index].c_str());
		var_data->SetNumberOfComponents(1);
      	var_data->SetNumberOfValues(num_cells);
		auto const& var = cell_variables[var_index];
		assert(var.size() == num_cells);
		for(std::size_t cell=0; cell<num_cells; ++cell){
			var_data->SetValue(cell, var[cell]);
		}
		ugrid->GetCellData()->AddArray(var_data);
	}

	// write the number of edges of each cell
	vtkNew<vtkIntArray> var_num_edges;
	var_num_edges->SetName(std::string("num_edges").c_str());
	var_num_edges->SetNumberOfComponents(1);
	var_num_edges->SetNumberOfValues(num_cells);
	for(std::size_t cell=0; cell<num_cells; ++cell){
		var_num_edges->SetValue(cell, static_cast<int>(cells_num_vertices[cell]));
	}
	ugrid->GetCellData()->AddArray(var_num_edges);

	// write mpi rank as a cell centered data
	#ifdef RICH_MPI
		vtkNew<vtkIntArray> var_mpi_rank;
		var_mpi_rank->SetName(std::string("mpi_rank").c_str());
		var_mpi_rank->SetNumberOfComponents(1);
		var_mpi_rank->SetNumberOfValues(num_cells);
		for(std::size_t cell=0; cell<num_cells; ++cell){
			var_mpi_rank->SetValue(cell, mpi_rank);
		}
		ugrid->GetCellData()->AddArray(var_mpi_rank);
	#endif

	// cell vector data
	for(std::size_t var_index=0; var_index<cell_vectors_names.size(); ++var_index){
		vtkNew<vtkDoubleArray> var_data;
		var_data->SetName(cell_vectors_names[var_index].c_str());
		var_data->SetNumberOfComponents(3);
		var_data->SetNumberOfTuples(num_cells);
		auto const& var_x = cell_vectors_x[var_index];
		auto const& var_y = cell_vectors_y[var_index];
		assert(var_x.size() == num_cells);
		assert(var_y.size() == num_cells);
		for(std::size_t cell=0; cell<num_cells; ++cell){
			var_data->SetTuple3(cell, var_x[cell], var_y[cell], 0.);
		}
		ugrid->GetCellData()->AddArray(var_data);
	}

	// ------------ write the vtu file to disk
    vtkNew<vtkXMLUnstructuredGridWriter> writer;
    writer->SetFileName(file_vtu.c_str());
    writer->SetInputData(ugrid);
  	writer->Write();

	// ------------ write the 'pieced' .pvtu file
	/*
	Write a 'pvtu' file which holds metadata about all per-cpu 'vtu' files.
	this 'pvtu' file should be opened in paraview/VisIt etc.
	see for example:
	https://gerstrong.github.io/blog/2016/08/20/hacking-vtk-for-parallelisation
	https://www.steinzone.de/wordpress/hacking-vtk-for-parallelisation-mpi-and-c/
	http://rotorbit.blogspot.com/2017/02/how-to-write-vtk-files-in-parallel.html
	https://github.com/rotorbit/RotorbitTutorials/blob/master/rotorbit_WriteVTK.tar.gz
	
	HOWEVER! there is a bug in all of those links: not all .vtu files paths are listed
	in the resulting .pvtu file. Instead, only the root rank's .vtu file is listed.
	this can be solved by adding manually the .vtu file paths of all processes to the .pvtu file.
	
	Instead, a solution to this problem is obtained by passing the MPI communicator 
	to the VTK library, as offered here:
	https://discourse.vtk.org/t/distributed-i-o-with-vtkxmlpunstructuredgridwriter/7418/3
	which gives a link to these lines libmesh implementation, which I use here:
	https://github.com/libMesh/libmesh/blob/484c8652977b6504e1613633b4c7d87297f94957/src/mesh/vtk_io.C#L50-L54
	https://github.com/libMesh/libmesh/blob/484c8652977b6504e1613633b4c7d87297f94957/src/mesh/vtk_io.C#L275-L286
	but it only works if VTK library was compiled with MPI.
	*/
	#ifdef RICH_MPI
		vtkNew<vtkXMLPUnstructuredGridWriter> pwriter;

		// Set VTK library to use the same MPI communicator as we do
		vtkNew<vtkMPICommunicator> vtk_comm;
		MPI_Comm mpi_comm(MPI_COMM_WORLD);
		vtkMPICommunicatorOpaqueComm vtk_opaque_comm(&mpi_comm);
		vtk_comm ->InitializeExternal(&vtk_opaque_comm);
		vtkNew<vtkMPIController> vtk_mpi_ctrl;
		vtk_mpi_ctrl->SetCommunicator(vtk_comm );
		pwriter->SetController(vtk_mpi_ctrl);

		// Tell the writer how many partitions ('pieces') exist and on which processor we are currently
		pwriter->SetNumberOfPieces(mpi_size);
		pwriter->SetStartPiece(mpi_rank);
		pwriter->SetEndPiece(mpi_rank);

		// ----- write file to disk
		auto const file_pvtu = file_name + ".pvtu";
		printf("%d/%d writing vtu file '%s'\n", mpi_rank+1, mpi_size, file_pvtu.c_str());
		pwriter->SetFileName(file_pvtu.c_str());
		pwriter->SetInputData(ugrid);
		pwriter->Write();
	#endif

    printf("writing vtu complete\n");
}

} //namespace

#endif // RICH_VTK