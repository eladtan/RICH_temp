#include "write_vtu_3d.hpp"

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

namespace write_vtu3d{

void write_vtu_3d(std::filesystem::path const& file_name,
            std::vector<std::string> const& cell_variable_names,
			   std::vector<std::vector<double>> const& cell_variables,
			   std::vector<std::string> const& cell_vectors_names,
			   std::vector<std::vector<Vector3D> > const& cell_vectors,
			   double const time,
			   std::size_t cycle,
               Tessellation3D const& tess){
	std::vector<Vector3D> const& vertices = tess.GetFacePoints();
	std::size_t const num_vertices = vertices.size();
	std::size_t const num_cells = tess.GetPointNo();

	#ifdef RICH_MPI
		int mpi_rank;
		int mpi_size;
		MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
		MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
	#endif
	
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
		points->SetPoint(p, vertices[p].x, vertices[p].y, vertices[p].z);
	}
    ugrid->SetPoints(points);

	// ----------- cells connectivity
    std::vector<face_vec > const& cell_faces = tess.GetAllCellFaces();
    std::vector<point_vec > const& points_in_face = tess.GetAllPointsInFace();
	ugrid->Allocate(num_cells);
    std::vector<vtkIdType> point_array_in_cell;
 	for(std::size_t cell=0; cell<num_cells; ++cell){
        vtkNew<vtkIdList> faces;
        point_array_in_cell.clear();
        size_t const Nfaces = cell_faces[cell].size();
        //vtkIdType ptIds[currect_cell_num_vertices];
        for(size_t i = 0; i < Nfaces; ++i)
        {
            size_t const face_index = cell_faces[cell][i];
            size_t const Nvertices_in_face = points_in_face[face_index].size();
            faces->InsertNextId(Nvertices_in_face);
            for(size_t j = 0; j < Nvertices_in_face; ++j)
            {
                faces->InsertNextId(points_in_face[face_index][j]);
                point_array_in_cell.push_back(points_in_face[face_index][j]);
            }
        }
        std::sort(point_array_in_cell.begin(), point_array_in_cell.end());
        point_array_in_cell = unique(point_array_in_cell);
        vtkIdType* ptIds = &point_array_in_cell[0];
        ugrid->InsertNextCell(VTK_POLYHEDRON, point_array_in_cell.size(), ptIds, Nfaces, faces->GetPointer(0));
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
		auto const& var = cell_vectors[var_index];
		for(std::size_t cell=0; cell<num_cells; ++cell){
			var_data->SetTuple3(cell, var[cell].x, var[cell].y, var[cell].z);
		}
		ugrid->GetCellData()->AddArray(var_data);
	}


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
		pwriter->SetUseSubdirectory(true);

		// ----- write file to disk
		std::filesystem::path pname(file_name);
		pname.replace_extension("pvtu");
		//printf("%d/%d writing vtu file '%s'\n", mpi_rank+1, mpi_size, file_pvtu.c_str());
		pwriter->SetFileName(pname.c_str());
		pwriter->SetInputData(ugrid);
		pwriter->Write();
	#else
		// ------------ write the vtu file to disk
		vtkNew<vtkXMLUnstructuredGridWriter> writer;
		writer->SetCompressionLevel(9);
		writer->SetFileName(file_name.c_str());
		writer->SetInputData(ugrid);
		writer->Write();
	#endif
}

} //namespace