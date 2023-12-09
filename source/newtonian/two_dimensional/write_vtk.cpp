#include "write_vtk.hpp"

#include <numeric>
#include <algorithm>
#include <cassert>

#ifdef RICH_MPI
	#include <mpi.h>
#endif


namespace write_vtk{

void write_vtk(std::string const& file_name,
			   std::vector<double> const& x_vertices,
		       std::vector<double> const& y_vertices,
			   std::vector<std::size_t> const& cells_num_vertices, //number of vertices each cell has
			   std::vector<std::string> const& cell_variable_names,
			   std::vector<std::vector<double>> const& cell_variables,
			    //   std::vector<std::string> const& vertex_vector_names,
			    //   std::vector<std::vector<Vector2d>> const& vertex_vectors,
			    //   std::vector<std::size_t> const& cell_materials,
			   double const time,
			   std::size_t cycle){


	#ifdef RICH_MPI
		int mpi_rank;
		int mpi_size;
		MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
		MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
		std::string const file_vtk = file_name + "_" + std::to_string(mpi_rank) + ".vtk";
	#else
		std::string const file_vtk = file_name + ".vtk";
	#endif

	printf("writing vtk file '%s'\n", file_vtk.c_str());
	
	FILE* file = fopen(file_vtk.c_str(), "w");

	// format according to Figure2 of 
	//https://vtk.org/wp-content/uploads/2015/04/file-formats.pdf

	// Part (1): Header
	fprintf(file, "# vtk DataFile Version 2.0\n");

	// Part (2): Title
	fprintf(file, "%s hydro2d mesh and data\n", file_vtk.c_str());

	// Part (3): Data type
	fprintf(file, "ASCII\n");

	// Part (4): Geometry/topology
	fprintf(file, "DATASET UNSTRUCTURED_GRID\n");

	// time and cycle (https://www.visitusers.org/index.php?title=Time_and_Cycle_in_VTK_files)
	fprintf(file, "FIELD FieldData 2\n");
	fprintf(file, "TIME 1 1 double\n");
	fprintf(file, "%1.10E\n", time);
	fprintf(file, "CYCLE 1 1 int\n");
	fprintf(file, "%ld\n\n", cycle);
	
	std::size_t const num_vertices = x_vertices.size();
	std::size_t const num_cells = cells_num_vertices.size();

	// // vertices
	assert(y_vertices.size() == num_vertices);
	fprintf(file, "POINTS %ld double\n", num_vertices);
	for(std::size_t i=0; i<num_vertices; ++i){
		fprintf(file, "%1.10E %1.10E 0\n", x_vertices[i], y_vertices[i]);
	}
	
	// cells
	// the number cells + sum(the number of vertices of each cell)
    std::size_t const num_cells_vertices = std::size_t(num_cells + 
										   std::accumulate(cells_num_vertices.begin(), 
										   cells_num_vertices.end(), 
										   0.0)); // MUST BE 0.0 to be double!!!

	fprintf(file, "CELLS %ld %ld\n", num_cells, num_cells_vertices);

	std::size_t vertex_index = 0;
	for(std::size_t cell=0; cell<num_cells; ++cell){
		std::size_t currect_cell_num_vertices = std::size_t(cells_num_vertices[cell]);
		fprintf(file, "%ld ", currect_cell_num_vertices);
		for(std::size_t ind=vertex_index; ind <= vertex_index+currect_cell_num_vertices-1; ++ind){
			fprintf(file, "%ld ", ind);
		}
		vertex_index += currect_cell_num_vertices;
		fprintf(file, "\n");
	}

	fprintf(file, "CELL_TYPES %ld\n", num_cells);

	for(std::size_t cell=0; cell<num_cells; ++cell){
		// '7' is for a general polygon
		fprintf(file, "7\n");
	}

	// Part (5): Dataset attributes

	// scalars cell variables
	assert(cell_variable_names.size() == cell_variables.size());
	for(std::size_t var_index=0; var_index<cell_variable_names.size(); ++var_index){
		if(var_index==0) fprintf(file, "CELL_DATA %ld\n", num_cells);
		auto const& var = cell_variables[var_index];
		assert(var.size() == num_cells);
		fprintf(file, "SCALARS %s double\n", cell_variable_names[var_index].c_str());
		fprintf(file, "LOOKUP_TABLE default\n");
		for(double const v : var){
			fprintf(file, "%1.10E\n", v);
		}
	}

	// // cell materials - to be used by VisIt
	// // https://www.visitusers.org/index.php?title=Materials_in_VTK_files#:~:text=VTK%20does%20not%20support%20materials,or%201%20material%20per%20cell.
	// if(cell_materials.size() > 0){
	// 	fprintf(file, "FIELD FieldData 1\n");
	// 	fprintf(file, "material 1 %ld int\n", num_cells);
	// 	for(auto const m : cell_materials){
	// 		fprintf(file, "%ld\n", m);
	// 	}
	// }

	// // vector vertex variables
	// assert(vertex_vector_names.size() == vertex_vectors.size());
	// for(std::size_t var_index=0; var_index<vertex_vector_names.size(); ++var_index){
	// 	if(var_index==0) fprintf(file, "POINT_DATA %ld\n", num_vertices);
	// 	auto const& var = vertex_vectors[var_index];
	// 	assert(var.size() == num_vertices);
	// 	fprintf(file, "VECTORS %s double\n", vertex_vector_names[var_index].c_str());
	// 	for(auto const& v : var){
	// 		fprintf(file, "%1.10E %1.10e 0\n", v.x, v.y);
	// 	}
	// }

	// close the file
	fclose(file);

    printf("writing vtk complete\n");


}
} //namespace