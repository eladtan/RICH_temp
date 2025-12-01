#ifndef WRITE_VTU_HPP_
#define WRITE_VTU_HPP_

#include <vector>
#include <string>

#ifdef RICH_VTK

namespace write_vtu{

/**
 * @brief write a vtu snapshot file (in parallel mode a pvtu file is also written)
 * 
 * @param file_name 
 * @param x_vertices 
 * @param y_vertices 
 * @param cells_num_vertices number of vertices each cell has (represents cell connectivity)
 * @param cell_variable_names 
 * @param cell_variables 
 * @param cell_vectors_names 
 * @param cell_vectors_x 
 * @param cell_vectors_y 
 * @param time 
 * @param cycle 
 */
void write_vtu(std::string const& file_name,
			   std::vector<double> const& x_vertices,
			   std::vector<double> const& y_vertices,
			   std::vector<std::size_t> const& cells_num_vertices,

			   std::vector<std::string> const& cell_variable_names,
			   std::vector<std::vector<double>> const& cell_variables,

			   std::vector<std::string> const& cell_vectors_names,
			   std::vector<std::vector<double>> const& cell_vectors_x,
			   std::vector<std::vector<double>> const& cell_vectors_y,
			   double const time,
			   std::size_t cycle);

} //namespace

#endif // RICH_VTK

#endif /* WRITE_VTU_HPP_ */