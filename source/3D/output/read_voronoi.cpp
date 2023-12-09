#include "read3D.hpp"

std::vector<Vector3D> ReadVoronoiPoints(std::string const &filename)
{
  std::vector<Vector3D> res;
  H5File file(filename, H5F_ACC_RDONLY);
  Group read_location = file.openGroup("/");
  const vector<double> x = read_double_vector_from_hdf5(read_location, "mesh_point_x");
  const vector<double> y = read_double_vector_from_hdf5(read_location, "mesh_point_y");
  const vector<double> z = read_double_vector_from_hdf5(read_location, "mesh_point_z");
  size_t const N = x.size();
  for(size_t i = 0; i < N; ++i)
    res.push_back(Vector3D(x[i], y[i], z[i]));
  return res;
}