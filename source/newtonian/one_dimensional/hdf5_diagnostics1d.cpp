#include "hdf5_diagnostics1d.hpp"

using std::vector;

namespace {
  void write_std_vector_to_hdf5
  (hid_t file_id,
   vector<double> const& num_list,
   string const& caption)
  {
    hsize_t dimsf[1];
    dimsf[0] = num_list.size();
    hid_t dataspace = H5Screate_simple(1, dimsf, nullptr);

    hid_t dataset = H5Dcreate2(file_id, caption.c_str(), H5T_NATIVE_DOUBLE,
                               dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    H5Dwrite(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &num_list.front());

    H5Dclose(dataset);
    H5Sclose(dataspace);
  }
}

void diagnostics1d::write_snapshot_to_hdf5
(hdsim1D const& sim,
 string const& fname)
{
  hid_t file = H5Fcreate(fname.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

  {
    vector<double> time_vector(1,0);
    time_vector[0] = sim.GetTime();
    write_std_vector_to_hdf5(file, time_vector, "time");
  }

  {
    vector<double> grid_vector(size_t(sim.GetCellNo()));
    for(size_t i=0;i<static_cast<size_t>(sim.GetCellNo());++i)
      grid_vector[size_t(i)] = sim.GetCellCenter(i);
    write_std_vector_to_hdf5(file, grid_vector, "grid");
  }

  {
    vector<double> density_vector(size_t(sim.GetCellNo()));
    vector<double> pressure_vector(size_t(sim.GetCellNo()));
    vector<double> x_velocity_vector(size_t(sim.GetCellNo()));
    vector<double> y_velocity_vector(size_t(sim.GetCellNo()));
    for(size_t i=0;i<static_cast<size_t>(sim.GetCellNo());++i){
      density_vector[size_t(i)] = sim.GetCell(i).Density;
      pressure_vector[size_t(i)] = sim.GetCell(i).Pressure;
      x_velocity_vector[size_t(i)] = sim.GetCell(i).Velocity.x;
      y_velocity_vector[size_t(i)] = sim.GetCell(i).Velocity.y;
    }
    write_std_vector_to_hdf5(file, density_vector, "density");
    write_std_vector_to_hdf5(file, pressure_vector, "pressure");
    write_std_vector_to_hdf5(file, x_velocity_vector, "x_velocity");
    write_std_vector_to_hdf5(file, y_velocity_vector, "y_velocity");
  }

  H5Fclose(file);
}
