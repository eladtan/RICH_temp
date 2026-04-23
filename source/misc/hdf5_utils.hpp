/*! \file hdf5_utils.hpp
  \brief Higher level hdf5 utilities
  \author Almog Yalinewich
 */

#ifndef HDF5_UTILS_HPP
#define HDF5_UTILS_HPP 1

#include <string>
#include <vector>
#include <algorithm>
#include <hdf5.h>

using std::string;
using std::vector;
using std::pair;

/*! \brief Master function for writing vectors to hdf5 files
  \param group_id Group or file handle to write into
  \param data Data to be written
  \param caption Name of dataset
  \param dt Data type id
 */
template<class T> void write_std_vector_to_hdf5
(hid_t group_id,
 const vector<T>& data,
 const string& caption,
 hid_t dt)
{
  hsize_t dimsf[1];
  dimsf[0] = static_cast<hsize_t>(data.size());
  hid_t dataspace = H5Screate_simple(1, dimsf, nullptr);

  hid_t plist = H5Pcreate(H5P_DATASET_CREATE);
  hsize_t chunk_size = dimsf[0];
  if(chunk_size > 100000)
    chunk_size = 100000;
  if(chunk_size == 0)
    chunk_size = 1;
  H5Pset_chunk(plist, 1, &chunk_size);
  H5Pset_deflate(plist, 6);

  hid_t dataset = H5Dcreate2(group_id, caption.c_str(), dt, dataspace,
                             H5P_DEFAULT, plist, H5P_DEFAULT);
  if(data.empty())
    H5Dwrite(dataset, dt, H5S_ALL, H5S_ALL, H5P_DEFAULT, nullptr);
  else
    H5Dwrite(dataset, dt, H5S_ALL, H5S_ALL, H5P_DEFAULT, &data[0]);

  H5Dclose(dataset);
  H5Pclose(plist);
  H5Sclose(dataspace);
}

/*! \brief Writes floating point data to hdf5
  \param group_id Group or file handle
  \param data Data to be written
  \param caption Name of dataset
 */
void write_std_vector_to_hdf5
(hid_t group_id,
 const vector<double>& data,
 const string& caption);

/*! \brief Writes integer data to hdf5
  \param group_id Group or file handle
  \param data Data to be written
  \param caption Name of dataset
 */
void write_std_vector_to_hdf5
(hid_t group_id,
 const vector<int>& data,
 const string& caption);

/*! \brief Writes size_t data to hdf5
  \param group_id Group or file handle
  \param data Data to be written
  \param caption Name of dataset
 */
void write_std_vector_to_hdf5(hid_t group_id, const vector<size_t>& data, const string& caption);

//! \brief Facilitates writing hdf5 files
class HDF5Shortcut
{
public:

  /*! \brief Class constructor
    \param fname Name of hdf5 file
   */
  explicit HDF5Shortcut(const string& fname);

  /*! \brief adds dataset
    \param field_name Name of dataset
    \param data_set Array of data
    \return Self reference
   */
  HDF5Shortcut& operator()(const string& field_name,
			   const vector<double>& data_set);
  
  /*! \brief adds dataset
    \param field_name Name of dataset
    \param data_set Array of data
    \return Self reference
   */
  HDF5Shortcut& operator()(const string& field_name,
			   const vector<int>& data_set);

  //! \brief Class destructor. This is the stage when the file is written
  ~HDF5Shortcut(void);

private:
  const string fname_;
  vector<pair<string,vector<double> > > double_data_;
  vector<pair<string,vector<int> > > int_data_;
};

#endif // HDF5_UTILS_HPP
