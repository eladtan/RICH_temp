#include "hdf5_utils.hpp"
#include "utils.hpp"

HDF5Shortcut::HDF5Shortcut(const string& fname) :
	fname_(fname), double_data_(), int_data_() {}

HDF5Shortcut& HDF5Shortcut::operator()(const string& field_name,
	const vector<double>& data_set)
{
	double_data_.push_back(pair<string, vector<double> >
		(field_name, data_set));
	return *this;
}

HDF5Shortcut& HDF5Shortcut::operator()(const string& field_name,
	const vector<int>& data_set)
{
	int_data_.push_back(pair<string, vector<int> >
		(field_name, data_set));
	return *this;
}

void write_std_vector_to_hdf5
(hid_t group_id,
	const vector<double>& data,
	const string& caption)
{
	write_std_vector_to_hdf5(group_id, data, caption, H5T_NATIVE_DOUBLE);
}

void write_std_vector_to_hdf5
(hid_t group_id,
	const vector<int>& data,
	const string& caption)
{
	write_std_vector_to_hdf5(group_id, data, caption, H5T_NATIVE_INT);
}

void write_std_vector_to_hdf5(hid_t group_id, const vector<size_t>& data, const string& caption)
{
	write_std_vector_to_hdf5(group_id, data, caption, H5T_NATIVE_ULLONG);
}


HDF5Shortcut::~HDF5Shortcut(void)
{
	hid_t file = H5Fcreate(fname_.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
	for (size_t i = 0; i < double_data_.size(); ++i)
		write_std_vector_to_hdf5(file, double_data_[i].second, double_data_[i].first);
	for (size_t i = 0; i < int_data_.size(); ++i)
		write_std_vector_to_hdf5(file, int_data_[i].second, int_data_[i].first);
	H5Fclose(file);
}
