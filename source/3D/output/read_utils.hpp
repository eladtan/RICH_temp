#ifndef READ3D_UTILS
#define READ3D_UTILS

#include <vector>
#include <string>
#include <H5Cpp.h>

using namespace H5;

template<class T>
inline std::vector<T> read_vector_from_hdf5(const Group &file, const std::string &caption, const DataType &datatype)
{
    DataSet dataset = file.openDataSet(caption);
    DataSpace filespace = dataset.getSpace();
    hsize_t dims_out[2];
    filespace.getSimpleExtentDims(dims_out, nullptr);
    const size_t NX = static_cast<size_t>(dims_out[0]);
    std::vector<T> result(NX);
    dataset.read(&result[0], datatype);
    return result;
}

inline std::vector<double> read_double_vector_from_hdf5(const Group &file, std::string const &caption)
{
    return read_vector_from_hdf5<double>(file, caption, PredType::NATIVE_DOUBLE);
}

inline std::vector<int> read_int_vector_from_hdf5(const Group &file, const std::string &caption)
{
    return read_vector_from_hdf5<int>(file, caption, PredType::NATIVE_INT);
}

inline std::vector<size_t> read_sizet_vector_from_hdf5(const Group &file, const std::string &caption)
{
    return read_vector_from_hdf5<size_t>(file, caption, PredType::NATIVE_ULLONG);
}

#endif // READ3D_UTILS