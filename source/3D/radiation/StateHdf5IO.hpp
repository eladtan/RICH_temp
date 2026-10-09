#ifndef RICH_STATE_HDF5_IO_HPP
#define RICH_STATE_HDF5_IO_HPP

// Exact HDF5 storage of restart state. Arrays are stored flat next to an
// explicit shape dataset (`<path>_shape`), and the data only when non-empty,
// so empty fields (polarization off, a single group, no learned cells)
// round-trip without depending on how HDF5 represents empty arrays.

#include <cstddef>
#include <string>
#include <vector>

#include "misc/universal_error.hpp"
#include "utils/hdf5/HDF5Reader.hpp"
#include "utils/hdf5/HDF5Writer.hpp"

namespace state_hdf5
{

template <typename T>
void WriteVector(HDF5Writer& writer, std::string const& path, std::vector<T> const& values)
{
    std::vector<size_t> const shape{values.size()};
    writer.WriteElement(path + "_shape", shape);
    if (!values.empty())
        writer.WriteElement(path, values);
}

template <typename T>
void WriteMatrix(HDF5Writer& writer, std::string const& path,
                 std::vector<std::vector<T>> const& values)
{
    size_t const rows = values.size();
    size_t const cols = rows > 0 ? values[0].size() : 0;
    std::vector<T> flat;
    flat.reserve(rows * cols);
    for (auto const& row : values) {
        if (row.size() != cols)
            throw UniversalError("Ragged restart-state matrix at " + path);
        flat.insert(flat.end(), row.begin(), row.end());
    }
    std::vector<size_t> const shape{rows, cols};
    writer.WriteElement(path + "_shape", shape);
    if (!flat.empty())
        writer.WriteElement(path, flat);
}

inline std::vector<size_t> ReadShape(HDF5Reader const& reader, std::string const& path,
                                     size_t expectedRank)
{
    if (!reader.Exists(path + "_shape"))
        throw UniversalError("Restart file is missing " + path);
    std::vector<size_t> shape;
    reader.ReadElement(path + "_shape", shape);
    if (shape.size() != expectedRank)
        throw UniversalError("Restart dataset " + path + " has the wrong rank");
    return shape;
}

// Reads a vector of any stored length.
template <typename T>
std::vector<T> ReadVector(HDF5Reader const& reader, std::string const& path)
{
    size_t const n = ReadShape(reader, path, 1)[0];
    std::vector<T> values;
    if (n == 0)
        return values;
    reader.ReadElement(path, values);
    if (values.size() != n)
        throw UniversalError("Restart dataset " + path + " is truncated");
    return values;
}

// Reads a vector that must be empty or hold exactly `expectedSize` entries.
template <typename T>
std::vector<T> ReadVector(HDF5Reader const& reader, std::string const& path, size_t expectedSize)
{
    std::vector<T> values = ReadVector<T>(reader, path);
    if (!values.empty() && values.size() != expectedSize) {
        UniversalError eo("Restart dataset size mismatch at " + path);
        eo.addEntry("stored", static_cast<double>(values.size()));
        eo.addEntry("expected", static_cast<double>(expectedSize));
        throw eo;
    }
    return values;
}

// Reads a matrix of any stored shape.
template <typename T>
std::vector<std::vector<T>> ReadMatrix(HDF5Reader const& reader, std::string const& path)
{
    std::vector<size_t> const shape = ReadShape(reader, path, 2);
    std::vector<std::vector<T>> values;
    if (shape[0] == 0)
        return values;
    values.assign(shape[0], std::vector<T>(shape[1]));
    if (shape[1] == 0)
        return values;
    std::vector<T> flat;
    reader.ReadElement(path, flat);
    if (flat.size() != shape[0] * shape[1])
        throw UniversalError("Restart dataset " + path + " is truncated");
    for (size_t i = 0; i < shape[0]; ++i)
        for (size_t j = 0; j < shape[1]; ++j)
            values[i][j] = flat[i * shape[1] + j];
    return values;
}

// Reads a matrix that must be empty or exactly rows x cols.
template <typename T>
std::vector<std::vector<T>> ReadMatrix(HDF5Reader const& reader, std::string const& path,
                                       size_t expectedRows, size_t expectedCols)
{
    std::vector<std::vector<T>> values = ReadMatrix<T>(reader, path);
    if (values.empty())
        return values;
    if (values.size() != expectedRows || values[0].size() != expectedCols) {
        UniversalError eo("Restart dataset shape mismatch at " + path);
        eo.addEntry("stored rows", static_cast<double>(values.size()));
        eo.addEntry("stored cols", static_cast<double>(values[0].size()));
        eo.addEntry("expected rows", static_cast<double>(expectedRows));
        eo.addEntry("expected cols", static_cast<double>(expectedCols));
        throw eo;
    }
    return values;
}

template <typename T>
T ReadScalar(HDF5Reader const& reader, std::string const& path)
{
    if (!reader.Exists(path))
        throw UniversalError("Restart file is missing " + path);
    T value{};
    reader.ReadElement(path, value);
    return value;
}

} // namespace state_hdf5

#endif // RICH_STATE_HDF5_IO_HPP
