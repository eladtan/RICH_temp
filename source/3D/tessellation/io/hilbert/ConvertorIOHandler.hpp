#ifndef CONVERTOR_IO_HANDLER_HPP
#define CONVERTOR_IO_HANDLER_HPP

#include <memory>
#include <string>
#include "3D/elementary/Vector3D.hpp"
#include <MeshDecomposer3D/hilbert/HilbertConvertor3D.hpp>

class HDF5Writer;
class HDF5Reader;

class ConvertorIOHandler
{
public:
    virtual ~ConvertorIOHandler() = default;

    virtual void dump(HDF5Writer &writer, const std::string &group, const HilbertConvertor3D<Vector3D> &convertor) const = 0;

    virtual std::shared_ptr<HilbertConvertor3D<Vector3D>> load(const HDF5Reader &reader, const std::string &group) const = 0;
};

#endif // CONVERTOR_IO_HANDLER_HPP
