#ifndef RECTANGULAR_CONVERTOR_IO_HANDLER_HPP
#define RECTANGULAR_CONVERTOR_IO_HANDLER_HPP

#include "ConvertorIOHandler.hpp"

class RectangularConvertorIOHandler : public ConvertorIOHandler
{
public:
    void dump(HDF5Writer &writer, const std::string &group, const HilbertConvertor3D<Vector3D> &convertor) const override;

    std::shared_ptr<HilbertConvertor3D<Vector3D>> load(const HDF5Reader &reader, const std::string &group) const override;
};

#endif // RECTANGULAR_CONVERTOR_IO_HANDLER_HPP
