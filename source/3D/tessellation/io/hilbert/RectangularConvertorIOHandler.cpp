#include "RectangularConvertorIOHandler.hpp"
#include "ConvertorIOHandlerFactory.hpp"
#include "ConvertorState.hpp"
#include <MeshDecomposer3D/hilbert/rectangular/HilbertRectangularConvertor3D.hpp>
#include "utils/hdf5/HDF5Writer.hpp"
#include "utils/hdf5/HDF5Reader.hpp"

void RectangularConvertorIOHandler::dump(HDF5Writer &writer, const std::string &group, const HilbertConvertor3D<Vector3D> &convertor) const
{
    ConvertorState state{convertor.getLL(), convertor.getUR(), convertor.getOrder()};
    writer.WriteElement(group + "/state", state);
}

std::shared_ptr<HilbertConvertor3D<Vector3D>> RectangularConvertorIOHandler::load(const HDF5Reader &reader, const std::string &group) const
{
    ConvertorState state;
    reader.ReadElement(group + "/state", state);
    return std::make_shared<HilbertRectangularConvertor3D<Vector3D>>(state.ll, state.ur, state.order);
}

namespace
{
    static bool reg = (ConvertorIO::registerHandler("rectangular",
        std::make_unique<RectangularConvertorIOHandler>()), true);
}
