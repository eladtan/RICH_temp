#include "HydroStepIOHandler.hpp"
#include "PhysicsStepIOHandlerFactory.hpp"

void HydroStepIOHandler::dump(HDF5Writer &writer, const std::string &group, const PhysicsStep &step) const
{}

void HydroStepIOHandler::load(const HDF5Reader &reader, const std::string &group, PhysicsStep &step) const
{}

namespace
{
    static bool reg = (PhysicsStepIO::registerHandler("hydro", std::make_unique<HydroStepIOHandler>()), true);
}
