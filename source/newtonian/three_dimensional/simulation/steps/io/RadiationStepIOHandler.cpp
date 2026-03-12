#include "RadiationStepIOHandler.hpp"
#include "PhysicsStepIOHandlerFactory.hpp"

void RadiationStepIOHandler::dump(HDF5Writer &writer, const std::string &group, const PhysicsStep &step) const
{
    (void)writer;
    (void)group;
    (void)step;
}

void RadiationStepIOHandler::load(const HDF5Reader &reader, const std::string &group, PhysicsStep &step) const
{
    (void)reader;
    (void)group;
    (void)step;
}

namespace
{
    static bool reg = (PhysicsStepIO::registerHandler("radiation", std::make_unique<RadiationStepIOHandler>()), true);
}
