#include "RadiationMCStepIOHandler.hpp"
#include "PhysicsStepIOHandlerFactory.hpp"
#include "3D/output/MC/read_write_particles.hpp"
#include "newtonian/three_dimensional/simulation/steps/RadiationMCStep.hpp"

void RadiationMCStepIOHandler::dump(HDF5Writer &writer, const std::string &group, const PhysicsStep &step) const
{
    const RadiationMCStep &mc = static_cast<const RadiationMCStep &>(step);
    writer.WriteElement(group + "/particles", mc.getParticles());
    writer.WriteElement(group + "/Erad_time_avg", mc.getEradTimeAvg());
}

void RadiationMCStepIOHandler::load(const HDF5Reader &reader, const std::string &group, PhysicsStep &step) const
{
    RadiationMCStep &mc = static_cast<RadiationMCStep &>(step);
    if(reader.Exists(group + "/particles"))
    {
        reader.ReadElement(group + "/particles", mc.getParticles());
    }
    if(reader.Exists(group + "/Erad_time_avg"))
    {
        reader.ReadElement(group + "/Erad_time_avg", mc.getEradTimeAvg());
    }
}

namespace
{
    static bool reg = (PhysicsStepIO::registerHandler("radiation-mc", std::make_unique<RadiationMCStepIOHandler>()), true);
}
