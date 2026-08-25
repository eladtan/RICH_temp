#include "HydroStepIOHandler.hpp"
#include "PhysicsStepIOHandlerFactory.hpp"
#include "newtonian/three_dimensional/simulation/steps/HydroStep.hpp"

void HydroStepIOHandler::dump(HDF5Writer &writer, const std::string &group, const PhysicsStep &step) const
{
    const HydroStep &hydro = static_cast<const HydroStep &>(step);
    const size_t version = 1;
    writer.WriteElement(group + "/individual_mesh_version", version);
    writer.WriteElement(group + "/individual_mesh_target_ids",
                        hydro.getIndividualMeshTargetIDs());
}

void HydroStepIOHandler::load(const HDF5Reader &reader, const std::string &group, PhysicsStep &step) const
{
    const std::string target_path = group + "/individual_mesh_target_ids";
    if(!reader.Exists(target_path))
        return;

    std::vector<size_t> target_ids;
    reader.ReadElement(target_path, target_ids);
    static_cast<HydroStep &>(step).restoreIndividualMeshTargetIDs(target_ids);
}

namespace
{
    static bool reg = (PhysicsStepIO::registerHandler("hydro", std::make_unique<HydroStepIOHandler>()), true);
}
