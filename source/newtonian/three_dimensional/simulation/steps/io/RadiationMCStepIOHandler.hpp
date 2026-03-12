#ifndef RADIATION_MC_STEP_IO_HANDLER_HPP
#define RADIATION_MC_STEP_IO_HANDLER_HPP

#include "PhysicsStepIOHandler.hpp"

class RadiationMCStepIOHandler : public PhysicsStepIOHandler
{
public:
    void dump(HDF5Writer &writer, const std::string &group, const PhysicsStep &step) const override;

    void load(const HDF5Reader &reader, const std::string &group, PhysicsStep &step) const override;
};

#endif // RADIATION_MC_STEP_IO_HANDLER_HPP
