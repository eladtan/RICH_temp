#ifndef PHYSICS_STEP_IO_HANDLER_HPP
#define PHYSICS_STEP_IO_HANDLER_HPP

#include <string>
#include "newtonian/three_dimensional/simulation/steps/PhysicsStep.hpp"

class HDF5Writer;
class HDF5Reader;

class PhysicsStepIOHandler
{
public:
    virtual ~PhysicsStepIOHandler() = default;

    virtual void dump(HDF5Writer &writer, const std::string &group, const PhysicsStep &step) const = 0;

    virtual void load(const HDF5Reader &reader, const std::string &group, PhysicsStep &step) const = 0;
};

#endif // PHYSICS_STEP_IO_HANDLER_HPP
