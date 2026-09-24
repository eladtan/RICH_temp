#include "gpu/DeviceParticle.hpp"
#include "gpu/GreyIMCKernel.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>

using namespace STORM;
using namespace STORM::gpu;

struct Result
{
    DeviceParticle particle;
    double deposited = 0.0, integrated = 0.0;
    int events = 0, error = 0;
};

STORM_GPU_INLINE_FUNCTION
Result Run(const bool slab, const double vx)
{
    // Absorbing, nonscattering material admits a closed-form solution.
    // The narrow box causes many explicit y/z reflections.
    DeviceVec3 center[] = {{0.5, 0.005, 0.005}};
    DeviceVec3 normals[] = {{1,0,0},{-1,0,0},{0,1,0},{0,-1,0},{0,0,1},{0,0,-1}};
    std::size_t offsets[] = {0,6};
    double planes[] = {0,-1,0,-0.01,0,-0.01};
    cell_index_t neighbors[] = {1,2,3,4,5,6};
    std::uint8_t boundary[] = {1,1,1,1,1,1};
    std::uint8_t behavior[6];
    for(int i=0;i<6;++i) behavior[i] = static_cast<std::uint8_t>(DeviceBoundaryFaceBehavior::ReflectingRigid);
    double absorption[] = {0.02}, scattering[] = {0}, fleck[] = {std::numeric_limits<double>::quiet_NaN()};
    Result output;
    GreyIMCViews<DeviceVec3> views;
    views.grid.cellCount = 1;
    views.grid.cellCenters = center;
    views.grid.cellFaceOffsets = offsets;
    views.grid.normals = normals;
    views.grid.facePlaneOffsets = planes;
    views.grid.nextCellIndices = neighbors;
    views.grid.boundaryCrossings = boundary;
    views.grid.deviceBoundaryBehaviors = behavior;
    views.grid.slabTransport = slab;
    views.grid.slabUpperY = views.grid.slabUpperZ = 0.01;
    views.absorptionOpacities = absorption;
    views.scatteringOpacities = scattering;
    views.fleckFactors = fleck;
    views.speedOfLight = 1;
    views.pendingMaterialEnergy = &output.deposited;
    views.pendingRadiationEnergy = &output.integrated;
    auto &p = output.particle;
    p.location = {0.5,0.002,0.004};
    p.velocity = {vx,-0.4,transport::Sqrt(0.84-vx*vx)};
    p.weight = p.initialWeight = 2;
    p.timeLeft = 0.9;
    p.rngKey = 17;
    for(int i=0;i<1000;++i)
    {
        const auto result = transport::AdvanceIMC(p, views);
        ++output.events;
        if(result.error != TransportError::None) { output.error = 1; break; }
        if(result.step.change == ParticleStatus::DONE) break;
    }
    return output;
}

int main() { auto r=Run(true,0.3); std::cout << "error="<<r.error<<" weight="<<r.particle.weight<<" deposited="<<r.deposited<<" integrated="<<r.integrated<<" events="<<r.events<<"\n"; }
