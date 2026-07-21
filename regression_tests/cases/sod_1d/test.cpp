#include <cmath>
#include <fstream>
#include "../../../source/newtonian/common/hllc.hpp"
#include "../../../source/newtonian/common/ideal_gas.hpp"
#include "../../../source/newtonian/one_dimensional/eos_consistent1d.hpp"
#include "../../../source/newtonian/one_dimensional/eulerian1d.hpp"
#include "../../../source/newtonian/one_dimensional/hdf5_diagnostics1d.hpp"
#include "../../../source/newtonian/one_dimensional/hdsim.hpp"
#include "../../../source/newtonian/one_dimensional/plm1d.hpp"
#include "../../../source/newtonian/one_dimensional/rigid_wall_1d.hpp"
#include "../../../source/newtonian/one_dimensional/spatial_distribution1d.hpp"
#include "../../../source/newtonian/one_dimensional/zero_force_1d.hpp"
#include "../../../source/misc/universal_error.hpp"
#include "../../../source/misc/utils.hpp"

using namespace interpolations1d;
using namespace diagnostics1d;

namespace {

class SimData {
public:
  SimData():
    eos_(1.4),
    plm_(),
    interp_(plm_, eos_),
    rs_(),
    vm_(),
    bc_(),
    force_(),
    sim_(pg_,
         linspace(0.0, 1.0, 400),
         interp_,
         Step(1.0, 0.125, 0.5),
         Step(1.0, 0.1, 0.5),
         Uniform(0.0),
         Uniform(0.0),
         eos_,
         rs_,
         vm_,
         bc_,
         force_) {}

  hdsim1D& getSim() { return sim_; }

private:
  const SlabSymmetry1D pg_;
  const IdealGas eos_;
  PLM1D plm_;
  EOSConsistent interp_;
  const Hllc rs_;
  const Eulerian1D vm_;
  const RigidWall1D bc_;
  const ZeroForce1D force_;
  hdsim1D sim_;
};

void main_loop(hdsim1D& sim)
{
  const double tf = 0.2;
  while(sim.GetTime() < tf) {
    try {
      sim.TimeAdvance2();
    }
    catch(UniversalError& eo) {
      eo.addEntry("time", sim.GetTime());
      eo.addEntry("cycle", sim.GetCycle());
      reportError(eo);
      throw;
    }
  }
}

} // namespace

int main()
{
  SimData sim_data;
  hdsim1D& sim = sim_data.getSim();
  main_loop(sim);
  std::ofstream out("sod_profile.txt");
  for(int i = 0; i < sim.GetCellNo(); ++i) {
    Primitive const cell = sim.GetCell(static_cast<size_t>(i));
    out << sim.GetCellCenter(static_cast<size_t>(i)) << " "
        << cell.Density << " "
        << cell.Pressure << "\n";
  }
  out.close();
  return 0;
}
