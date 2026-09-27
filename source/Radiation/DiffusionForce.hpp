#ifndef DIFFUSION_FORCE_HPP
#define DIFFUSION_FORCE_HPP 1
#include "../newtonian/three_dimensional/SourceTerm3D.hpp"
#include "Diffusion.hpp"

class DiffusionForce : public SourceTerm3D
{
public:

    DiffusionForce(Diffusion const& diffusion, EquationOfState const& eos, bool const momentum_limit = true): diffusion_(diffusion),
      next_dt_(1e-6 * std::numeric_limits<double>::max()), eos_(eos), momentum_limit_(momentum_limit){}

	    void operator()(const Tessellation3D& tess,const vector<ComputationalCell3D>& cells,
	      const vector<Conserved3D>& fluxes,const vector<Vector3D>& point_velocities, const double t, double dt,
	      vector<Conserved3D> &extensives) const;

	    double SuggestInverseTimeStep(void) const;

	    bool SupportsIndividualTimeSteps(void) const override { return true; }
	    bool SupportsPartialMesh(void) const override { return true; }

	    void ApplyIndividual(const Tessellation3D& tess,
	      const vector<ComputationalCell3D>& cells,
	      const vector<Conserved3D>& fluxes,
	      const vector<Vector3D>& point_velocities,
	      double time,
	      const IndividualStepContext& context,
	      IndividualSourcePhase phase,
	      vector<Conserved3D>& extensives) const override;

	    void SuggestIndividualTimeSteps(const Tessellation3D& tess,
	      const vector<ComputationalCell3D>& cells,
	      const IndividualStepContext& context,
	      vector<double>& time_step_limits) const override;

	    // Its per-cell limits come from the last radiation-force evaluation of
	    // each cell and cannot be re-evaluated on a rebuilt mesh here: a box
	    // growth during individual stepping is refused (every rank throws).
	    void SynchronizedIndividualLimits(const Tessellation3D& tess,
	      const vector<ComputationalCell3D>& cells,
	      const vector<Conserved3D>& extensives,
	      double time,
	      vector<double>& limits,
	      vector<Vector3D>& accelerations) const override;

	private:
	    void ApplyImpl(const Tessellation3D& tess,
	      const vector<ComputationalCell3D>& cells,
	      double dt,
	      const IndividualStepContext* context,
	      IndividualSourcePhase phase,
	      vector<Conserved3D>& extensives) const;

	    Diffusion const& diffusion_;
	    mutable double next_dt_;
	    mutable vector<double> individual_time_step_limits_;
	    EquationOfState const& eos_;
	    bool const momentum_limit_;
};

#endif
