/*! \file SourceTerm3D.hpp
\brief Abstract class for source terms
\author Elad Steinberg
*/

#ifndef SOURCETERM3D_HPP
#define SOURCETERM3D_HPP 1
#include "3D/tessellation/Tessellation3D.hpp"
#include "misc/utils.hpp"
#include "computational_cell.hpp"
#include "conserved_3d.hpp"
#include "newtonian/three_dimensional/simulation/IndividualTimeStep.hpp"

enum class IndividualSourcePhase
{
	Full,
	FirstHalf,
	SecondHalf
};

//! \brief Abstract class for external forces
class SourceTerm3D
{
public:
	/*!
	\brief Calcualtes the change in conserved variables done on a cell from a source term
	\param tess The tessellation
	\param cells The hydrodynmic variables of the cell
	\param fluxes The hydrodynamic fluxes
	\param point_velocities Velocities of the mesh generating points
	\param t Time
	\param dt The time step
	\param tracerstickernames The names of the tracers and stickers
	\param extensives The updates extenesives, given as input and output.
	*/
	virtual void operator()(const Tessellation3D& tess,const vector<ComputationalCell3D>& cells,
		const vector<Conserved3D>& fluxes,const vector<Vector3D>& point_velocities, const double t, double dt,
		vector<Conserved3D> &extensives) const = 0;

  /*! \brief Informs simulation about the time step
    \return Inverse of the time step (to allow for an infinite time step)
   */
	virtual double SuggestInverseTimeStep(void)const;

	virtual void ApplyIndividual(const Tessellation3D& tess,
		const vector<ComputationalCell3D>& cells,
		const vector<Conserved3D>& fluxes,
		const vector<Vector3D>& point_velocities,
		double time,
		const IndividualStepContext& context,
		IndividualSourcePhase phase,
		vector<Conserved3D>& extensives) const;

	virtual void SuggestIndividualTimeSteps(const Tessellation3D& tess,
		const vector<ComputationalCell3D>& cells,
		const IndividualStepContext& context,
		vector<double>& time_step_limits) const;

	virtual ~SourceTerm3D(void);

	virtual bool SupportsIndividualTimeSteps(void) const { return false; }

	virtual bool SupportsPartialMesh(void) const { return false; }

	virtual bool UsesIndividualAccelerationCache(void) const { return false; }

};

//! \brief No force
class ZeroForce3D : public SourceTerm3D
{
public:
	bool SupportsIndividualTimeSteps(void) const override { return true; }
	bool SupportsPartialMesh(void) const override { return true; }
	void operator()(const Tessellation3D& /*tess*/, const vector<ComputationalCell3D>& /*cells*/,
		const vector<Conserved3D>& /*fluxes*/, const vector<Vector3D>& /*point_velocities*/, const double /*t*/, 
			double /*dt*/, vector<Conserved3D> &/*extensives*/) const override;
};

#endif //SOURCETERM3D_HPP
