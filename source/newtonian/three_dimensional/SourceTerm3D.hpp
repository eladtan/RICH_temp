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

	/*! \brief Apply the source to the active cells of an individual event
	  \details The default runs operator() once per active cell, on a copy of
	  every extensive, with that cell's time step scaled by the phase (half or
	  full), and keeps only that cell's result.  That is O(N_active * N), and a
	  stateful source or one coupling cells can behave differently when called
	  repeatedly while other cells' updates are discarded.  A source reporting
	  SupportsIndividualTimeSteps should override it.
	*/
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

	/*! \brief Per-cell limits of a synchronized individual state whose mesh
	  changed outside an event (a box growth), before anything advances on it
	  \details The counterpart of SuggestIndividualTimeSteps for every owned
	  cell of the full mesh `tess` (cells and extensives in mesh order), in the
	  same units (the time-step function applies its source factor), reduced
	  into `limits` by minimum.  A source with an individual acceleration
	  cache also refreshes it: `accelerations` becomes the acceleration of
	  every owned cell, the one its next first half kick uses.  Collective
	  under MPI.  The default is SuggestIndividualTimeSteps' default, the
	  source's global limit for every cell; a source that overrides that must
	  override this too.
	*/
	virtual void SynchronizedIndividualLimits(const Tessellation3D& tess,
		const vector<ComputationalCell3D>& cells,
		const vector<Conserved3D>& extensives,
		double time,
		vector<double>& limits,
		vector<Vector3D>& accelerations) const;

	virtual ~SourceTerm3D(void);

	virtual bool SupportsIndividualTimeSteps(void) const { return false; }

	virtual bool SupportsPartialMesh(void) const { return false; }

	virtual bool UsesIndividualAccelerationCache(void) const { return false; }

	//! Whether RefreshIndividualAccelerations can refresh the acceleration
	//! cache (a conservative force whose acceleration evaluates arbitrary
	//! targets).
	virtual bool SupportsIndividualAccelerationRefresh(void) const { return false; }

	/*! \brief The acceleration-cache owner's acceleration of every owned cell
	  of the full mesh `tess` (cells and extensives in mesh order), without
	  limits; valid only when SupportsIndividualAccelerationRefresh() holds.
	  Collective under MPI.  The default supplies none.
	*/
	virtual void RefreshIndividualAccelerations(const Tessellation3D& /*tess*/,
		const vector<ComputationalCell3D>& /*cells*/,
		const vector<Conserved3D>& /*extensives*/,
		double /*time*/,
		vector<Vector3D>& /*accelerations*/) const {}

	/*! \brief Whether the first-half individual phase needs the interval-start mesh
	  The hydro step builds a mesh at the interval-start generator positions
	  only to serve this phase; the fluxes use the event mesh.  A source whose
	  first half is a kick from per-cell cached state answers false when every
	  active cell carries that state, and the build is skipped;
	  ApplyIndividualFirstHalfFromCache then applies the phase on canonical
	  owned-cell arrays.  The answer is local and the caller reduces it
	  collectively: one rank needing geometry makes every rank build.
	  \param context Canonical event context (owned cells, scheduler order)
	  \return true when ApplyIndividual with FirstHalf needs a mesh
	*/
	virtual bool IndividualFirstHalfNeedsGeometry(
		const IndividualStepContext& context) const;

	/*! \brief Apply the first-half individual phase from cached state
	  Valid only after IndividualFirstHalfNeedsGeometry returned false on
	  every rank.  The arrays are canonical: owned cells in scheduler order,
	  indexed directly by the active indices of `context`.
	  \param cells Canonical primitives
	  \param point_velocities Canonical generator velocities
	  \param time Interval-start time of the event
	  \param context Canonical event context
	  \param extensives Canonical conserved variables, updated in place
	*/
	virtual void ApplyIndividualFirstHalfFromCache(
		const vector<ComputationalCell3D>& cells,
		const vector<Vector3D>& point_velocities,
		double time,
		const IndividualStepContext& context,
		vector<Conserved3D>& extensives) const;

};

//! \brief No force
class ZeroForce3D : public SourceTerm3D
{
public:
	bool SupportsIndividualTimeSteps(void) const override { return true; }
	bool SupportsPartialMesh(void) const override { return true; }
	bool IndividualFirstHalfNeedsGeometry(
		const IndividualStepContext& /*context*/) const override { return false; }
	void ApplyIndividualFirstHalfFromCache(
		const vector<ComputationalCell3D>& /*cells*/,
		const vector<Vector3D>& /*point_velocities*/,
		double /*time*/,
		const IndividualStepContext& /*context*/,
		vector<Conserved3D>& /*extensives*/) const override {}
	//! No change; validates the active indices as the default does.
	void ApplyIndividual(const Tessellation3D& tess,
		const vector<ComputationalCell3D>& cells,
		const vector<Conserved3D>& fluxes,
		const vector<Vector3D>& point_velocities,
		double time,
		const IndividualStepContext& context,
		IndividualSourcePhase phase,
		vector<Conserved3D>& extensives) const override;
	void operator()(const Tessellation3D& /*tess*/, const vector<ComputationalCell3D>& /*cells*/,
		const vector<Conserved3D>& /*fluxes*/, const vector<Vector3D>& /*point_velocities*/, const double /*t*/, 
			double /*dt*/, vector<Conserved3D> &/*extensives*/) const override;
};

#endif //SOURCETERM3D_HPP
