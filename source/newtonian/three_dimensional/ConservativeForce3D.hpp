/*! \file ConservativeForce3D.hpp
\brief Abstract class for conservative force's acceleration
\author Elad Steinberg
*/

#ifndef CONSFORCE3D_HPP
#define CONSFORCE3D_HPP 1

#include <cstdint>

#include "SourceTerm3D.hpp"

//! \brief Physical acceleration
class Acceleration3D
{
public:
	/*!
	\brief Calculates the acceleration that the cells feel
	\param tess The tessellation
	\param cells The primitive cells
	\param fluxes The vector of the fluxes
	\param time The simulation time
	\param tracerstickernames The names of the tracers and stickers
	\param acc The calculated acceleration, given as output
	*/
	virtual void operator()(const Tessellation3D& tess,const vector<ComputationalCell3D>& cells,
				const vector<Conserved3D>& fluxes,const double time,
		vector<Vector3D> &acc) const = 0;

	virtual bool SupportsIndividualTargetEvaluation(void) const { return false; }

	//! Evaluate active-cell accelerations from the canonical gravity sources.
	//! Target cells and time let compound fields apply the same physical masks
	//! and time-dependent terms as their full-mesh operator.
	virtual void EvaluateIndividualTargets(
		std::pair<Vector3D, Vector3D> const& bounds,
		vector<Vector3D> const& source_points,
		vector<double> const& source_masses,
		vector<std::uint64_t> const& source_ids,
		vector<Vector3D> const& target_points,
		vector<ComputationalCell3D> const& target_cells,
		double time,
		vector<Vector3D>& acc) const;

	virtual ~Acceleration3D(void);
};

class ConstantAcceleration3D : public Acceleration3D
{
private:
	Vector3D const g_;
public:
	ConstantAcceleration3D(Vector3D const g);

	void operator()(const Tessellation3D& tess, const vector<ComputationalCell3D>& cells,
		const vector<Conserved3D>& fluxes, const double time, vector<Vector3D>& acc) const;

	bool SupportsIndividualTargetEvaluation(void) const override { return true; }

	void EvaluateIndividualTargets(
		std::pair<Vector3D, Vector3D> const& bounds,
		vector<Vector3D> const& source_points,
		vector<double> const& source_masses,
		vector<std::uint64_t> const& source_ids,
		vector<Vector3D> const& target_points,
		vector<ComputationalCell3D> const& target_cells,
		double time,
		vector<Vector3D>& acc) const override;
};

/*! \brief Class for conservative forces
\author Elad Steinberg
*/
class ConservativeForce3D : public SourceTerm3D
{
public:
	/*! \brief Class constructor
	\param acc The acceleration force
	\param mass_flux To include the mass flux term in the calculation or not (eq.  94 or eq. 82 in Arepo)
	*/
	explicit ConservativeForce3D(const Acceleration3D& acc, bool mass_flux = false);

	/*!
	\brief Class destructor
	*/
  ~ConservativeForce3D(void) override;

	void operator()(const Tessellation3D& tess, const vector<ComputationalCell3D>& cells,
		const vector<Conserved3D>& fluxes, const vector<Vector3D>& point_velocities, const double t, double dt,
			vector<Conserved3D> &extensives) const override;

	double SuggestInverseTimeStep(void)const override;

	bool SupportsIndividualTimeSteps(void) const override { return true; }
	bool SupportsPartialMesh(void) const override
	{return acc_.SupportsIndividualTargetEvaluation();}
	bool UsesIndividualAccelerationCache(void) const override { return true; }
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

private:
	const Acceleration3D& acc_;
	const bool mass_flux_;
	mutable double dt_;
	mutable std::vector<Vector3D> acc_buf_;
	mutable std::vector<double> individual_time_step_limits_;
};

#endif // CONSFORCE3D_HPP
