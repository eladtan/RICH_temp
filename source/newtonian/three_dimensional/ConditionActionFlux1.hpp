/*! \file ConditionActionFlux1.hpp
  \brief First order flux calculator based on a series of conditions and actions
  \author Elad Steinberg
 */

#ifndef CONDITION_ACTION_FLUX1_HPP
#define CONDITION_ACTION_FLUX1_HPP 1


#include "flux_calculator_3d.hpp"
#include "../common/riemann_solver.hpp"
#include "../../misc/utils.hpp"
#include "SpatialReconstruction3D.hpp"
#include "../common/LagrangianHLLC3D.hpp"

//using namespace std;

//! \brief First order flux calculator based on a series of conditions and actions
class ConditionActionFlux1 : public FluxCalculator3D
{
public:

	//! \brief Determines the kind of interface
	class Condition3D
	{
	public:

		/*! \brief Checks if an interface satisfies a condition
		\param face_index The index of the face to check
		\param tess Tessellation
		\param cells Computational cells
		\param tracerstickernames The names of the tracers and stickers
		\return A pair of booleans. The first is whether the face satisfies a condition, and the second is an auxiliary variable.
		*/
		virtual pair<bool, bool> operator()(size_t face_index, const Tessellation3D& tess,
			const vector<ComputationalCell3D>& cells) const = 0;

		virtual ~Condition3D(void);
	};

	//! \brief Action taken to calculate flux
	class Action3D
	{
	public:

		/*! \brief Calculates flux
		\param face_index The index of the face
		\param tess Tessellation
		\param cells Computational cells
		\param eos Equation of state
		\param aux Auxiliary variable for assymetric problems (true means the relevant cell is on the left side, false mean right)
		\param face_velocity Velocity of the face
		\param res The flux
		\param time The time
		\param tracerstickernames The names of the tracers and stickers
		\param face_values The values of the primitives at the face
		*/
		virtual void operator()(size_t face_index, const Tessellation3D& tess, const Vector3D& face_velocity,
			const vector<ComputationalCell3D>& cells, const EquationOfState& eos, const bool aux, Conserved3D &res,
			double time, std::pair<ComputationalCell3D,ComputationalCell3D>
			const& face_values, std::vector<Vector3D> const & point_velocities, Vector3D & ustar) const = 0;

		virtual ~Action3D(void);

	  /*! \brief Resets data
	   */
		virtual void Reset(void) const {}
	};

	/*!
	\brief Class constructor
	 \param sequence Series of condition and action action pairs. Both have to be dynamically allocated pointers, and will be explicitly destructed upon descruction of the class
	 \param interp The interpolation class
	*/
	explicit ConditionActionFlux1(const vector<pair<const Condition3D*, const Action3D*> >& sequence,
		SpatialReconstruction3D const& interp);

	//! \brief Class destructor
	~ConditionActionFlux1(void) override;

	std::vector<std::pair<ComputationalCell3D, ComputationalCell3D> > operator()(vector<Conserved3D> &fluxes, const Tessellation3D& tess, const vector<Vector3D>& edge_velocities,
		const vector<ComputationalCell3D>& cells, const vector<Conserved3D>& extensives, const EquationOfState& eos,
		const double time, const double dt, std::vector<Vector3D> const & point_velocities, vector<Vector3D> & ustar_vec) const override
;

private:
	const vector<pair<const Condition3D*, const Action3D*> > sequence_;
	SpatialReconstruction3D const& interp_;
};

//! \brief Calculates flux between two regular bulk cells
class RegularFlux3D : public ConditionActionFlux1::Action3D
{
public:

	/*! \brief Class constructor
	\param rs Riemann solver
	*/
	explicit RegularFlux3D(const RiemannSolver3D& rs);

	void operator()(size_t face_index, const Tessellation3D& tess, const Vector3D& face_velocity,
		const vector<ComputationalCell3D>& cells, const EquationOfState& eos, const bool aux, Conserved3D &res,
		double time,std::pair<ComputationalCell3D, ComputationalCell3D>
		const& face_values, std::vector<Vector3D> const & point_velocities, Vector3D & ustar)const override;

private:

	const RiemannSolver3D& rs_;
};

//! \brief Calculates flux assuming rigid wall boundary conditions
class RigidWallFlux3D : public ConditionActionFlux1::Action3D
{
public:

	/*! \brief Class constructor
	\param rs Riemann solver
	*/
	explicit RigidWallFlux3D(const RiemannSolver3D& rs);

	void operator()(size_t face_index, const Tessellation3D& tess, const Vector3D& face_velocity,
		const vector<ComputationalCell3D>& cells, const EquationOfState& eos, const bool aux, Conserved3D &res,
		double time,std::pair<ComputationalCell3D, ComputationalCell3D>
		const& face_values, std::vector<Vector3D> const & point_velocities, Vector3D & ustar) const override;

private:
	const RiemannSolver3D& rs_;
};
//! \brief Calculates zero flux
class ZeroFlux3D : public ConditionActionFlux1::Action3D
{
public:

	/*! \brief Class constructor
	*/
	explicit ZeroFlux3D() { ; }

	void operator()(size_t face_index, const Tessellation3D& tess, const Vector3D& face_velocity,
		const vector<ComputationalCell3D>& cells, const EquationOfState& eos, const bool aux, Conserved3D& res,
		double time, std::pair<ComputationalCell3D, ComputationalCell3D>
		const& face_values, std::vector<Vector3D> const & point_velocities, Vector3D & ustar) const override;

};

//! \brief Estimate flux assuming free flow boundary conditions
class FreeFlowFlux3D : public ConditionActionFlux1::Action3D
{
public:

	/*! \brief Class constructor
	\param rs Riemann solver
	*/
	explicit FreeFlowFlux3D(const RiemannSolver3D& rs);

	void operator()(size_t face_index, const Tessellation3D& tess, const Vector3D& face_velocity,
		const vector<ComputationalCell3D>& cells, const EquationOfState& eos, const bool aux, Conserved3D &res,
		double time,std::pair<ComputationalCell3D, ComputationalCell3D>
		const& face_values, std::vector<Vector3D> const & point_velocities, Vector3D & ustar) const override;

private:
	const RiemannSolver3D& rs_;
};

//! \brief A flux scheme that minimises mass transfer between cells
class LagrangianFlux3D : public ConditionActionFlux1::Action3D
{
public:

	/*! \brief Class constructor
	\param rs Riemann solver with no mass flux
	\param rs2 Riemann solver with mass flux
	\param criteria The criteria for calculating mass flux
	*/
	explicit LagrangianFlux3D(const LagrangianHLLC3D& rs, double const max_velocity_scale_error=1e5) : rs_(rs), max_velocity_scale_error(max_velocity_scale_error) {}

	void operator()(size_t face_index, const Tessellation3D& tess, const Vector3D& face_velocity,
		const vector<ComputationalCell3D>& cells, const EquationOfState& eos, const bool aux, Conserved3D &res,
		double time, std::pair<ComputationalCell3D, ComputationalCell3D>
		const& face_values, std::vector<Vector3D> const & point_velocities, Vector3D & ustar) const override;

private:
	const LagrangianHLLC3D& rs_;
	double const max_velocity_scale_error;
};

//! \brief Checks if a certain face is a boundary face
class IsBoundaryFace3D : public ConditionActionFlux1::Condition3D
{
public:

	IsBoundaryFace3D(void);

	pair<bool, bool> operator()(size_t face_index, const Tessellation3D& tess,
		const vector<ComputationalCell3D>& cells)const override;
};

//! \brief Check if an interface is inside the domain
class IsBulkFace3D : public ConditionActionFlux1::Condition3D
{
public:

	IsBulkFace3D(void);

	pair<bool, bool> operator()(size_t face_index, const Tessellation3D& tess,
		const vector<ComputationalCell3D>& cells)const override;
};

class IsLagrangianFace3D : public ConditionActionFlux1::Condition3D
{
public:

	IsLagrangianFace3D(double const mass_fraction = 0.1);

	pair<bool, bool> operator()(size_t face_index, const Tessellation3D& tess,
		const vector<ComputationalCell3D>& cells)const override;
private:
	double const mass_fraction_;
};

//! \brief Determines if the interface is between a regular and a special cell
class RegularSpecialEdge3D : public ConditionActionFlux1::Condition3D
{
public:

	/*! \brief Class constructor
	\param sticker_name Sticker name
	*/
	explicit RegularSpecialEdge3D(const string& sticker_name);

	pair<bool, bool> operator()(size_t face_index, const Tessellation3D& tess,
		const vector<ComputationalCell3D>& cells) const override;

private:
	const string sticker_name_;
};

//! \brief Determines if the interface is between two special cells
class BothSpecialEdge3D : public ConditionActionFlux1::Condition3D
{
public:

	/*! \brief Class constructor
	\param sticker_name Sticker name
	*/
	explicit BothSpecialEdge3D(const string& sticker_name);

	pair<bool, bool> operator()(size_t face_index, const Tessellation3D& tess,
		const vector<ComputationalCell3D>& cells) const override;

private:
	const string sticker_name_;
};

//! \brief Determines if the interface is between two special cells
class TwoSpecialEdge3D : public ConditionActionFlux1::Condition3D
{
public:

	/*! \brief Class constructor
	\param sticker_name1 First sticker name
	\param sticker_name2 Second sticker name
	*/
	explicit TwoSpecialEdge3D(const string& sticker_name1, const string& sticker_name2);

	pair<bool, bool> operator()(size_t face_index, const Tessellation3D& tess,
		const vector<ComputationalCell3D>& cells) const override;

private:
	const string sticker_name1_;
	const string sticker_name2_;
};


#endif //CONDITION_ACTION_FLUX1_HPP
