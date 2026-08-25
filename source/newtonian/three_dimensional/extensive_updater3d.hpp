/*! \file extensive_updater3d.hpp
\author Elad Steinberg
\brief Base class for extensive updater scheme
*/

#ifndef EXTENSIVE_UPDATER3D_HPP
#define EXTENSIVE_UPDATER3D_HPP 1

#include <stdexcept>

#include "computational_cell.hpp"
#include "conserved_3d.hpp"
#include "3D/tessellation/Tessellation3D.hpp"
#include "../common/equation_of_state.hpp"
#include "newtonian/three_dimensional/simulation/IndividualTimeStep.hpp"

using std::vector;

//! \brief Base class for extensive update scheme
class ExtensiveUpdater3D
{
public:

	/*! \brief Updates the extensive variables
	\param fluxes Fluxes
	\param tess Tessellation
	\param dt Time step
	\param cells Computational cells
	\param extensives Extensive variables
	\param tracerstickernames The names of the tracers and stickers
	\param time The time
	\param edge_velocities Edge velocities
	\param interp_values Interpolated values
	*/
	virtual void operator()(const vector<Conserved3D>& fluxes,const Tessellation3D& tess,
		const double dt,const vector<ComputationalCell3D>& cells,vector<Conserved3D>& extensives,double time, const vector<Vector3D>& edge_velocities,
		const vector<Vector3D>& point_velocities,
		std::vector<std::pair<ComputationalCell3D, ComputationalCell3D> > const& interp_values) const = 0;

	virtual bool SupportsIndividualTimeSteps(void) const{return false;}

	virtual void UpdateIndividual(const vector<Conserved3D>&,
		const Tessellation3D&,
		const IndividualStepContext&,
		const vector<ComputationalCell3D>&,
		vector<Conserved3D>&,
		double,
		const vector<Vector3D>&,
		const vector<Vector3D>&,
		const std::vector<std::pair<ComputationalCell3D, ComputationalCell3D> >&,
		const vector<ComputationalCell3D>* = nullptr,
		vector<Conserved3D>* = nullptr) const
	{
		throw std::runtime_error("Extensive updater does not support individual timesteps");
	}

	//! \brief Class constructor
	virtual ~ExtensiveUpdater3D(void);
};

#endif // EXTENSIVE_UPDATER3D_HPP
