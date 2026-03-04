/*! \file time_step_function3D.hpp
  \brief Abstract class for time step calculator
  \author Almog Yalinewich
 */

#ifndef TIME_STEP_FUNCTION3D_HPP
#define TIME_STEP_FUNCTION3D_HPP 1

#include "3D/tessellation/Tessellation3D.hpp"
#include "computational_cell.hpp"
#include "../two_dimensional/computational_cell_2d.hpp"
#include "../common/equation_of_state.hpp"

 //! \brief Abstract class for time step calculator
class TimeStepFunction3D
{
public:

	/*! \brief Calculates the time step
	  \param tess Tessellation
	  \param cells Computational cells
	  \param eos Equation of state
	  \param face_velocities The velocities of the faces
	  \param time The simulation time
	  \param tracerstickernames The names of the stickers and tracers
	  \return Time step
	 */
	virtual double operator()(const Tessellation3D& tess, const vector<ComputationalCell3D>& cells,
		const EquationOfState& eos, const vector<Vector3D>& face_velocities, const double time) = 0;

	//! \brief Class destructor
	virtual ~TimeStepFunction3D(void);

	virtual void SetTimeStep(double dt) = 0;

	virtual double GetTimeStep(void) const = 0;

	virtual double SuggestTimeStep(void) const = 0;
};

#endif  // TIME_STEP_FUNCTION3D_HPP
