#ifndef DIAGNOSTIC_APPENDIX_3D_HPP
#define DIAGNOSTIC_APPENDIX_3D_HPP

#include "newtonian/three_dimensional/hdsim_3d.hpp"
#include <vector>

//! \brief Appendix to data dump
class DiagnosticAppendix3D
{
public:

	/*! \brief Calculates additional data
	\param sim Hydrodynamic simulation
	\return Calculated data
	*/
	virtual vector<double> operator()(const HDSim3D& sim) const = 0;

	/*! \brief Returns the name of the new field
	*/

	/*! \brief Get appendix title
	  \return Title
	 */
	virtual string getName(void) const = 0;

	//! \brief Class destructor
	virtual ~DiagnosticAppendix3D(void) = default;
};

#endif // DIAGNOSTIC_APPENDIX_3D_HPP