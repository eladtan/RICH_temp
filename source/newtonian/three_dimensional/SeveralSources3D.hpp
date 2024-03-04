/*! \file SeveralSources3D.hpp
\brief class for several forces
\author Elad Steinberg
*/

#ifndef SEVERALSOURCES3D_HPP
#define SEVERALSOURCES3D_HPP 1

#include "SourceTerm3D.hpp"
#include <memory>
class SeveralSources3D : public SourceTerm3D
{
public:
	/*! \brief Class constructor
	\param sources The vector of different sources to apply
	*/
	explicit SeveralSources3D(vector<std::shared_ptr<SourceTerm3D>> sources) : sources_(sources){}

	void operator()(const Tessellation3D& tess, const vector<ComputationalCell3D>& cells,
		const vector<Conserved3D>& fluxes, const vector<Vector3D>& point_velocities, const double t, double dt,
			vector<Conserved3D> &extensives, const vector<Vector3D> & ustar_vec, const std::vector<std::pair<ComputationalCell3D, ComputationalCell3D> > & face_values) const override;

	double SuggestInverseTimeStep(void)const override;

private:
	vector<std::shared_ptr<SourceTerm3D>> sources_;
};

#endif // SEVERALSOURCES3D_HPP
