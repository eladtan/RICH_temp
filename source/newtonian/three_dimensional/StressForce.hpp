/*! \file StressForce.hpp
\brief Adding force from the stress/strain tensor
\author Omri Reved
*/

#ifndef STRESSFORCE_HPP
#define STRESSFORCE_HPP 1

#include "SourceTerm3D.hpp"
#include "LinearGauss3D.hpp"

class StressForce : public SourceTerm3D
{
public:
    
    StressForce(LinearGauss3D &lg_i): lg(lg_i) {};
    ~StressForce();

    void operator()(const Tessellation3D& tess, const vector<ComputationalCell3D>& cells,
		const vector<Conserved3D>& fluxes, const vector<Vector3D>& point_velocities, const double t, double dt,
			vector<Conserved3D> &extensives) const override;

	double SuggestInverseTimeStep(void)const override;
private:
    LinearGauss3D & lg;
};



#endif