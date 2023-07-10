/*! \file StressForce.hpp
\brief Adding force from the stress/strain tensor
\author Omri Reved
*/

#ifndef STRESSFORCE_HPP
#define STRESSFORCE_HPP 1

#include "SourceTerm3D.hpp"
#include "LinearGauss3D.hpp"
#include "../../3D/GeometryCommon/Mat33.hpp"
#include <limits>

class StressForce : public SourceTerm3D
{
public:
    
    StressForce(LinearGauss3D const& lg, int N):lg_(lg) {};
    void calc_velocity_derivatives(int i, Mat33<double>& res, const vector<ComputationalCell3D>& cells, const Tessellation3D & tess) const;
void operator()(const Tessellation3D &tess, const vector<ComputationalCell3D> &cells,
                    const vector<Conserved3D> &fluxes, const vector<Vector3D> &point_velocities, const double t, double dt,
                    vector<Conserved3D> &extensives) const override;

    void calc_velocity_derivatives(size_t i, Mat33<double>& res, const vector<ComputationalCell3D>& cells, const Tessellation3D& tess) const;
private:
    LinearGauss3D const& lg_;    
};
#endif