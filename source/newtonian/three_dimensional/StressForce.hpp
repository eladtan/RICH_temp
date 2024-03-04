/* \file StressForce.hpp
\brief Advancing the stress
*/

#ifndef STRESSFORCE_HPP
#define STRESSFORCE_HPP 1

#include "SourceTerm3D.hpp"
#include "LinearGauss3D.hpp"
#include "3D/elementary/Mat33.hpp"
#include "computational_cell.hpp"
#include <limits>

class StressForce : public SourceTerm3D
{
public:
    StressForce(const bool is_lagrangian = true) : is_lagrangian_(is_lagrangian) {};
    void operator()(const Tessellation3D & tess, const vector<ComputationalCell3D> & cells,
    const vector<Conserved3D> &fluxes, const vector<Vector3D> & point_velocityies, const double t, double dt,
    vector<Conserved3D> &extensives, const std::vector<Vector3D>& ustar_vec,
    std::vector<std::pair<ComputationalCell3D, ComputationalCell3D> > const & interp_values) const override;

    
    const bool is_lagrangian_;
};
#endif