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
    StressForce(LinearGauss3D const& lg, int N, vector<double> G_arr, vector<double> Y_arr): lg_(lg), G0_arr_(G_arr), Y0_arr_(Y_arr), G_cells_arr_(N), Y_cells_arr_(N) {force_vec.resize(N);};
    void operator()(const Tessellation3D &tess, const vector<ComputationalCell3D> &cells,
                    const vector<Conserved3D> &fluxes, const vector<Vector3D> &point_velocities, const double t, double dt,
                    vector<Conserved3D> &extensives) const override;

    void calc_velocity_derivatives(size_t i, Mat33<double>& res, const vector<ComputationalCell3D>& cells, const Tessellation3D& tess) const;

    mutable std::vector<Vector3D> force_vec;
    mutable std::vector<double> G_cells_arr_;

private:
    LinearGauss3D const& lg_;
    std::vector<double> G0_arr_;
    std::vector<double> Y0_arr_;
    mutable std::vector<double> Y_cells_arr_;
};
#endif