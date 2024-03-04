#include "SeveralSources3D.hpp"

void SeveralSources3D::operator()(const Tessellation3D &tess, const vector<ComputationalCell3D> &cells,
                                  const vector<Conserved3D> &fluxes, const vector<Vector3D> &point_velocities, const double t, double dt,
                                  vector<Conserved3D> &extensives, const vector<Vector3D> & ustar_vec, const std::vector<std::pair<ComputationalCell3D, ComputationalCell3D> > & face_values) const
{
    size_t const Nforces = sources_.size();
    for(size_t i = 0; i < Nforces; ++i)
        sources_[i]->operator()(tess, cells, fluxes, point_velocities, t, dt, extensives);
}

double SeveralSources3D::SuggestInverseTimeStep(void) const
{
    size_t const Nforces = sources_.size();
    double max_inverse = 0;
    for(size_t i = 0; i < Nforces; ++i)
        max_inverse = std::max(max_inverse, sources_[i]->SuggestInverseTimeStep());
    return max_inverse;
}