#include "StressForce.hpp"
#ifdef RICH_MPI
#include "mpi/mpi_commands.hpp"
#endif

namespace
{
    void calc_velocity_derivatives(std::vector<Mat33<double>> & res, const vector<ComputationalCell3D>&cells, const Tessellation3D& tess, const std::vector<Vector3D> & ustar_vec, const vector<Vector3D> &point_velocities, std::vector<std::pair<ComputationalCell3D, ComputationalCell3D> > const & face_values, const bool is_lagrangian)
    {
        size_t N = tess.GetPointNo();
        std::vector<Mat33<double>> tmp(N);
        bool is_first, is_second;
        std::vector<face_vec> const & faces = tess.GetAllCellFaces();
        double tracers_dot = 0;
        for(size_t face_idx = 0; face_idx<ustar_vec.size(); ++face_idx)
        {
            auto [first, second] = tess.GetFaceNeighbors(face_idx);
            is_first = (first < N) || (!tess.IsPointOutsideBox(first));
            is_second = (second < N) || (!tess.IsPointOutsideBox(second));
            const Vector3D r_ij = normalize(tess.GetMeshPoint(second) - tess.GetMeshPoint(first));
            const double A = tess.GetArea(face_idx);
            Vector3D ustar = ustar_vec[face_idx];

            if(is_first && is_second)
            {
                if(is_lagrangian)
                {
                    tracers_dot = 0;
                    for(size_t t = 0; t < cells[0].tracerNames.size(); ++t)
                        tracers_dot += cells[first].tracers[t] * cells[second].tracers[t];
                    if(tracers_dot < 0.9)
                        ustar = tess.CalcFaceVelocity(face_idx, point_velocities[first], point_velocities[second]);
                }
                for(int k = 0; k < 3; ++k)
                {
                    for(int l = 0; l < 3; ++l)
                    {
                        if(first < N)
                            tmp[first].AddAt(ustar[k]*r_ij[l]*A, l, k);
                        if(second < N)
                            tmp[second].AddAt(-ustar[k]*r_ij[l]*A, l, k);
                    }
                }
            }
            else
            {
                if (first < N)
                {
                    for(int k = 0; k < 3; ++k)
                    {
                        for(int l = 0; l < 3; ++l)
                        {
                            tmp[first].AddAt(ustar[k]*r_ij[l]*A, l, k);
                        }
                    }
                }                
                if (second < N)
                {
                    for(int k = 0; k < 3; ++k)
                    {
                        for(int l = 0; l < 3; ++l)
                        {
                            tmp[second].AddAt(-ustar[k]*r_ij[l]*A, l, k);
                        }
                    }
                }
            }
        }
        for(size_t cell = 0; cell < N; ++cell)
            res[cell] = tmp[cell]/tess.GetVolume(cell);
    }
};

 void StressForce::operator()(const Tessellation3D & tess, const vector<ComputationalCell3D> & cells,
    const vector<Conserved3D> &fluxes, const vector<Vector3D> & point_velocities, const double t, double dt,
    vector<Conserved3D> &extensives, const std::vector<Vector3D>& ustar_vec,
    std::vector<std::pair<ComputationalCell3D, ComputationalCell3D> > const & interp_values) const
    {
        std::vector<size_t> neighbors;
        size_t const N = tess.GetPointNo();
        std::vector<Mat33<double>> velocity_derivatives(N);
        Mat33<double> omega, sigma_star, sigmap1, sigma_rot, dstrain;
        Mat33<double> zeros_mat;
        double dEPS, beta;
        calc_velocity_derivatives(velocity_derivatives, cells, tess, ustar_vec, point_velocities, interp_values, is_lagrangian_);
        for(size_t cell = 0; cell < N; ++cell)
        {
            if(cells[cell].G > 1) //todo
            {
                dstrain = 0.5 * (velocity_derivatives[cell] + velocity_derivatives[cell].transpose()) * dt;
                omega = velocity_derivatives[cell] * dt - dstrain;

                sigma_rot = cells[cell].stress + cells[cell].stress * omega - omega * cells[cell].stress;
                sigma_rot *= std::sqrt((cells[cell].stress.J2())/(std::numeric_limits<double>::epsilon() + sigma_rot.J2()));
                sigma_star = sigma_rot + 2 * cells[cell].G * deviator(dstrain);

                beta = std::min(std::sqrt(2./(std::numeric_limits<double>::epsilon() + 3.*sigma_star.J2())) * cells[cell].Y0, 1.);
                sigmap1 = beta * sigma_star;
                dEPS = (beta == 1) ? 0. : std::sqrt((deviator(dstrain) - (sigmap1-sigma_rot)/(2.*cells[cell].G)).J2());
                extensives[cell].mass_stress = extensives[cell].mass*sigma_star;
                extensives[cell].mass_eps += extensives[cell].mass*dEPS;
                extensives[cell].mass_eps_dt = extensives[cell].mass*dEPS/dt;
            }
            else
            {
                extensives[cell].mass_stress = zeros_mat;
                extensives[cell].Eelast = 0;
            }
        }
        bool is_first, is_second;
        std::vector<face_vec> const & faces = tess.GetAllCellFaces();
        double tracers_dot = 0;
        for(size_t face_idx = 0; face_idx<ustar_vec.size(); ++face_idx)
        {
            auto [first, second] = tess.GetFaceNeighbors(face_idx);
            is_first = (first < N) || (!tess.IsPointOutsideBox(first));
            is_second = (second < N) || (!tess.IsPointOutsideBox(second));
            const double A = tess.GetArea(face_idx);
            ComputationalCell3D left = interp_values[face_idx].first;
            ComputationalCell3D right = interp_values[face_idx].second;
            
            if(fluxes[face_idx].mass > 0)
            {
                if(first < N)
                    extensives[first].mass_stress -= fluxes[face_idx].mass * dt * A * left.stress;
                if(second < N)
                    extensives[second].mass_stress += fluxes[face_idx].mass * dt * A * left.stress;
            }
            else
            {
                if(fluxes[face_idx].mass < 0)
                {
                    if(first < N)
                        extensives[first].mass_stress -= fluxes[face_idx].mass * dt * A * right.stress;
                    if(second < N)
                        extensives[second].mass_stress += fluxes[face_idx].mass * dt * A * right.stress;
                }
            }
        }
    }