#include "StressForce.hpp"

void StressForce::operator()(const Tessellation3D& tess, const vector<ComputationalCell3D>& cells,
		const vector<Conserved3D>& fluxes, const vector<Vector3D>& point_velocities, const double t, double dt,
			vector<Conserved3D> &extensives) const
{
    int N = tess.GetPointNo();
    vector<Mat33<double>> velocity_derivatives(N), strain_rate(N), omega(N), sigma_star(N), sigmap1(N);
    vector<double> beta(N);

    for (int i=0; i<N; i++)
    {
        velocity_derivatives[i].Set(lg.GetSlopesUnlimited()[i].xderivative.velocity.x, lg.GetSlopesUnlimited()[i].yderivative.velocity.x, lg.GetSlopesUnlimited()[i].zderivative.velocity.x, lg.GetSlopesUnlimited()[i].xderivative.velocity.y, lg.GetSlopesUnlimited()[i].yderivative.velocity.y, lg.GetSlopesUnlimited()[i].zderivative.velocity.y,lg.GetSlopesUnlimited()[i].xderivative.velocity.z, lg.GetSlopesUnlimited()[i].yderivative.velocity.z, lg.GetSlopesUnlimited()[i].zderivative.velocity.z);
        strain_rate[i] = 0.5*(velocity_derivatives[i] + velocity_derivatives[i].transpose());
        omega[i] = velocity_derivatives[i]-strain_rate[i];
        sigma_star[i] = cells[i].stress + dt*(2*cells[i].G*deviator(strain_rate[i]) + cells[i].stress*omega[i] - omega[i]*cells[i].stress);
        beta[i] = std::max(2.*cells[i].Y0*1./(3.*sigma_star[i].J2()), 1.);
        sigmap1[i] = beta[i]*sigma_star[i];
        extensives[i].mass_stress = extensives[i].mass*sigmap1[i];
    }

    std::vector<size_t> neighbors;
    face_vec faces;

    for (int i=0; i<N; i++)
    {
        faces = tess.GetCellFaces(i);
        tess.GetNeighbors(i, neighbors);
        size_t const Nneigh = neighbors.size();
        Vector3D const point = tess.GetMeshPoint(i);
        Vector3D force(0, 0, 0);
        for(size_t j = 0; j < Nneigh; ++j)
        {
            size_t const neighbor_j = neighbors[j];
            Vector3D r_ij = point - tess.GetMeshPoint(neighbor_j);
            r_ij *= 1.0 / abs(r_ij);
            double Emid = 0;           
            if(tess.IsPointOutsideBox(neighbor_j))
            {
                force += 0.5 * sigmap1[neighbor_j] *  r_ij * (tess.GetArea(faces[j]));
            }
            else
            {
                force += 0.5 * sigmap1[i] *  r_ij * (tess.GetArea(faces[j]));
            }
        }

        extensives[i].momentum += force*dt;
        extensives[i].energy += ScalarProd(force, point_velocities[i]) * dt;
        extensives[i].internal_energy += (1-beta[i])*strain_rate[i]%sigmap1[i]*dt + (1.-beta[i])*1./(2.*cells[i].G)*(sigmap1[i]%sigmap1[i]);
    }

} 