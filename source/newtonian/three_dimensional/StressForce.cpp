#include "StressForce.hpp"

void StressForce::calc_velocity_derivatives(size_t i, Mat33<double> &res, const vector<ComputationalCell3D>& cells, const Tessellation3D& tess) const
{
    Mat33<double> tmp;
    std::array<double, 9>  m;
    std::fill_n(m.begin(), 9, 0.0);
    std::vector<size_t> neighbors;
    tess.GetNeighbors(i, neighbors);
    size_t const Nneigh = neighbors.size();
    std::vector<Vector3D> c_ij(Nneigh);
    Vector3D center = tess.GetMeshPoint(i);
    Vector3D cell_cm = tess.GetCellCM(i);
    face_vec faces = tess.GetCellFaces(i);
    double tracers_dot = 0;
    size_t N = tess.GetPointNo();

    for (size_t j=0; j<Nneigh; j++)
    {
        size_t neigh_j = neighbors[j];
        if (neigh_j < N)
        {
            for (size_t t=0; t < cells[i].tracerNames.size(); t++)
            tracers_dot += cells[i].tracers[t] * cells[neigh_j].tracers[t];

            if (tracers_dot > .25) /////////////////////////
            {
                c_ij[j] = tess.GetCellCM(neigh_j);
                c_ij[j] += cell_cm;
                c_ij[j] *= -0.5;
                const Vector3D r_ij = normalize(tess.GetMeshPoint(neigh_j) - center);
                const double A = tess.GetArea(faces[j]);
                m[0] -= c_ij[j].x*r_ij.x*A;
                m[1] -= c_ij[j].y*r_ij.x*A;
                m[2] -= c_ij[j].z*r_ij.x*A;
                m[3] -= c_ij[j].x*r_ij.y*A;
                m[4] -= c_ij[j].y*r_ij.y*A;
                m[5] -= c_ij[j].z*r_ij.y*A;
                m[6] -= c_ij[j].x*r_ij.z*A;
                m[7] -= c_ij[j].y*r_ij.z*A;
                m[8] -= c_ij[j].z*r_ij.z*A;

                for(int k=0; k<3; k++)
                {
                    for (int l=0; l<3; l++)
                    {
                        tmp.AddAt((cells[neigh_j].velocity(k) + cells[i].velocity(k)) * r_ij(l)*A, k, l);
                    }
                }
            }
            else
            {
                c_ij[j] = tess.GetCellCM(neigh_j);
                c_ij[j] += cell_cm;
                c_ij[j] *= -0.5;
                const Vector3D r_ij = normalize(tess.GetMeshPoint(neigh_j) - center);
                const double A = tess.GetArea(faces[j]);
                m[0] -= c_ij[j].x*r_ij.x*A;
                m[1] -= c_ij[j].y*r_ij.x*A;
                m[2] -= c_ij[j].z*r_ij.x*A;
                m[3] -= c_ij[j].x*r_ij.y*A;
                m[4] -= c_ij[j].y*r_ij.y*A;
                m[5] -= c_ij[j].z*r_ij.y*A;
                m[6] -= c_ij[j].x*r_ij.z*A;
                m[7] -= c_ij[j].y*r_ij.z*A;
                m[8] -= c_ij[j].z*r_ij.z*A;

                for(int k=0; k<3; k++)
                {
                    for (int l=0; l<3; l++)
                    {
                        tmp.AddAt(2. * cells[i].velocity(k) * r_ij(l)*A, k, l);
                    }
                }
            }
        }
        else
        {
            c_ij[j] = tess.GetCellCM(neigh_j);
            c_ij[j] += cell_cm;
            c_ij[j] *= -0.5;
            const Vector3D r_ij = normalize(tess.GetMeshPoint(neigh_j) - center);
            const double A = tess.GetArea(faces[j]);
            m[0] -= c_ij[j].x*r_ij.x*A;
            m[1] -= c_ij[j].y*r_ij.x*A;
            m[2] -= c_ij[j].z*r_ij.x*A;
            m[3] -= c_ij[j].x*r_ij.y*A;
            m[4] -= c_ij[j].y*r_ij.y*A;
            m[5] -= c_ij[j].z*r_ij.y*A;
            m[6] -= c_ij[j].x*r_ij.z*A;
            m[7] -= c_ij[j].y*r_ij.z*A;
            m[8] -= c_ij[j].z*r_ij.z*A;

            double perp_velocity = 2*ScalarProd(cells[i].velocity, r_ij);
            for(int k=0; k<3; k++)
            {
                for (int l=0; l<3; l++)
                {
                    tmp.AddAt((cells[i].velocity(k)-perp_velocity*r_ij(k)) * r_ij(l)*A, k, l);
                }
            }
        }
    }

    double v_inv = 1.0 / tess.GetVolume(i);
		for (size_t j = 0; j < 9; ++j)
			m[j] *= v_inv;
		m[0] += 1;
		m[4] += 1;
		m[8] += 1;
		// Find the det
		const double det = -m[2] * m[4] * m[6] + m[1] * m[5] * m[6] + m[2] * m[3] * m[7] - m[0] * m[5] * m[7] -
			m[1] * m[3] * m[8] + m[0] * m[4] * m[8];

    Mat33<double> m_invT;
    m_invT.Set(m[4] * m[8] - m[5] * m[7], m[5] * m[6] - m[3] * m[8], m[3] * m[7] - m[6] * m[4],
               m[2] * m[7] - m[1] * m[8], m[0] * m[8] - m[2] * m[6], m[6] * m[1] - m[0] * m[7],
               m[1] * m[5] - m[2] * m[4], m[2] * m[3] - m[5] * m[0], m[4] * m[0] - m[1] * m[3]);
    m_invT *= 1/(2 * tess.GetVolume(i) * det);

    res = tmp*m_invT;
}


void StressForce::operator()(const Tessellation3D& tess, const vector<ComputationalCell3D>& cells,
		const vector<Conserved3D>& fluxes, const vector<Vector3D>& point_velocities, const double t, double dt,
			vector<Conserved3D> &extensives) const
{
    size_t const N = tess.GetPointNo();
    vector<Mat33<double>> velocity_derivatives(N), strain_rate(N), omega(N), sigma_star(N), sigmap1(N);
    vector<double> beta(N);

    for (size_t i=0; i<N; i++)
    {
        calc_velocity_derivatives(i, velocity_derivatives[i], cells, tess);
        strain_rate[i] = 0.5*(velocity_derivatives[i] + velocity_derivatives[i].transpose());
        omega[i] = velocity_derivatives[i]-strain_rate[i];
        sigma_star[i] = cells[i].stress + dt*(2*cells[i].G*deviator(strain_rate[i]) + cells[i].stress*omega[i] - omega[i]*cells[i].stress);
        beta[i] = std::min(sqrt(2./(1.0e-33+3.*sigma_star[i].J2()))*cells[i].Y0, 1.0e0);
        sigmap1[i] = beta[i]*sigma_star[i];
        extensives[i].mass_stress = extensives[i].mass*sigmap1[i];
    }

    std::vector<size_t> neighbors;
    face_vec faces;

    for (size_t i=0; i<N; i++)
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
            if(!tess.IsPointOutsideBox(neighbor_j))
            {
                force += 0.5 * sigmap1[neighbor_j] *  r_ij * (tess.GetArea(faces[j]));
            }
            else
            {
                force += 0.5 * sigmap1[i] *  r_ij * (tess.GetArea(faces[j]));
            }
        }

        extensives[i].momentum += force*dt;
        extensives[i].energy += ScalarProd(force, point_velocities[i]) * dt + strain_rate[i]%sigmap1[i]*dt;
        extensives[i].internal_energy += strain_rate[i]%sigmap1[i]*dt - 1./(4.*cells[i].G)*(sigmap1[i]%sigmap1[i]-cells[i].stress%cells[i].stress);
    }

} 