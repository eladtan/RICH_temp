#include "StressForce.hpp"
#ifdef RICH_MPI
#include "../../mpi/mpi_commands.hpp"
#endif
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
    for (size_t j=0; j<Nneigh; ++j)
    {
        size_t neigh_j = neighbors[j];
        if (neigh_j < N || !tess.IsPointOutsideBox(neigh_j))
        {
            tracers_dot = 0;
            for (size_t t=0; t < cells[i].tracerNames.size(); t++)
            {
                tracers_dot += cells[i].tracers[t] * cells[neigh_j].tracers[t];
            }
            
            if (cells[neigh_j].G > 1e10) 
            {
                c_ij[j] = tess.GetCellCM(neigh_j);
                c_ij[j] += cell_cm;
                c_ij[j] *= -0.5;
                c_ij[j] += tess.FaceCM(faces[j]);
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
                        tmp.AddAt((cells[neigh_j].velocity[k] + cells[i].velocity[k]) * r_ij[l]*A, l, k);
                    }
                }
            }
            else
            {
                c_ij[j] = tess.GetCellCM(neigh_j);
                c_ij[j] += cell_cm;
                c_ij[j] *= -0.5;
                c_ij[j] += tess.FaceCM(faces[j]);
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
                        tmp.AddAt(2. * cells[i].velocity[k] * r_ij[l]*A, l, k);
                    }
                }
            }
        }
        else
        {
            c_ij[j] = tess.GetCellCM(neigh_j);
            c_ij[j] += cell_cm;
            c_ij[j] *= -0.5;
            c_ij[j] += tess.FaceCM(faces[j]);
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

            double perp_velocity = ScalarProd(cells[i].velocity, r_ij);
            // if (tess.GetCellCM(i).x > 0.)
            // {
            //     for(int k=0; k<3; k++)
            //     {
            //         for (int l=0; l<3; l++)
            //         {
            //             tmp.AddAt(2.*(cells[i].velocity[k]) * r_ij[l]*A, l, k);
            //         }
            //     }
            // }
            // else
            // {
                for(int k=0; k<3; k++)
                {
                    for (int l=0; l<3; l++)
                    {
                        tmp.AddAt(2.*(cells[i].velocity[k]-perp_velocity*r_ij[k]) * r_ij[l]*A, l, k);
                    }
                }
            // }
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

    Mat33<double> m_inv;
    m_inv.Set(m[4] * m[8] - m[5] * m[7], m[2] * m[7] - m[1] * m[8], m[1] * m[5] - m[2] * m[4],
              m[5] * m[6] - m[3] * m[8], m[0] * m[8] - m[2] * m[6], m[2] * m[3] - m[5] * m[0],
              m[3] * m[7] - m[6] * m[4], m[6] * m[1] - m[0] * m[7], m[4] * m[0] - m[1] * m[3]);
    m_inv *= 1./(2. * tess.GetVolume(i) * det);

    res = m_inv*tmp;
}


void StressForce::operator()(const Tessellation3D& tess, const vector<ComputationalCell3D>& cells,
		const vector<Conserved3D>& fluxes, const vector<Vector3D>& point_velocities, const double t, double dt,
			vector<Conserved3D> &extensives) const
{
    size_t const N = tess.GetPointNo();
    vector<Mat33<double>> dstrain(N), omega(N), sigma_star(N), sigmap1(N), velocity_derivatives(N), sigma_rot(N);
    Mat33<double> zeros_mat;
    double dstrain_pl;
    vector<double> beta(N), dEPS(N);
    vector<Vector3D> force_vec(N);

    for (size_t i=0; i<N; ++i)
    {
        if (cells[i].G > 1e10)
        {
            calc_velocity_derivatives(i, velocity_derivatives[i], cells, tess);
            dstrain[i] = 0.5*(velocity_derivatives[i] + velocity_derivatives[i].transpose())*dt;
            omega[i] = velocity_derivatives[i]*dt-dstrain[i];
            
            sigma_rot[i] = cells[i].stress - cells[i].stress*omega[i] + omega[i]*cells[i].stress;
            sigma_rot[i] *= std::sqrt((cells[i].stress.J2())/(1e-12 + sigma_rot[i].J2()));
            sigma_star[i] = sigma_rot[i] + 2*cells[i].G*deviator(dstrain[i]);

            beta[i] = std::min(sqrt(2./(1.0e-33+3.*(sigma_star[i].J2())))*cells[i].Y0, 1.);
            sigmap1[i] = beta[i]*sigma_star[i];
            dEPS[i] = sqrt((dstrain[i] - (sigmap1[i]-sigma_rot[i])/(2.*cells[i].G)).J2());
            extensives[i].mass_stress = extensives[i].mass*sigmap1[i];
            extensives[i].mass_eps += extensives[i].mass*dEPS[i];
            extensives[i].mass_eps_dot = extensives[i].mass * dEPS[i]/dt;
        }
        else
        {
            sigmap1[i] = zeros_mat;
            extensives[i].mass_stress = zeros_mat;
        }
    }
#ifdef RICH_MPI
    Mat33<double> mat_dummy;
    MPI_exchange_data(tess, sigmap1, true, &mat_dummy);
#endif
    std::vector<size_t> neighbors;
    face_vec faces;
    
    for (size_t i=0; i<N; ++i)
    {
        if (cells[i].G > 1e10)
        {
            faces = tess.GetCellFaces(i);
            tess.GetNeighbors(i, neighbors);
            size_t const Nneigh = neighbors.size();
            Vector3D const point = tess.GetMeshPoint(i);
            for(size_t j = 0; j < Nneigh; ++j)
            {
                size_t const neigh_j = neighbors[j];
                Vector3D r_ij = tess.GetMeshPoint(neigh_j) - point;
                r_ij *= 1.0 / abs(r_ij);
                if (neigh_j < N || !tess.IsPointOutsideBox(neigh_j))
                {
                    force_vec[i] += 0.5 * sigmap1[neigh_j] * r_ij * (tess.GetArea(faces[j]));
                }
                else
                {
                    force_vec[i] += 0.5 * sigmap1[i] * r_ij * (tess.GetArea(faces[j]));
                }
            }
            extensives[i].momentum += force_vec[i]*dt;
            extensives[i].energy += 0.5*(sigmap1[i]+sigma_rot[i])%dstrain[i]*tess.GetVolume(i);
            extensives[i].internal_energy += std::max(0.5*(sigmap1[i]+sigma_rot[i])%(dstrain[i]-1/(2*cells[i].G) * (sigmap1[i]-sigma_rot[i]))*tess.GetVolume(i), 0.);
        }

        if (extensives[i].internal_energy < 0 || extensives[i].energy < 0)
        {
            UniversalError eo("negative energy after stress");
            std::cout << "Bad cell in stress, cell " << i <<" dt "<<dt<<" ID "<<cells[i].ID<< std::endl;
                std::cout << "mass " << extensives[i].mass << " energy " << extensives[i].energy << " internalE " <<
                    extensives[i].internal_energy << " momentum" << abs(extensives[i].momentum) << " volume " << tess.GetVolume(i)
                    << std::endl;
                std::cout << "Old cell, density " << cells[i].density << " pressure " << cells[i].pressure << " vx " <<
                    cells[i].velocity.x << " vy " << cells[i].velocity.y << " vz " << cells[i].velocity.z << std::endl;
                for (size_t j = 0; j < ComputationalCell3D::tracerNames.size(); ++j)
                {
                std::cout << ComputationalCell3D::tracerNames[j] << " old cell " << cells[i].tracers[j] << 
                        " extensive "<<extensives[i].tracers[j]<<std::endl;
                }
            throw eo;
        }
    }
} 