#include "MultigroupDiffusionForce.hpp"
#include "Diffusion.hpp" // CalcSingleFluxLimiter
#include "boost/math/special_functions/pow.hpp"

using boost::math::pow;

MultigroupDiffusionForce::MultigroupDiffusionForce(MultigroupDiffusion const& multigroup_diffusion,
                                                   EquationOfState const& eos,
                                                   bool const momentum_limit) : 
                                                                multigroup_diffusion_(multigroup_diffusion),
                                                                next_dt_(1e-6 * std::numeric_limits<double>::max()),
                                                                eos_(eos),
                                                                momentum_limit_(momentum_limit){}

void MultigroupDiffusionForce::operator()(Tessellation3D const& tess, 
                                          std::vector<ComputationalCell3D> const& cells,
                                          std::vector<Conserved3D> const& fluxes,
                                          std::vector<Vector3D> const& point_velocities, 
                                          double const t,
                                          double const dt,
                                          std::vector<Conserved3D>& extensives) const {
    int rank = 0;
#ifdef RICH_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif

    std::vector<Conserved3D> old_extensives(extensives);
    
    std::vector<std::size_t> neighbors;
    face_vec faces;
    Vector3D grad_Eg(0.0, 0.0, 0.0);
    
    std::size_t const N = tess.GetPointNo();
    std::vector<double> flux_limiter(N, 0.0);
    std::vector<double> R2_g(N, 0.0);
    std::vector<double> new_Eg(N, 0.0);

    std::vector<double> dEr(tess.GetTotalFacesNumber(), 0.0);


    if(N == 0){
        std::cout << "MultigroupDiffusionForce::operator(): Tessellation is empty!" << std::endl;
    }

    ComputationalCell3D dummy_cell;
    Vector3D dummy_v;

    // update extensive Eg
    for(std::size_t g=0; g < ENERGY_GROUPS_NUM; ++g){
        for(std::size_t i=0; i < N; ++i){
            new_Eg[i] = cells[i].Eg[g] * cells[i].density;
        }

#ifdef RICH_MPI
        MPI_exchange_data2(tess, new_Eg, true);
#endif
        // build R2_g 
        for(std::size_t i = 0; i < N; ++i){
            double const volume = tess.GetVolume(i);
            faces = tess.GetCellFaces(i);
            tess.GetNeighbors(i, neighbors);

            std::size_t Nneighbors = neighbors.size();

            Vector3D const r_i = tess.GetMeshPoint(i);

            for(std::size_t j=0; j < Nneighbors; ++j){
                std::size_t const neighbor_j = neighbors[j];
                
                Vector3D r_ij = tess.GetMeshPoint(neighbor_j) - r_i;
                r_ij *= 1.0 / abs(r_ij);

                double Eg_mid = 0.0;

                if(!tess.IsPointOutsideBox(neighbor_j)){
                    Eg_mid = 0.5 * (new_Eg[i] + new_Eg[neighbor_j]);
                } else {
                    multigroup_diffusion_.boundary_calculator.getOutsideValuesGroup(g, tess, i, neighbor_j, cells, new_Eg, Eg_mid, dummy_v);

                    Eg_mid += new_Eg[i];
                    Eg_mid *= 0.5;
                }

                grad_Eg += r_ij * (tess.GetArea(faces[j]) * Eg_mid);
            }

            grad_Eg *= -1.0 / (multigroup_diffusion_.length_scale_ * volume);

            dummy_cell = cells[i];
            dummy_cell.density *= multigroup_diffusion_.mass_scale_ / pow<3>(multigroup_diffusion_.length_scale_);
            
            double const Dg = multigroup_diffusion_.coefficient_calculator.CalcDiffusionCoefficientGroup(dummy_cell, g);
            double const lambda_g = multigroup_diffusion_.flux_limiter_ ? CG::CalcSingleFluxLimiter(grad_Eg, Dg, new_Eg[i]) : 1.0;
            
            flux_limiter[i] = lambda_g;
            
            R2_g[i] = multigroup_diffusion_.flux_limiter_ ? lambda_g / 3.0 + pow<2>(lambda_g * abs(grad_Eg) * Dg / (CG::speed_of_light*new_Eg[i])) : 1.0 / 3.0;

            if(not momentum_limit_){
                flux_limiter[i] = 1.0;
                R2_g[i] = 1.0 / 3.0;
            }
        }

#ifdef RICH_MPI
        MPI_exchange_data2(tess, R2_g, true);
#endif
        // calculate dEg and build dEr
        for(std::size_t i=0; i<N; ++i){
            faces = tess.GetCellFaces(i);
            
            tess.GetNeighbors(i, neighbors);
            std::size_t Nneighbors = neighbors.size();
            
            Vector3D const r_i = tess.GetMeshPoint(i);
            
            for(std::size_t j=0; j < Nneighbors; ++j){
                std::size_t const neighbor_j = neighbors[j];

                
                Vector3D r_ij = r_i - tess.GetMeshPoint(neighbor_j);
                r_ij *= 1.0 / abs(r_ij);
                
                Vector3D velocity_outside;
                double Eg_outside, R2_g_outside, density_outside;
                
                bool is_outside = tess.IsPointOutsideBox(neighbor_j);
                
                if(!is_outside){
                    velocity_outside = cells[neighbor_j].velocity;
                    Eg_outside = new_Eg[neighbor_j];
                    R2_g_outside = R2_g[neighbor_j];
                    density_outside = cells[neighbor_j].density;
                } else {
                    multigroup_diffusion_.boundary_calculator.getOutsideValuesGroup(g, tess, i, neighbor_j, cells, new_Eg, Eg_outside, velocity_outside);
                    R2_g_outside = R2_g[i];
                    density_outside = cells[i].density;
                }
                double dEg_ij = 0.0;
                if(((tess.GetFaceNeighbors(faces[j]).second == i) && fluxes[faces[j]].mass < 0.0) || ((tess.GetFaceNeighbors(faces[j]).first == i) && fluxes[faces[j]].mass > 0.0))
                    dEg_ij = -std::abs(fluxes[faces[j]].mass) * 0.5 * (1.0 - R2_g[i]) * cells[i].Eg[g] * tess.GetArea(faces[j]) * dt ;
                else
                    dEg_ij = std::abs(fluxes[faces[j]].mass) * 0.5 * (1.0 - R2_g_outside) * Eg_outside * tess.GetArea(faces[j]) * dt / density_outside;

                dEr[faces[j]] += dEg_ij;
                
                extensives[i].Eg[g] += dEg_ij;
            }
        }
    }

    std::vector<double> new_Er(N, 0.0);
    for(std::size_t i=0; i<N; ++i){
        new_Er[i] = cells[i].Erad * cells[i].density;
    }
    double max_Er = *std::max_element(new_Er.begin(), new_Er.end());

#ifdef RICH_MPI
    MPI_exchange_data2(tess, new_Er, true);
    MPI_Allreduce(MPI_IN_PLACE, &max_Er, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
#endif

    for(std::size_t i=0; i < N; ++i){
        faces = tess.GetCellFaces(i);

        tess.GetNeighbors(i, neighbors);
        std::size_t Nneighbors = neighbors.size();

        for(std::size_t j=0; j < Nneighbors; ++j){
            std::size_t const neighbor_j = neighbors[j];
            double sum_Eg = 0.0;
            double Eg_outside = 0.0;
            
            bool is_cell_i = (tess.GetFaceNeighbors(faces[j]).first == i && fluxes[faces[j]].mass > 0.0) || (tess.GetFaceNeighbors(faces[j]).first != i && fluxes[faces[j]].mass < 0.0);
            bool is_outside = tess.IsPointOutsideBox(neighbors[j]);

            for(std::size_t g=0; g < ENERGY_GROUPS_NUM; ++g){
                if(!is_outside){
                    Eg_outside = cells[neighbors[j]].Eg[g];
                } else {
                    multigroup_diffusion_.boundary_calculator.getOutsideValuesGroup(g, tess, i, neighbors[j], cells, new_Eg, Eg_outside, dummy_v);
                }

                if(is_cell_i){
                    sum_Eg += cells[i].Eg[g];
                } else {
                    sum_Eg += Eg_outside;
                }
            }

            double Er_outside = 0.0;
            if(!is_outside) {
                Er_outside = new_Er[neighbors[j]];
            } else {
                multigroup_diffusion_.boundary_calculator.getOutsideValuesGray(tess, i, neighbor_j, cells, new_Er, Er_outside, dummy_v);
            }

            double dEr_ij = 0.0;
            if(is_cell_i){
                dEr_ij = dEr[faces[j]]/sum_Eg * new_Er[i];
            } else {
                dEr_ij = dEr[faces[j]]/sum_Eg * Er_outside;

            }
            extensives[i].Erad += dEr_ij;
        }
    }

    double max_diff = 0.0;
    std::size_t max_loc = 0;
}

double MultigroupDiffusionForce::SuggestInverseTimeStep(void) const {
    return 1.0 / next_dt_;
}