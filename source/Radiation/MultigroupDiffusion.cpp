#include "Diffusion.hpp" // for CalcSingleFluxLimiter and FleckFactor
#include "MultigroupDiffusion.hpp"
// TODO: make a units namespace used by all the program 
#include "tau_matrix_calculator/src/units.hpp"

using boost::math::pow;

void fill_zero(std::vector<double>& vec) {
    std::fill(vec.begin(), vec.end(), 0.0);
}

void fill_zero(std::vector<std::vector<double>>& mat) {
    for(std::vector<double>& row : mat) {
        std::fill(row.begin(), row.end(), 0.0);
    }
}

MultigroupDiffusion::MultigroupDiffusion(std::vector<double> const& energy_groups_center_, 
                                         std::vector<double> const& energy_groups_boundary_,
                                         MultigroupDiffusionCoefficientCalculator const& coefficient_calc,
                                         MultigroupDiffusionBoundaryCalculator const& boundary_calc,
                                         EquationOfState const& eos,
                                         std::vector<std::string> const zero_cells,
                                         bool const flux_limiter,
                                         bool const hydro_on,
                                         bool const compton_on,
                                         bool const doppler_on,
                                         bool const mix_frame_on,
                                         double const minimum_temperature):
                                                                energy_groups_center(energy_groups_center_),
                                                                energy_groups_boundary(energy_groups_boundary_),
                                                                energy_groups_width(ENERGY_GROUPS_NUM, 0.0),
                                                                coefficient_calculator(coefficient_calc),
                                                                boundary_calculator(boundary_calc),
                                                                current_group(0),
                                                                gray(false),
                                                                cells_temp(),
                                                                cells_cgs(),
                                                                extensives_temp(),
                                                                sigma_absorption_group(ENERGY_GROUPS_NUM, std::vector<double>()),
                                                                sigma_scattering_group(ENERGY_GROUPS_NUM, std::vector<double>()),
                                                                planck_integal_group(ENERGY_GROUPS_NUM, std::vector<double>()),
                                                                sigma_absorption_planck(),
                                                                sigma_absorption_average(),
                                                                sigma_scattering_gray(),
                                                                fleck_factor(),
                                                                new_Eg(),
                                                                new_Eg_full(),
                                                                old_Eg(ENERGY_GROUPS_NUM, std::vector<double>()),
                                                                new_Er(),
                                                                new_Er_full(),
                                                                old_Er(),
                                                                old_Tm(),
                                                                max_abs_grad_E(),
                                                                max_neighbor_abs_grad_E(),
                                                                grad(),
                                                                lambda_face_gray(),
                                                                sigma_ratio_lambda_face_gray(),
                                                                lambda_cell_gray(),
                                                                sigma_ratio_lambda_cell_gray(),
                                                                doppler_on_(doppler_on),
                                                                mix_frame_on_(mix_frame_on),
                                                                minimum_temperature_(minimum_temperature),
                                                                R2(3, std::vector<double>()),
                                                                D(3, std::vector<double>()),
                                                                displayed_warning_(false),
                                                                tau_engine( energy_groups_center_,
                                                                            energy_groups_boundary_, 
                                                                            20000, // num of samples
                                                                            true), // force detailed balance
                                                                tau(ENERGY_GROUPS_NUM, std::vector<double>(ENERGY_GROUPS_NUM, 0.0)),
                                                                dtau_dUm(ENERGY_GROUPS_NUM, std::vector<double>(ENERGY_GROUPS_NUM, 0.0)),
                                                                S(ENERGY_GROUPS_NUM, std::vector<double>(ENERGY_GROUPS_NUM, 0.0)),
                                                                dSdUm(ENERGY_GROUPS_NUM, std::vector<double>(ENERGY_GROUPS_NUM, 0.0)),
                                                                n(ENERGY_GROUPS_NUM, 0.0),
                                                                cell_id_of_compton_matrices(std::numeric_limits<std::size_t>::max()),
                                                                Gammas(),
                                                                Q_vector(ENERGY_GROUPS_NUM, 0.0),
                                                                Q(0.0),
                                                                Upsilon_vector(ENERGY_GROUPS_NUM, 0.0),
                                                                Upsilon(0.0),
                                                                sum_dSdUm(ENERGY_GROUPS_NUM, 0.0),
                                                                compton_initialized_(false),
                                                                RadiationDriver(eos,
                                                                                zero_cells,
                                                                                flux_limiter,
                                                                                hydro_on,
                                                                                compton_on) {

    if(energy_groups_center.size() != ENERGY_GROUPS_NUM){
        std::cout << "bad energy_groups_center.size()" << std::endl;
        exit(1);
    }

    if(energy_groups_boundary.size() != ENERGY_GROUPS_NUM + 1){
        std::cout << "bad energy_groups_boundary.size()" << std::endl;
        exit(1);
    }

    for(std::size_t g=0; g < ENERGY_GROUPS_NUM; ++g){
        energy_groups_width[g] = energy_groups_boundary[g+1] - energy_groups_boundary[g];
    }
}

bool MultigroupDiffusion::prestep(Tessellation3D const& tess,
                                  std::vector<ComputationalCell3D> const& cells) const {
    auto const N = tess.GetPointNo();

    sigma_absorption_group = std::vector<std::vector<double>>(ENERGY_GROUPS_NUM, std::vector<double>(N, 0.0));
    sigma_scattering_group = std::vector<std::vector<double>>(ENERGY_GROUPS_NUM, std::vector<double>(N, 0.0));
    planck_integal_group = std::vector<std::vector<double>>(ENERGY_GROUPS_NUM, std::vector<double>(N, 0.0));
    
    new_Eg.resize(N, 0.0);
    new_Eg_full.resize(N, 0.0);
    
    sigma_absorption_planck.resize(N, 0.0);
    sigma_absorption_average.resize(N, 0.0);
    sigma_scattering_gray.resize(N, 0.0);
    fleck_factor.resize(N, 0.0);

    new_Er.resize(N, 0.0);
    new_Er_full.resize(N, 0.0);

    old_Er.resize(N, 0.0);
    old_Tm.resize(N, 0.0);

    max_abs_grad_E.resize(N, 0.0);
    max_neighbor_abs_grad_E.resize(N, 0.0);

    for(std::size_t i=0; i < N; ++i){
        old_Er[i] = cells[i].Erad * cells[i].density;
        old_Tm[i] = cells[i].temperature;
    }

    old_Eg.resize(N);
    for(std::size_t i=0; i < N; ++i){
        old_Eg[i].resize(ENERGY_GROUPS_NUM, 0.0);
        for(std::size_t g=0; g<ENERGY_GROUPS_NUM; ++g){
            old_Eg[i][g] = cells[i].Eg[g] * cells[i].density;
        }
    }

    auto const Nfaces = tess.GetTotalFacesNumber();
    grad.resize(Nfaces);
    
    lambda_face_gray.resize(Nfaces, std::pair<double, double>(0.0, 0.0));
    sigma_ratio_lambda_face_gray.resize(Nfaces, std::pair<double, double>(0.0, 0.0));

    lambda_cell_gray.resize(N, 0.0);
    sigma_ratio_lambda_cell_gray.resize(N, 0.0);

    std::vector<std::size_t> neighbors;
    face_vec faces; 
    for(std::size_t i=0; i < N; ++i){
        tess.GetNeighbors(i, neighbors);
        faces = tess.GetCellFaces(i);

        Vector3D CM_i = tess.GetCellCM(i); 

        auto const Nneighbors = neighbors.size();
        for(std::size_t j=0; j < Nneighbors; ++j){
            std::size_t const neighbor_j = neighbors[j];

            if(!tess.IsPointOutsideBox(neighbor_j)){
                if(i < neighbor_j){
                    Vector3D const CM_ij = CM_i - tess.GetCellCM(neighbor_j);
                    grad[faces[j]] = CM_ij * (1.0 / (length_scale_*ScalarProd(CM_ij, CM_ij)));
                }
            }
        }
    }

    R2 = std::vector<std::vector<double>>(3, std::vector<double>(N, 0.0));
    D  = std::vector<std::vector<double>>(3, std::vector<double>(N, 0.0));
    
    if(!compton_initialized_ and compton_on_){
        Vector tmp_grid = {1e-2, 1., 3., 4., 6., 10., 20., 30., 40., 60., 80., 100.};
        for(auto& temp : tmp_grid){
            temp *= units::kev_to_kelvin;
        }
        tau_engine.generate_tables(tmp_grid);
        compton_initialized_=true;
    }
    
    Gammas.resize(N, 0.0);

    return true;
}

bool MultigroupDiffusion::poststep() const {
    std::vector<ComputationalCell3D>().swap(cells_temp);
    std::vector<ComputationalCell3D>().swap(cells_cgs);
    std::vector<Conserved3D>().swap(extensives_temp);
    return true;
}

double MultigroupDiffusion::calculate_dt(double const dt,
                                         Tessellation3D& tess,
                                         std::vector<ComputationalCell3D>& cells) const {
    int rank = 0;
#ifdef RICH_MPI
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif

	double max_Er = *std::max_element(old_Er.begin(), old_Er.end());
    double max_Tm = *std::max_element(old_Tm.begin(), old_Tm.end());
    double max_rhoT = 0;
    auto const N = tess.GetPointNo();            
    for(size_t i=0; i < N; ++i)
        max_rhoT = std::max(max_rhoT, cells[i].density * cells[i].temperature);

#ifdef RICH_MPI
	MPI_Allreduce(MPI_IN_PLACE, &max_Er, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &max_Tm, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &max_rhoT, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
#endif	

    size_t const Nzero = zero_cells_.size();
	std::vector<size_t> zero_indeces;
	for(size_t i = 0; i < Nzero; ++i)
		zero_indeces.push_back(binary_index_find(ComputationalCell3D::stickerNames, zero_cells_[i]));
	double max_diff = std::numeric_limits<double>::min() * 100;
    int max_which = 0;
	int max_loc = 0;
	for(size_t i = 0; i < N; ++i)
	{
        int which_one = 0;
		bool to_calc = true;
		for(size_t j = 0; j < Nzero; ++j)
			if(cells[i].stickers[zero_indeces[j]])
				to_calc = false;
		if(not to_calc)
			continue;
		double const equlibrium_factor = 1.0; //std::abs(cells[i].temperature - std::pow(new_Er[i] / CG::radiation_constant, 0.25)) < 0.02 * cells[i].temperature ? 0.05 : 1;
		double diff = equlibrium_factor * std::abs(cells[i].Erad * cells[i].density - old_Er[i]) / (cells[i].Erad * cells[i].density+0.01*max_Er);

        double const temp = cells[i].density * equlibrium_factor* std::abs(cells[i].temperature - old_Tm[i]) / (cells[i].density * cells[i].temperature+1e-5*max_rhoT);
        
        if(temp > diff){
            which_one = 1;
            diff = temp;
        }

        for(std::size_t g=0; g<ENERGY_GROUPS_NUM; ++g){
            double const temp = equlibrium_factor* std::abs(cells[i].Eg[g]*cells[i].density - old_Eg[i][g]) / (cells[i].Eg[g]*cells[i].density + 0.01/ENERGY_GROUPS_NUM*max_Er);
            if(temp > diff){
                which_one = 2 + g;
                diff = temp;
            }
        }
		// if(fleck_factor[i] < 0.5)
			// diff *= 0.1;
		if(diff > max_diff)
		{
            max_which = which_one;
			max_diff = diff;
			max_loc = i;
		}
	}

	struct
    {
        double val;
        int mpi_id;
    }max_data;
    
    max_data.mpi_id = rank;
    max_data.val = max_diff;
#ifdef RICH_MPI
	MPI_Allreduce(MPI_IN_PLACE, &max_data, 1, MPI_DOUBLE_INT, MPI_MAXLOC, MPI_COMM_WORLD);
	max_diff = max_data.val;
	ComputationalCell3D cdummy;
	MPI_exchange_data(tess, cells, true, &cdummy);	
#endif
	if(rank == max_data.mpi_id)
	{
        // max_loc /= ENERGY_GROUPS_NUM;
		std::cout<<"Radiation time step ID "<<cells[max_loc].ID<<" old Er "<<old_Er[max_loc]<<" new Er "<<cells[max_loc].Erad * cells[max_loc].density<<
		" diff "<<max_diff<<" Tgas "<<cells[max_loc].temperature<<" Trad "<<std::pow(cells[max_loc].density * cells[max_loc].Erad * mass_scale_ / (length_scale_ * pow<2>(time_scale_) * CG::radiation_constant), 0.25)<<" max_Er "<<max_Er<<" rank "<<rank<<" density "<<cells[max_loc].density<<
		" width "<<tess.GetWidth(max_loc)<<" Tgas_old "<<old_Tm[max_loc]<<" loc="<<tess.GetMeshPoint(max_loc)<<std::endl;
        std::cout<<"kp="<<sigma_absorption_planck[max_loc]<<" kr="<<sigma_absorption_average[max_loc]<<std::endl;
        // for(size_t j = 0; j < ENERGY_GROUPS_NUM; ++j)
        //     std::cout<<"Eg["<<j<<"]="<<cells[max_loc].Eg[j]*cells[max_loc].density<<" old Eg["<<j<<"]="<<old_Eg[max_loc][j]<<" bg="<<
        //     planck_integral::planck_energy_density_group_integral(energy_groups_boundary[j], energy_groups_boundary[j+1], cells[max_loc].temperature) * pow<2>(time_scale_) * length_scale_ / mass_scale_<< 
        //     " bg_old="<<
        //     planck_integral::planck_energy_density_group_integral(energy_groups_boundary[j], energy_groups_boundary[j+1], old_Tm[max_loc]) * pow<2>(time_scale_) * length_scale_ / mass_scale_<< ", sigma[g]=" << sigma_absorption_group[max_loc][j] << ", cdt*sigma_g=" <<sigma_absorption_group[max_loc][j]*CG::speed_of_light*dt*time_scale_<< std::endl;
        // std::cout << "which one " << max_which << std::endl;
		// PrintDebugData(max_loc);
	}

    return std::min(dt * 0.15 / max_diff, dt*1.2);
}

bool MultigroupDiffusion::step(double const tolerance, 
                               int& total_iters, 
                               Tessellation3D const& tess, 
                               std::vector<ComputationalCell3D>& cells,
                               std::vector<Conserved3D>& extensives,
                               double const dt,
                               double const time) const {

    auto const N = tess.GetPointNo();

    extensives_temp = extensives;
    cells_temp = cells;
    
    cells_cgs = cells;
    for(std::size_t i=0; i<N; ++i){
        cells_cgs[i].density *= mass_scale_ / pow<3>(length_scale_);
        cells_cgs[i].internal_energy *= pow<2>(length_scale_) / pow<2>(time_scale_);
        cells_cgs[i].Erad *= pow<2>(length_scale_) / pow<2>(time_scale_);
        cells_cgs[i].velocity *= length_scale_ / time_scale_;
        for(std::size_t g=0; g<ENERGY_GROUPS_NUM; ++g){
            cells_cgs[i].Eg[g] *= pow<2>(length_scale_) / pow<2>(time_scale_);
        }
    }

    int rank = 0;
#ifdef RICH_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	ComputationalCell3D cdummy;
	MPI_exchange_data(tess, cells_cgs, true, &cdummy);	
#endif

    calculate_group_absorption_and_scattering_coefficients(tess, cells_cgs, dt * time_scale_);
    calculate_planck_integrals(tess, cells_cgs);
    calculate_gray_absorption_and_scattering_coefficients(tess, cells);
    calculate_fleck_factor(tess, cells, dt * time_scale_);

    std::size_t tot_iters = 0;
    new_Eg = CG::BiCGSTAB(tolerance, total_iters, tess, cells, dt, *this, time, new_Eg_full);
    tot_iters += total_iters;
    if(rank == 0)
        std::cout<<"Total iterations: "<<tot_iters<<std::endl;
    
    PostCGFull(tess, extensives, dt, cells, new_Eg, new_Eg_full);
    
    cells_cgs = cells;
    
    for(std::size_t i=0; i<N; ++i){
        cells_cgs[i].density *= mass_scale_ / pow<3>(length_scale_);
        cells_cgs[i].internal_energy *= pow<2>(length_scale_) / pow<2>(time_scale_);
        cells_cgs[i].Erad *= pow<2>(length_scale_) / pow<2>(time_scale_);
        cells_cgs[i].velocity *= length_scale_ / time_scale_;
        for(std::size_t g=0; g<ENERGY_GROUPS_NUM; ++g){
            cells_cgs[i].Eg[g] *= pow<2>(length_scale_) / pow<2>(time_scale_);
        }
    }
#ifdef RICH_MPI
    MPI_exchange_data(tess, cells, true, &cdummy);
    MPI_exchange_data(tess, cells_cgs, true, &cdummy);
#endif

//     std::size_t constexpr max_iter=1;
//     for(std::size_t iter=1; iter <= max_iter; ++iter){    
//         gray = false;
        
//         std::size_t tot_iters = 0;
//         for(std::size_t g=0; g<ENERGY_GROUPS_NUM; ++g){
//             current_group=g;
//             new_Eg = CG::BiCGSTAB(tolerance, total_iters, tess, cells, dt, *this, time, new_Eg_full);

//             tot_iters += total_iters;
//             PostCG(tess, extensives, dt, cells, new_Eg, new_Eg_full);
//         }

//         for(std::size_t i=0; i<N; ++i)
//         {
//             for(std::size_t g=0; g<ENERGY_GROUPS_NUM; ++g){
//                     cells_cgs[i].Eg[g] = cells[i].Eg[g] * pow<2>(length_scale_) / pow<2>(time_scale_);
//                 }
//         }
//  #ifdef RICH_MPI
//         ComputationalCell3D cdummy;
//         MPI_exchange_data(tess, cells_cgs, true, &cdummy);	
// #endif
      
//         calculate_gray_absorption_and_scattering_coefficients(tess, cells);

//         gray = true;
//         new_Er = CG::BiCGSTAB(tolerance, total_iters, tess, cells, dt, *this, time, new_Er_full);

//         tot_iters += total_iters;
//         if(rank == 0)
//             std::cout << "iter " << iter << " tot_iters " << tot_iters << std::endl;

//         PostCG(tess, extensives, dt, cells, new_Er, new_Er_full);

//         if(iter <= max_iter){
//             cells_cgs = cells;
//             for(std::size_t i=0; i<N; ++i){
//                 cells_cgs[i].density *= mass_scale_ / pow<3>(length_scale_);
//                 cells_cgs[i].internal_energy *= pow<2>(length_scale_) / pow<2>(time_scale_);
//                 cells_cgs[i].Erad *= pow<2>(length_scale_) / pow<2>(time_scale_);
//                 cells_cgs[i].velocity *= length_scale_ / time_scale_;
//                 for(std::size_t g=0; g<ENERGY_GROUPS_NUM; ++g){
//                     cells_cgs[i].Eg[g] *= pow<2>(length_scale_) / pow<2>(time_scale_);
//                 }
//             }

// #ifdef RICH_MPI
//             ComputationalCell3D cdummy;
//             MPI_exchange_data(tess, cells_cgs, true, &cdummy);	
// #endif
//         }
//     }

    // if(doppler_on_) solve_doppler_shift(tess, cells, dt, extensives);
    
    return true;
}



void MultigroupDiffusion::BuildMatrix(Tessellation3D const& tess, 
                                      mat& A, 
                                      size_t_mat& A_indeces, 
                                      std::vector<ComputationalCell3D> const& cells, 
                                      double const dt, 
                                      std::vector<double>& b, 
                                      std::vector<double>& x0, 
                                      double const current_time) const {
    // if(gray){
    //     BuildMatrixGray(tess, A, A_indeces, cells, dt, b, x0, current_time);
    // } else {
    //     assert(current_group < ENERGY_GROUPS_NUM);
    //     BuildMatrixGroup(current_group, tess, A, A_indeces, cells, dt, b, x0, current_time);
    // }
    BuildMatrixGroupFull(tess, A, A_indeces, cells, dt, b, x0, current_time);
}

void MultigroupDiffusion::BuildMatrixGroupFull(Tessellation3D const& tess, 
                                           mat& A, 
                                           size_t_mat& A_indeces, 
                                           std::vector<ComputationalCell3D> const& cells, 
                                           double const dt, 
                                           std::vector<double>& b, 
                                           std::vector<double>& x0, 
                                           double const current_time) const {
    int rank = 0;
#ifdef RICH_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif

    std::size_t const Nlocal = tess.GetPointNo();
    double const dt_cgs = dt * time_scale_;
    double const cdt = CG::speed_of_light*dt_cgs;

    x0.resize(Nlocal * ENERGY_GROUPS_NUM, 0.0);
    b.resize(Nlocal * ENERGY_GROUPS_NUM, 0.0);
    // build the `initial guess` and `b`
    for(std::size_t i=0; i < Nlocal; ++i){
        auto const& cell_cgs = cells_cgs[i];
        auto const volume_cgs = tess.GetVolume(i) * pow<3>(length_scale_);
        double const f = fleck_factor[i];
        
        auto const Um = get_radiation_energy_density(cell_cgs.temperature);
        double aT4_np1 = f * Um;
        for(size_t group=0; group<ENERGY_GROUPS_NUM; ++group)
            aT4_np1 += (1 - f) * sigma_absorption_group[i][group] * cell_cgs.Eg[group] * cell_cgs.density / sigma_absorption_planck[i];
        for(size_t group=0; group<ENERGY_GROUPS_NUM; ++group){
            double const Eg_i = cell_cgs.Eg[group] * cell_cgs.density;
            // build the initial guess
            auto const bg = planck_integal_group[i][group];
            double const Eg_guess = (Eg_i + bg * cdt * sigma_absorption_group[i][group] * aT4_np1)
                / (1 + cdt * sigma_absorption_group[i][group]);
            x0[i * ENERGY_GROUPS_NUM + group] = std::max(0.5 * Eg_i, std::min(Eg_guess, 2 * Eg_i));
            // build `b` vector, first term
            b[i * ENERGY_GROUPS_NUM + group] = volume_cgs * old_Eg[i][group] * mass_scale_ / (length_scale_*pow<2>(time_scale_));

            // second term
            auto const cdtkgbgf = f*cdt*sigma_absorption_group[i][group]*bg;
            b[i * ENERGY_GROUPS_NUM + group] += volume_cgs*cdtkgbgf*Um;
        }
    }
    
    // Initialize Matrix
    A.clear(); 
    A.resize(Nlocal * ENERGY_GROUPS_NUM);
    A_indeces.clear();
    A_indeces.resize(Nlocal * ENERGY_GROUPS_NUM);

    // Add the emission term to the matrix
    for(std::size_t i=0; i < Nlocal; ++i){
        if(compton_on_) {
            generate_S_and_dSdUm_matrices(cells[i], i);
            calculate_compton_quantities(cells[i], i);
        }
        
        double const f = fleck_factor[i];
        double const volume = tess.GetVolume(i) * pow<3>(length_scale_);
        for(size_t group=0; group<ENERGY_GROUPS_NUM; ++group){
            auto const bg = planck_integal_group[i][group];
            // double const sigma_planck_1 = 1.0 / sigma_absorption_planck[i];
            double const Gamma_1 = 1.0 / Gammas[i];

            double const cdtkg = cdt * sigma_absorption_group[i][group];
            double const implicit_self_contribution = -(1 - f) * cdtkg * Gamma_1 * bg * sigma_absorption_group[i][group];
            
            double implicit_self_compton_contribution = 0.0;

            if(compton_on_){
                double const implicit_compton_contribution_to_b = get_implicit_compton_contribution_to_b(tess, cells[i], i, group, dt_cgs);
                b[i * ENERGY_GROUPS_NUM + group] += implicit_compton_contribution_to_b;

                implicit_self_compton_contribution = get_implicit_compton_contribution(tess, cells[i], i, group, group, dt_cgs);
            }

            A[i * ENERGY_GROUPS_NUM + group].push_back(volume*(1.0 + cdtkg + implicit_self_contribution) + implicit_self_compton_contribution);
            A_indeces[i * ENERGY_GROUPS_NUM + group].push_back(i * ENERGY_GROUPS_NUM + group);

            for(size_t group_j=0; group_j<ENERGY_GROUPS_NUM; ++group_j){
                if(group_j!= group){
                    double const implicit_conribution_group_j = -volume*bg * (1 - f) * sigma_absorption_group[i][group_j] * sigma_absorption_group[i][group] * cdt * Gamma_1;
                    
                    double implicit_compton_contribution_group_j = 0.0;
                    if(compton_on_){
                        implicit_compton_contribution_group_j = get_implicit_compton_contribution(tess, cells[i], i, group, group_j, dt_cgs);
                    }
                    
                    A[i * ENERGY_GROUPS_NUM + group].push_back(implicit_conribution_group_j + implicit_compton_contribution_group_j);
                    
                    A_indeces[i * ENERGY_GROUPS_NUM + group].push_back(i * ENERGY_GROUPS_NUM + group_j);
                }
            }
            double A0 = A[i * ENERGY_GROUPS_NUM + group][0];
            if(A0 < 0){
                std::cout << "Negative A[i][i] in matrix build" << std::endl;
            }
        }
    }

    // find max_abs_grad_E for flux_limiter limiter gradient factor
    std::vector<std::size_t> neighbors;
    face_vec faces;
    Vector3D dummy_v;
    if(flux_limiter_){
        for(std::size_t i=0; i < Nlocal; ++i){
            double abs_grad_E_temp = 0.0;
            
            tess.GetNeighbors(i, neighbors);
            faces = tess.GetCellFaces(i);

            auto const Nneighbors = neighbors.size();
            double const Er_i = cells_cgs[i].Erad * cells_cgs[i].density;

            for(std::size_t j=0; j < Nneighbors; ++j){
                std::size_t const neighbor_j = neighbors[j];
                if(neighbor_j < Nlocal || !tess.IsPointOutsideBox(neighbor_j)){
                    double const Er_j = cells_cgs[neighbor_j].Erad * cells_cgs[neighbor_j].density;
                    auto const abs_dE = std::abs(Er_i - Er_j);
                    auto const abs_grad_E = abs_dE * fastabs(grad[faces[j]]);

                    abs_grad_E_temp = std::max(abs_grad_E_temp, abs_grad_E);
                }
            }
            max_abs_grad_E[i] = abs_grad_E_temp;
        }
#ifdef RICH_MPI
        MPI_exchange_data2(tess, max_abs_grad_E, true);
#endif 
    }

    // Add the diffusion terms
    for(std::size_t i=0; i < Nlocal; ++i){
        faces = tess.GetCellFaces(i);

        tess.GetNeighbors(i, neighbors);
        std::size_t const Nneighbors = neighbors.size();
        
        Vector3D const r_i = tess.GetMeshPoint(i);

        auto& cell_i = cells_cgs[i]; // reference and not const reference is because we change cell_i.temperature to calculate the diffusion coefficient 

        for(std::size_t j=0; j < Nneighbors; ++j){
            std::size_t const neighbor_j = neighbors[j];
            
            auto r_ij = r_i - tess.GetMeshPoint(neighbor_j);

            double const abs_r_ij = abs(r_ij);
            r_ij *= 1.0 / abs_r_ij; // normalize the vector perpendicular to the face between cells i and j
            
            double Eg_j = 0;
            bool const outside_point = tess.IsPointOutsideBox(neighbor_j);
            auto *cell_j = outside_point? nullptr : &cells_cgs[neighbor_j];
            for(size_t group=0; group<ENERGY_GROUPS_NUM; ++group){
                double const Eg_i = cell_i.Eg[group] * cell_i.density;

                if(!outside_point){
                    Eg_j = cell_j->Eg[group] * cell_j->density;

                    // since the diffusion terms are symmetric we only go update the matrix if i < j
                    if(i < neighbor_j){
                        auto const& face_j = faces[j];
                        Vector3D const& gradient = grad[face_j];

                        // calculate the diffusion coefficient on the boundary using the maximal temperature of the cells
                        double const T_i = cell_i.temperature;
                        double const T_j = cell_j->temperature;
                        double const max_T = std::max(old_Tm[i], old_Tm[j]);

                        cell_j->temperature = max_T;
                        cell_i.temperature = max_T;
                        
                        double const D_i = coefficient_calculator.CalcDiffusionCoefficientGroup(cell_i, group);
                        double const D_j = coefficient_calculator.CalcDiffusionCoefficientGroup(*cell_j, group);
                        
                        cell_i.temperature = T_i;
                        cell_j->temperature = T_j;

                        double const D_ij = 2.0 * D_i * D_j / (D_i + D_j);

                        double lambda = 1.0;
                        if(flux_limiter_){
                            double const dEg = Eg_i - Eg_j;

                            double const gradE_magnitude = std::max(std::abs(fastabs(gradient)*dEg), std::numeric_limits<double>::min()*1e40);

                            max_neighbor_abs_grad_E[i] = std::max(max_neighbor_abs_grad_E[i], max_abs_grad_E[neighbor_j]);

                            double const grad_factor = std::max(0.15 * (max_abs_grad_E[i] + max_abs_grad_E[neighbor_j])/gradE_magnitude, 1.0);

                            lambda = CG::CalcSingleFluxLimiter(gradient*dEg*grad_factor, D_ij, 0.5*(Eg_i + Eg_j));
                        }
                        double const lambdaD = lambda*D_ij;

                        double const A_j = tess.GetArea(face_j) * pow<2>(length_scale_);
                        double const flux  = dt_cgs * lambdaD * ScalarProd(gradient, r_ij) * A_j;

                        A[i * ENERGY_GROUPS_NUM + group][0] += flux;
                        A[i * ENERGY_GROUPS_NUM + group].push_back(-flux);
                        A_indeces[i * ENERGY_GROUPS_NUM + group].push_back(neighbor_j * ENERGY_GROUPS_NUM + group);

                        if(neighbor_j < Nlocal){ // check that neighboring cell is not boundary
                            A[neighbor_j * ENERGY_GROUPS_NUM + group][0] += flux;
                            A[neighbor_j * ENERGY_GROUPS_NUM + group].push_back(-flux);
                            A_indeces[neighbor_j * ENERGY_GROUPS_NUM + group].push_back(i * ENERGY_GROUPS_NUM + group);
                        }
                    }
                } else { // boundary condition
                    if(i < neighbor_j){
                        boundary_calculator.setBoundaryValuesGroup(group, tess, i, neighbor_j, dt_cgs, cells_cgs, tess.GetArea(faces[j])*pow<2>(length_scale_), A[i * ENERGY_GROUPS_NUM + group][0], b[i * ENERGY_GROUPS_NUM + group], faces[j]);
                    }
                }
            }
        }
    }

    // double E0 = cells_cgs[0].Eg[0] * cells_cgs[0].density;
    // double A00 = A[0][0];
    // double A01 = A[0][1];
    // double b0 = b[0];
    // double A111 = A[0][ENERGY_GROUPS_NUM];
    // double A211 = A[0][ENERGY_GROUPS_NUM+1];

    // double A10 = A[ENERGY_GROUPS_NUM][0];
    // double A11 = A[ENERGY_GROUPS_NUM][1];
    // double A21 = A[ENERGY_GROUPS_NUM][2];
    // A111 = A[ENERGY_GROUPS_NUM][ENERGY_GROUPS_NUM];
    // A211 = A[ENERGY_GROUPS_NUM][ENERGY_GROUPS_NUM+1];
    // double b1 = b[ENERGY_GROUPS_NUM];

    // Add velocity term
   for(std::size_t i=0; i<Nlocal; ++i){
        double const volume = tess.GetVolume(i) * pow<3>(length_scale_);
        
        faces = tess.GetCellFaces(i);
        tess.GetNeighbors(i, neighbors);
        std::size_t const Nneighbors = neighbors.size();
        double div_V = 0;
        Vector3D const r_i = tess.GetMeshPoint(i);
        for(std::size_t j=0; j<Nneighbors; ++j){
            std::size_t const neighbor_j = neighbors[j];

            auto const r_ij = normalize(r_i - tess.GetMeshPoint(neighbor_j));

            double const A_ij = tess.GetArea(faces[j]) * pow<2>(length_scale_);
            Vector3D velocity_j;
            bool const is_outside = tess.IsPointOutsideBox(neighbor_j);
            if(!is_outside){
                velocity_j = cells_cgs[neighbor_j].velocity;
            } else {
                double dummyEg_i, dummy_Eg_j;
                boundary_calculator.getOutsideValuesGroup(0, tess, i, neighbor_j, cells_cgs, dummyEg_i, dummy_Eg_j, velocity_j);
            }// TODO add R2 part
            div_V -= 0.5*ScalarProd(cells_cgs[i].velocity+velocity_j, r_ij) * A_ij;
            for(size_t group=0; group<ENERGY_GROUPS_NUM; ++group){
                A[i * ENERGY_GROUPS_NUM + group][0] += -0.5*ScalarProd(cells_cgs[i].velocity+velocity_j, r_ij) * A_ij * dt_cgs / 3.0;
            }
        }
 
        if(doppler_on_){
            for(std::size_t g=0; g<ENERGY_GROUPS_NUM; ++g){
                std::size_t gp = g+1 == ENERGY_GROUPS_NUM ? g : g+1;
                std::size_t gm = g == 0                   ? g : g-1;

                double const dnu_gp = energy_groups_width[gp];
                double const dnu_g = energy_groups_width[g];
                double const dnu_gm = energy_groups_width[gm];

                double nu_gp = energy_groups_boundary[gp];
                double nu_gm = energy_groups_boundary[g];
                if(g ==0)
                {
                    if(div_V > 0){
                        nu_gp = 0;
                    } else {
                        nu_gm = 0;
                    }
                }
                if(g == (ENERGY_GROUPS_NUM - 1))
                {
                    if(div_V < 0){
                        nu_gm = 0;
                    } else {
                        nu_gp = 0;
                    }
                }

                double const coeff = div_V * dt_cgs;
                if(div_V < 0.0){
                    A[i * ENERGY_GROUPS_NUM + g][0] -= coeff * nu_gp / dnu_g;

                    auto it = std::find(A_indeces[i * ENERGY_GROUPS_NUM + g].begin(), A_indeces[i * ENERGY_GROUPS_NUM + g].end(), i * ENERGY_GROUPS_NUM + gm);

                    if(it == A_indeces[i * ENERGY_GROUPS_NUM + g].end()){
                        throw UniversalError("Not found [i*ENERGY_GROUPS_NUM + gm] in A_indeces (1)");
                    } 

                    std::size_t const gm_counter = static_cast<std::size_t>(it - A_indeces[i * ENERGY_GROUPS_NUM + g].begin());

                    if(A_indeces[i * ENERGY_GROUPS_NUM + g][gm_counter] != i*ENERGY_GROUPS_NUM + gm) {
                        throw UniversalError("Not found [i*ENERGY_GROUPS_NUM + gm] in A_indeces (2)");
                    }

                    A[i * ENERGY_GROUPS_NUM + g][gm_counter] += coeff * nu_gm / dnu_gm;
                } else {
                    A[i * ENERGY_GROUPS_NUM + g][0] += coeff * nu_gm / dnu_g;

                    auto it = std::find(A_indeces[i * ENERGY_GROUPS_NUM + g].begin(), A_indeces[i*ENERGY_GROUPS_NUM + g].end(), i * ENERGY_GROUPS_NUM + gp);

                    if(it == A_indeces[i * ENERGY_GROUPS_NUM + g].end()){
                        throw UniversalError("Not found [i*ENERGY_GROUPS_NUM + gp] in A_indeces (3)");
                    } 

                    std::size_t const gp_counter = static_cast<std::size_t>(it - A_indeces[i * ENERGY_GROUPS_NUM + g].begin());
                    
                    if(A_indeces[i * ENERGY_GROUPS_NUM + g][gp_counter] != i*ENERGY_GROUPS_NUM + gp) {
                        throw UniversalError("Not found [i*ENERGY_GROUPS_NUM + gp] in A_indeces (4)");
                    }

                    A[i * ENERGY_GROUPS_NUM + g][gp_counter] -= coeff * nu_gp / dnu_gp;
                }

            }
        }    
    }

    // Find maximum number of neighbors and allocate data
    // THIS SHOULD BE IN PRESTEP BUT BiCGSTAB CREATES A NEW MATRIX EVERY TIME IT IS CALLED. 
    // MAYBE MATRIX BUILDER SHOULD HOLD A MATRIX AS AN ATTRIBUTE
    
    std::size_t max_neighbors = 0;
    for(std::size_t i=0; i < Nlocal; ++i){
        for(size_t group=0; group<ENERGY_GROUPS_NUM; ++group)
            max_neighbors = std::max(max_neighbors, A[i * ENERGY_GROUPS_NUM + group].size());
    }
    ++max_neighbors;
    
    for(std::size_t i=0; i < Nlocal; ++i){
        for(size_t group=0; group<ENERGY_GROUPS_NUM; ++group){
            A[i * ENERGY_GROUPS_NUM + group].resize(max_neighbors, 0);
            A_indeces[i * ENERGY_GROUPS_NUM + group].resize(max_neighbors, max_size_t);

            if(A[i * ENERGY_GROUPS_NUM + group][0] < 0){
                std::cout << "Negative A in matrix build" << std::endl;
            }
        }
    }
}


void MultigroupDiffusion::BuildMatrixGroup(std::size_t group,
                                           Tessellation3D const& tess, 
                                           mat& A, 
                                           size_t_mat& A_indeces, 
                                           std::vector<ComputationalCell3D> const& cells, 
                                           double const dt, 
                                           std::vector<double>& b, 
                                           std::vector<double>& x0, 
                                           double const current_time) const {
    int rank = 0;
#ifdef RICH_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif

    std::size_t const Nlocal = tess.GetPointNo();
    double const dt_cgs = dt * time_scale_;
    double const cdt = CG::speed_of_light*dt_cgs;

    x0.resize(Nlocal, 0.0);
    b.resize(Nlocal, 0.0);
    // build the `initial guess` and `b`
    for(std::size_t i=0; i<Nlocal; ++i){
        auto const& cell_cgs = cells_cgs[i];
        
        double const Eg_i = cell_cgs.Eg[group] * cell_cgs.density;
        // build the initial guess
        x0[i] = Eg_i;

        auto const volume_cgs = tess.GetVolume(i) * pow<3>(length_scale_);

        // build `b` vector, first term
        b[i] = volume_cgs * old_Eg[group][i] * mass_scale_ / (length_scale_*pow<2>(time_scale_));

        // second term
        auto const bg = planck_integal_group[group][i];
        auto const Um = get_radiation_energy_density(cell_cgs.temperature);
        auto const cdtkgbg = std::min(cdt*sigma_absorption_group[group][i], max_coupling_strength)*bg;
        b[i] += volume_cgs*cdtkgbg*Um;
    }
    
    // Initialize Matrix
    A.clear(); 
    A.resize(Nlocal);
    A_indeces.clear();
    A_indeces.resize(Nlocal);

    // Add the emission term to the matrix
    for(std::size_t i=0; i < Nlocal; ++i){
        A_indeces[i].push_back(i);

        double const volume = tess.GetVolume(i) * pow<3>(length_scale_);

        double cdtkg = std::min(cdt * sigma_absorption_group[group][i], max_coupling_strength);

        A[i].push_back(volume*(1.0 + cdtkg));

        if(A[i][0] < 0){
            std::cout << "Negative A[i][i] in matrix build" << std::endl;
        }
    }

    // find max_abs_grad_E for flux_limiter limiter gradient factor
    std::vector<std::size_t> neighbors;
    face_vec faces;
    Vector3D dummy_v;
    if(flux_limiter_ && group == 0){
        // this depends on the order and the assumption is that it is done before the gray step
        // TODO: MOVE TO PRESTEP
        for(std::size_t i=0; i < Nlocal; ++i){
            double abs_grad_E_temp = 0.0;
            
            tess.GetNeighbors(i, neighbors);
            faces = tess.GetCellFaces(i);

            auto const Nneighbors = neighbors.size();
            // won't work with iterations
            double const Er_i = cells_cgs[i].Erad * cells_cgs[i].density;

            for(std::size_t j=0; j < Nneighbors; ++j){
                std::size_t const neighbor_j = neighbors[j];
                if(neighbor_j < Nlocal || !tess.IsPointOutsideBox(neighbor_j)){
                    double const Er_j = cells_cgs[neighbor_j].Erad * cells_cgs[neighbor_j].density;
                    auto const abs_dE = std::abs(Er_i - Er_j);
                    auto const abs_grad_E = abs_dE * fastabs(grad[faces[j]]);

                    abs_grad_E_temp = std::max(abs_grad_E_temp, abs_grad_E);
                }
            }

            max_abs_grad_E[i] = abs_grad_E_temp;
        }

#ifdef RICH_MPI
        MPI_exchange_data2(tess, max_abs_grad_E, true);
#endif 
    }

    // Add the diffusion terms
    for(std::size_t i=0; i < Nlocal; ++i){
        faces = tess.GetCellFaces(i);

        tess.GetNeighbors(i, neighbors);
        std::size_t const Nneighbors = neighbors.size();
        
        Vector3D const r_i = tess.GetMeshPoint(i);

        auto& cell_i = cells_cgs[i]; // reference and not const reference is because we change cell_i.temperature to calculate the diffusion coefficient 
        double const Eg_i = cell_i.Eg[group] * cell_i.density;

        for(std::size_t j=0; j < Nneighbors; ++j){
            std::size_t const neighbor_j = neighbors[j];
            
            auto r_ij = r_i - tess.GetMeshPoint(neighbor_j);

            double const abs_r_ij = abs(r_ij);
            r_ij *= 1.0 / abs_r_ij; // normalize the vector perpendicular to the face between cells i and j
            
            double Eg_j = 0;

            if(!tess.IsPointOutsideBox(neighbor_j)){
                auto& cell_j = cells_cgs[neighbor_j]; // reference and not const reference is because we change cell_i.temperature to calculate the diffusion coefficient

                Eg_j = cell_j.Eg[group] * cell_j.density;

                // since the diffusion terms are symmetric we only go update the matrix if i < j
                if(i < neighbor_j){
                    auto const& face_j = faces[j];
                    Vector3D const& gradient = grad[face_j];

                    // calculate the diffusion coefficient on the boundary using the maximal temperature of the cells
                    double const T_i = cell_i.temperature;
                    double const T_j = cell_j.temperature;
                    double const max_T = std::max(old_Tm[i], old_Tm[j]);

                    cell_j.temperature = max_T;
                    cell_i.temperature = max_T;
                    
                    double const D_i = coefficient_calculator.CalcDiffusionCoefficientGroup(cell_i, group);
                    double const D_j = coefficient_calculator.CalcDiffusionCoefficientGroup(cell_j, group);
                    
                    cell_i.temperature = T_i;
                    cell_j.temperature = T_j;

                    double const D_ij = 2.0 * D_i * D_j / (D_i + D_j);


                    double lambda = 1.0;
                    if(flux_limiter_){
                        double const dEg = Eg_i - Eg_j;

                        double const gradE_magnitude = std::max(std::abs(fastabs(gradient)*dEg), std::numeric_limits<double>::min()*1e40);

                        max_neighbor_abs_grad_E[i] = std::max(max_neighbor_abs_grad_E[i], max_abs_grad_E[neighbor_j]);

                        double const grad_factor = std::max(0.15 * (max_abs_grad_E[i] + max_abs_grad_E[neighbor_j])/gradE_magnitude, 1.0);

                        lambda = CG::CalcSingleFluxLimiter(gradient*dEg*grad_factor, D_ij, 0.5*(Eg_i + Eg_j));
                    }

                    double const lambdaD = lambda*D_ij;

                    double const A_j = tess.GetArea(face_j) * pow<2>(length_scale_);
                    double const flux  = dt_cgs * lambdaD * ScalarProd(gradient, r_ij) * A_j;

                    A[i][0] += flux;
                    A[i].push_back(-flux);
                    A_indeces[i].push_back(neighbor_j);

                    if(neighbor_j < Nlocal){ // check that neighboring cell is not boundary
                        A[neighbor_j][0] += flux;
                        A[neighbor_j].push_back(-flux);
                        A_indeces[neighbor_j].push_back(i);
                    }
                }

            } else { // boundary condition
                if(i < neighbor_j){
                    boundary_calculator.setBoundaryValuesGroup(group, tess, i, neighbor_j, dt_cgs, cells_cgs, tess.GetArea(faces[j])*pow<2>(length_scale_), A[i][0], b[i], faces[j]);
                }
            }
        }
    }

    if(mix_frame_on_){
        // momentum relativity _term  
        Vector3D grad_Eg(0.0, 0.0, 0.0);
        for(std::size_t i=0; i<Nlocal; ++i){
            double const volume = tess.GetVolume(i) * pow<3>(length_scale_);
            
            faces = tess.GetCellFaces(i);
            tess.GetNeighbors(i, neighbors);
            std::size_t const Nneighbors = neighbors.size();

            Vector3D const r_i = tess.GetMeshPoint(i);
            double const Eg_i = cells_cgs[i].Eg[group] * cells_cgs[i].density;

            grad_Eg.Set(0.0, 0.0, 0.0); 
            for(std::size_t j=0; j<Nneighbors; ++j){
                std::size_t const neighbor_j = neighbors[j];
                double Eg_j;
                if(!tess.IsPointOutsideBox(neighbor_j)){
                    Eg_j = cells_cgs[neighbor_j].Eg[group] * cells_cgs[neighbor_j].density;
                } else {
                    boundary_calculator.getOutsideValuesGroup(group, tess, i, neighbor_j, cells_cgs, Eg_i, Eg_j, dummy_v);
                }

                auto r_ij = r_i - tess.GetMeshPoint(neighbor_j);
                r_ij *= 1.0 / abs(r_ij);
                
                grad_Eg += r_ij * (0.5*tess.GetArea(faces[j])*(Eg_i + Eg_j));
            }
            grad_Eg *= -(1.0 / volume) * pow<2>(length_scale_);

            double const grad_magnitude = fastabs(grad_Eg);
            if(grad_magnitude < 0.5 * max_neighbor_abs_grad_E[i] && grad_magnitude > std::numeric_limits<double>::min() * 1e40){
                grad_Eg *= 1.0 / grad_magnitude; // normalize the gradient
                grad_Eg *= 0.5 * max_neighbor_abs_grad_E[i]; // bound it from below
            }

            double const Dg_cell = coefficient_calculator.CalcDiffusionCoefficientGroup(cells_cgs[i], group);
            double const lambda_g = flux_limiter_ ? CG::CalcSingleFluxLimiter(grad_Eg, Dg_cell, Eg_i) : 1.0;

            double const v_limiter = std::min(1.0, 0.1 * CG::speed_of_light / (fastabs(cells_cgs[i].velocity ) + 1e-2)); // limit velocity to 0.1*speed_of_light

            for(std::size_t j=0; j<Nneighbors; ++j){
                std::size_t const neighbor_j = neighbors[j];

                Vector3D r_ij = r_i - tess.GetMeshPoint(neighbor_j);
                r_ij *= 1.0 / abs(r_ij);

                Vector3D const& gradient = grad[faces[j]];
                Vector3D const proj_grad = r_ij * ScalarProd(gradient, r_ij); // projection of the gradient to the face normal

                double const A_ij = tess.GetArea(faces[j]) * pow<2>(length_scale_);
                double const sigma_rossland = CG::speed_of_light / (Dg_cell * 3.0);
                double const sigma_abs = sigma_absorption_group[group][i];
                
                double const nu_g = energy_groups_center[group];
                
                double sigma_p;
                double sigma_m;
                double dnu;
                
                if(group == 0){
                    sigma_p = sigma_absorption_group[group+1][i];
                    sigma_m = sigma_absorption_group[group][i];
                    dnu = energy_groups_center[group+1]-energy_groups_center[group];
                } else if(group+1 == ENERGY_GROUPS_NUM){
                    sigma_p = sigma_absorption_group[group][i];
                    sigma_m = sigma_absorption_group[group-1][i];
                    dnu = energy_groups_center[group]-energy_groups_center[group-1];
                } else {
                    sigma_p = sigma_absorption_group[group+1][i];
                    sigma_m = sigma_absorption_group[group-1][i];
                    dnu = energy_groups_center[group+1]-energy_groups_center[group-1];
                }
                
                double const dsigma_abs_dnu = doppler_on_ ? (sigma_p - sigma_m) / dnu : 0.0; 

                
                double const momentum_term_coefficient = -0.5 * (lambda_g / 3.0) * A_ij * (v_limiter * 2.0 * sigma_abs / sigma_rossland - 1.0 + nu_g*dsigma_abs_dnu) * ScalarProd(cells_cgs[i].velocity, r_ij);
                
                double const momentum_relativity_term = dt_cgs * momentum_term_coefficient;
                if(!tess.IsPointOutsideBox(neighbor_j)){
                    
                    A[i][0] += momentum_relativity_term;
                    
                    // TODO:: this seems slow.. we can probably speed this up using a simple counter 
                    auto it = std::find(A_indeces[i].begin(), A_indeces[i].end(), neighbor_j);

                    if(it == A_indeces[i].end()){
                        throw UniversalError("Key not equal in multigroup diffusion");
                    }

                    std::size_t const neighbor_counter = static_cast<std::size_t>(it - A_indeces[i].begin());
                    if(A_indeces[i][neighbor_counter] != neighbor_j){
                        throw UniversalError("Key not equal in multigroup diffusion");
                    }

                    A[i][neighbor_counter] += momentum_relativity_term;
                } else {
                    boundary_calculator.setMomentumTermBoundaryGroup(group, tess, i, neighbor_j, dt_cgs, cells_cgs, momentum_relativity_term, A[i][0], b[i]);
                }
            }
        }
    }  else {
        for(std::size_t i=0; i<Nlocal; ++i){
            double const volume = tess.GetVolume(i) * pow<3>(length_scale_);
            
            faces = tess.GetCellFaces(i);
            tess.GetNeighbors(i, neighbors);
            std::size_t const Nneighbors = neighbors.size();

            Vector3D const r_i = tess.GetMeshPoint(i);

            for(std::size_t j=0; j<Nneighbors; ++j){
                std::size_t const neighbor_j = neighbors[j];

                auto const r_ij = normalize(r_i - tess.GetMeshPoint(neighbor_j));

                double const A_ij = tess.GetArea(faces[j]) * pow<2>(length_scale_);
                Vector3D velocity_j;

                if(!tess.IsPointOutsideBox(neighbor_j)){
                    velocity_j = cells_cgs[neighbor_j].velocity;
                } else {
                    double dummyEg_i, dummy_Eg_j;
                    boundary_calculator.getOutsideValuesGroup(group, tess, i, neighbor_j, cells_cgs, dummyEg_i, dummy_Eg_j, velocity_j);
                }
                // TODO add the R2_g part 
                A[i][0] += -0.5*ScalarProd(cells_cgs[i].velocity+velocity_j, r_ij) * A_ij * dt_cgs / 3.0;
            }
        }
    }
    // Find maximum number of neighbors and allocate data
    // THIS SHOULD BE IN PRESTEP BUT BiCGSTAB CREATES A NEW MATRIX EVERY TIME IT IS CALLED. 
    // MAYBE MATRIX BUILDER SHOULD HOLD A MATRIX AS AN ATTRIBUTE
    
    std::size_t max_neighbors = 0;
    for(std::size_t i=0; i < Nlocal; ++i){
        max_neighbors = std::max(max_neighbors, tess.GetNeighbors(i).size());
    }
    ++max_neighbors;
    
    for(std::size_t i=0; i < Nlocal; ++i){
        A[i].resize(max_neighbors, 0);
        A_indeces[i].resize(max_neighbors, max_size_t);

        if(A[i][0] < 0){
            std::cout << "Negative A in matrix build" << std::endl;
        }
    }
}

void MultigroupDiffusion::BuildMatrixGray(Tessellation3D const& tess, 
                                          mat& A, 
                                          size_t_mat& A_indeces, 
                                          std::vector<ComputationalCell3D> const& cells, 
                                          double const dt, 
                                          std::vector<double>& b, 
                                          std::vector<double>& x0, 
                                          double const current_time) const {
    int rank = 0;
#ifdef RICH_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif

    std::size_t const Nlocal = tess.GetPointNo();
    double const dt_cgs = dt * time_scale_;
    double const cdt = CG::speed_of_light*dt_cgs;

    x0.resize(Nlocal, 0.0);
    b.resize(Nlocal, 0.0);

    for(std::size_t i=0; i<Nlocal; ++i){
        auto const cell_cgs = cells_cgs[i];
        
        double const Er_i = cell_cgs.Erad * cell_cgs.density; 
        // build the initial guess
        // TODO:INITIAL GUESS SHOULD BE THE SUM OF THE NEW GROUP ENERGIES?
        x0[i] = Er_i;

        auto const volume_cgs = tess.GetVolume(i) * pow<3>(length_scale_);

        // build `b` vector, first term
        b[i] = volume_cgs * old_Er[i]* mass_scale_ / (length_scale_*pow<2>(time_scale_));

        double const sigma_planck = sigma_absorption_planck[i];
        double const T = old_Tm[i];
        double const f = fleck_factor[i];
        
        // second term 
        double const Um = get_radiation_energy_density(T);
        double const cdtkpf = cdt*sigma_planck*f;
        b[i] += volume_cgs * cdtkpf * Um;
    }

    // Initialize Matrx
    A.clear();
    A.resize(Nlocal);
    A_indeces.clear();
    A_indeces.resize(Nlocal);
    
    // Add the emission term to the matrix
    for(std::size_t i=0; i < Nlocal; ++i){
        A_indeces[i].push_back(i);

        double const volume = tess.GetVolume(i) * pow<3>(length_scale_);

        double cdtkrf = cdt*sigma_absorption_average[i]*fleck_factor[i];

        A[i].push_back(volume*(1.0 + cdtkrf));

        if(A[i][0] < 0){
            std::cout << "Negative A[i][i] in matrix build" << std::endl;
        }
    }

    std::vector<std::size_t> neighbors;
    face_vec faces;
    Vector3D dummy_v;

    // Add the diffusion terms
    for(std::size_t i=0; i < Nlocal; ++i){
        faces = tess.GetCellFaces(i);

        tess.GetNeighbors(i, neighbors);
        std::size_t const Nneighbors = neighbors.size();

        Vector3D const r_i = tess.GetMeshPoint(i);
        
        auto& cell_cgs_i = cells_cgs[i]; // reference and not const reference is because we change cell_i.temperature to calculate the diffusion coefficient 
        double const Er_i = cell_cgs_i.Erad * cell_cgs_i.density;

        for(std::size_t j=0; j < Nneighbors; ++j){
            std::size_t const neighbor_j = neighbors[j];

            auto r_ij = r_i - tess.GetMeshPoint(neighbor_j);
            
            double const abs_r_ij = abs(r_ij);
            r_ij *= 1.0 / abs_r_ij; // normalize the vector perpendicular to the face between cells i and j

            double Er_j = 0.0;

            if(!tess.IsPointOutsideBox(neighbor_j)){
                auto& cell_cgs_j = cells_cgs[neighbor_j]; // reference and not const reference is because we change cell_i.temperature to calculate the diffusion coefficient
                Er_j = cell_cgs_j.Erad * cell_cgs_j.density;

                if(i < neighbor_j){
                    auto const& face_j = faces[j];
                    Vector3D const& gradient = grad[face_j];

                    double const T_i = cell_cgs_i.temperature;
                    double const T_j = cell_cgs_j.temperature;
                    double const max_T = std::max(old_Tm[i], old_Tm[j]);

                    cell_cgs_i.temperature = max_T;
                    cell_cgs_j.temperature = max_T;

                    double lambdaD_i_to_j = 0.0;
                    double lambdaD_j_to_i = 0.0;
                    double sum_U_i = 0.0;
                    double sum_U_j = 0.0;
                    for(std::size_t g=0; g < ENERGY_GROUPS_NUM; ++g){
                        double const Dg_i = coefficient_calculator.CalcDiffusionCoefficientGroup(cell_cgs_i, g);
                        double const Dg_j = coefficient_calculator.CalcDiffusionCoefficientGroup(cell_cgs_j, g);

                        double const Dg_ij = 2.0 * Dg_i * Dg_j / (Dg_i + Dg_j);

                        double lambda_g = 1.0;
                        if(flux_limiter_){
                            // TODO: add grad_factor to both here and to gray
                            // Using old Eg for consistency with the group step
                            double const Eg_old_i = cell_cgs_i.Eg[g]*cell_cgs_i.density;
                            double const Eg_old_j = cell_cgs_j.Eg[g]*cell_cgs_j.density;
                            double const dEg = Eg_old_i - Eg_old_j;

                            double const gradE_magnitude = std::max(std::abs(fastabs(gradient)*dEg), std::numeric_limits<double>::min()*1e40);

                            double grad_factor = std::max(0.15 * (max_abs_grad_E[i] + max_abs_grad_E[neighbor_j])/gradE_magnitude, 1.0);                            

                            lambda_g = CG::CalcSingleFluxLimiter(gradient*dEg*grad_factor, Dg_ij, 0.5*(Eg_old_i + Eg_old_j));
                        }

                        double const lambda_gD = lambda_g * Dg_ij;
                        
                        // cell_cgs holds the old Eg but after the group step we need to use cells.
                        double const Eg_i = cells[i].Eg[g] * cells[i].density * mass_scale_ /(length_scale_ * pow<2>(time_scale_));
                        double const Eg_j = cells[neighbor_j].Eg[g] * cells[neighbor_j].density * mass_scale_ /(length_scale_ * pow<2>(time_scale_));

                        lambdaD_i_to_j += lambda_gD * Eg_i;
                        sum_U_i += Eg_i;

                        lambdaD_j_to_i += lambda_gD * Eg_j;
                        sum_U_j += Eg_j;
                    }

                    lambdaD_i_to_j /= sum_U_i;
                    lambdaD_j_to_i /= sum_U_j;

                    cell_cgs_i.temperature = T_i;
                    cell_cgs_j.temperature = T_j;

                    double const A_j = tess.GetArea(face_j) * pow<2>(length_scale_);
                    double const flux_factor = dt_cgs * ScalarProd(gradient, r_ij) * A_j;
                    
                    double const flux_i_to_j = flux_factor * lambdaD_i_to_j;

                    A[i][0] += flux_i_to_j;
                    
                    double const flux_j_to_i = flux_factor * lambdaD_j_to_i;
                    A[i].push_back(-flux_j_to_i);
                    A_indeces[i].push_back(neighbor_j);

                    if(neighbor_j < Nlocal){
                        A[neighbor_j][0] += flux_j_to_i; 
                        A[neighbor_j].push_back(-flux_i_to_j);
                        A_indeces[neighbor_j].push_back(i);
                    }
                }
            } else { // boundary conditions
                if(i < neighbor_j){
                    boundary_calculator.setBoundaryValuesGray(tess, i, neighbor_j, dt_cgs, cells_cgs, tess.GetArea(faces[j]) * pow<2>(length_scale_), A[i][0], b[i], faces[j]);
                }
            }
        }
    }

    // add the momentum terms
    Vector3D grad_Eg(0.0, 0.0, 0.0);
    std::vector<double> relativity_term_neighbors;
    std::vector<double> momentum_term_neighbors;
    std::vector<double> lambda_neighbors;
    std::vector<double> sigma_ratio_lambda_neighbors;

    for(std::size_t i=0; i < Nlocal; ++i){
        double const f = fleck_factor[i];
        double const volume = tess.GetVolume(i) * pow<3>(length_scale_);

        faces = tess.GetCellFaces(i);
        tess.GetNeighbors(i, neighbors);
        std::size_t const Nneighbors = neighbors.size();

        
        relativity_term_neighbors.resize(Nneighbors, 0.0);
        std::fill(relativity_term_neighbors.begin(), relativity_term_neighbors.end(), 0.0);
        
        momentum_term_neighbors.resize(Nneighbors, 0.0);
        std::fill(momentum_term_neighbors.begin(), momentum_term_neighbors.end(), 0.0);

        lambda_neighbors.resize(Nneighbors, 0.0);
        std::fill(lambda_neighbors.begin(), lambda_neighbors.end(), 0.0);

        sigma_ratio_lambda_neighbors.resize(Nneighbors, 0.0);
        std::fill(sigma_ratio_lambda_neighbors.begin(), sigma_ratio_lambda_neighbors.end(), 0.0);

        Vector3D const r_i = tess.GetMeshPoint(i);

        double lambda_i = 0.0;
        double sigma_ratio_lambda_i = 0.0;

        double relativity_term_i = 0.0;
        double momentum_term_i = 0.0;

        for(std::size_t g=0; g < ENERGY_GROUPS_NUM; ++g){
            grad_Eg.Set(0.0, 0.0, 0.0);

            double const Eg_i = cells_cgs[i].Eg[g] * cells_cgs[i].density;

            for(std::size_t j=0; j < Nneighbors; ++j){
                std::size_t const neighbor_j = neighbors[j];
                
                double Eg_j;
                if(!tess.IsPointOutsideBox(neighbor_j)){
                    Eg_j = cells_cgs[neighbor_j].Eg[g] * cells_cgs[neighbor_j].density;
                } else {
                    boundary_calculator.getOutsideValuesGroup(g, tess, i, neighbor_j, cells_cgs, Eg_i, Eg_j, dummy_v);
                }
                
                auto r_ij = r_i - tess.GetMeshPoint(neighbor_j);
                r_ij *= 1.0 / abs(r_ij);

                grad_Eg += r_ij * (0.5*tess.GetArea(faces[j])*(Eg_i + Eg_j));
            }

            grad_Eg *= -(1.0/volume) * pow<2>(length_scale_);
            
            double const grad_magnitude = fastabs(grad_Eg);

            if(grad_magnitude < 0.5 * max_neighbor_abs_grad_E[i] && grad_magnitude > std::numeric_limits<double>::min() * 1e40){
                grad_Eg *= 1.0 / grad_magnitude; // normalize the gradient
                grad_Eg *= 0.5 * max_neighbor_abs_grad_E[i]; // bound it from below
            }

            double const Dg_cell = coefficient_calculator.CalcDiffusionCoefficientGroup(cells_cgs[i], g);

            double const lambda_g = flux_limiter_ ? CG::CalcSingleFluxLimiter(grad_Eg, Dg_cell, Eg_i) : 1.0;
            double const sigma_rossland_g = CG::speed_of_light / (3.0 * Dg_cell);
            double const sigma_abs_g = sigma_absorption_group[g][i];

            lambda_i += lambda_g * cells[i].Eg[g] * cells[i].density * mass_scale_ / (length_scale_ * pow<2>(time_scale_));

            sigma_ratio_lambda_i += (sigma_abs_g * lambda_g / sigma_rossland_g) * cells[i].Eg[g] * cells[i].density * mass_scale_ / (length_scale_ * pow<2>(time_scale_));
            

            for(std::size_t j=0; j < Nneighbors; ++j){
                std::size_t const neighbor_j = neighbors[j];

                if(!tess.IsPointOutsideBox(neighbor_j)){
                    lambda_neighbors[j] += lambda_g * cells[neighbor_j].Eg[g] * cells[neighbor_j].density * mass_scale_ / (length_scale_ * pow<2>(time_scale_));

                    sigma_ratio_lambda_neighbors[j] += (sigma_abs_g * lambda_g / sigma_rossland_g) * cells[neighbor_j].Eg[g] * cells[neighbor_j].density * mass_scale_ / (length_scale_ * pow<2>(time_scale_));
                } else {
                    double Eg_outside;
                    double const Eg_i_current = cells[i].Eg[g] * cells[i].density * mass_scale_ / (length_scale_ * pow<2>(time_scale_));                    
                    boundary_calculator.getOutsideValuesGroup(g, tess, i, neighbor_j, cells_cgs, Eg_i_current, Eg_outside, dummy_v);

                    lambda_neighbors[j] += lambda_g * Eg_outside;
                    sigma_ratio_lambda_neighbors[j] += (sigma_abs_g * lambda_g / sigma_rossland_g) * Eg_outside;
                }
            }
        }

        auto const sum_Eg_i = std::accumulate(cells[i].Eg.begin(), cells[i].Eg.end(), 0.) * cells[i].density * mass_scale_ / (length_scale_ * pow<2>(time_scale_));

        lambda_i /= sum_Eg_i;
        lambda_cell_gray[i] = lambda_i;

        sigma_ratio_lambda_i /= sum_Eg_i;
        sigma_ratio_lambda_cell_gray[i] = sigma_ratio_lambda_i;
        
        for(std::size_t j=0; j < Nneighbors; ++j){
            std::size_t const neighbor_j = neighbors[j];

            if(!tess.IsPointOutsideBox(neighbor_j)){
                auto const sum_Eg_j = std::accumulate(cells[neighbor_j].Eg.begin(), cells[neighbor_j].Eg.end(), 0.) * cells[neighbor_j].density * mass_scale_ / (length_scale_ * pow<2>(time_scale_));

                lambda_neighbors[j] /= sum_Eg_j;
                sigma_ratio_lambda_neighbors[j] /= sum_Eg_j;
            } else {
                double Er_outside;
                double const Er_i = cells_cgs[i].Erad * cells_cgs[i].density;
                boundary_calculator.getOutsideValuesGray(tess, i, neighbor_j, cells_cgs, Er_i, Er_outside, dummy_v);

                lambda_neighbors[j] /= Er_outside;
                sigma_ratio_lambda_neighbors[j] /= Er_outside;
            }

            if(tess.GetFaceNeighbors(faces[j]).first == i){
                lambda_face_gray[faces[j]].first = lambda_neighbors[j];
                sigma_ratio_lambda_face_gray[faces[j]].first = sigma_ratio_lambda_neighbors[j];
            } else {
                assert(tess.GetFaceNeighbors(faces[j]).second == i);
                lambda_face_gray[faces[j]].second = lambda_neighbors[j];
                sigma_ratio_lambda_face_gray[faces[j]].second = sigma_ratio_lambda_neighbors[j];
            }
        }

        if(mix_frame_on_){
            double const v_limiter = std::min(1.0, 0.1 * CG::speed_of_light / (fastabs(cells_cgs[i].velocity) + 1e-2));

            for(std::size_t j=0; j < Nneighbors; ++j){
                std::size_t const neighbor_j = neighbors[j];

                Vector3D r_ij = r_i - tess.GetMeshPoint(neighbor_j);
                r_ij *= 1.0 / abs(r_ij);

                double const A_ij = tess.GetArea(faces[j]) * pow<2>(length_scale_);

                double const momentum_relativity_coefficient = -0.5 * dt_cgs * A_ij * ScalarProd(cells_cgs[i].velocity, r_ij) / 3.0;

                double const relativity_term_i = momentum_relativity_coefficient * v_limiter * 2.0 * f * sigma_ratio_lambda_i;
                
                double const momentum_term_i = -momentum_relativity_coefficient * lambda_i;

                double const relativity_term_j = momentum_relativity_coefficient * v_limiter * 2.0 * f * sigma_ratio_lambda_neighbors[j];

                double const momentum_term_j = -momentum_relativity_coefficient * lambda_neighbors[j];


                if(!tess.IsPointOutsideBox(neighbor_j)){
                    A[i][0] += relativity_term_i;
                    A[i][0] += momentum_term_i;

                    auto it = std::find(A_indeces[i].begin(), A_indeces[i].end(), neighbor_j);
                    
                    if(it == A_indeces[i].end()){
                        throw UniversalError("Key not equal in multigroup diffusion");
                    }

                    std::size_t const neighbor_counter = static_cast<std::size_t>(it - A_indeces[i].begin());

                    if(A_indeces[i][neighbor_counter] != neighbor_j){
                        throw UniversalError("Key not equal in multigroup diffusion");
                    }

                    A[i][neighbor_counter] += relativity_term_j;
                    A[i][neighbor_counter] += momentum_term_j;
                } else {
                    boundary_calculator.setMomentumTermBoundaryGray(tess, i, neighbor_j, dt_cgs, cells_cgs, relativity_term_i + momentum_term_i, relativity_term_j + momentum_term_j, A[i][0], b[i]);
                }
            } 
        } else {
            for(std::size_t j=0; j<Nneighbors; ++j){
                std::size_t const neighbor_j = neighbors[j];

                auto const r_ij = normalize(r_i - tess.GetMeshPoint(neighbor_j));

                double const A_ij = tess.GetArea(faces[j]) * pow<2>(length_scale_);

                Vector3D velocity_j;

                if(!tess.IsPointOutsideBox(neighbor_j)){
                    velocity_j = cells_cgs[neighbor_j].velocity;
                } else {
                    double dummyEg_i, dummy_Eg_j;
                    boundary_calculator.getOutsideValuesGray(tess, i, neighbor_j, cells_cgs, dummyEg_i, dummy_Eg_j, velocity_j);
                }
                // TODO add the R2_g part 
                A[i][0] += -0.5*ScalarProd(cells_cgs[i].velocity+velocity_j, r_ij) * A_ij * dt_cgs / 3.0;
            }
        }
    }


    // Find maximum number of neighbors and allocate data
    // THIS SHOULD BE IN PRESTEP BUT BiCGSTAB CREATES A NEW MATRIX EVERY TIME IT IS CALLED. 
    // MAYBE MATRIX BUILDER SHOULD HOLD A MATRIX AS AN ATTRIBUTE
    
    std::size_t max_neighbors = 0;
    for(std::size_t i=0; i < Nlocal; ++i){
        max_neighbors = std::max(max_neighbors, tess.GetNeighbors(i).size());
    }
    ++max_neighbors;
    
    for(std::size_t i=0; i < Nlocal; ++i){
        A[i].resize(max_neighbors, 0);
        A_indeces[i].resize(max_neighbors, max_size_t);

        if(A[i][0] < 0){
            std::cout << "Negative A in matrix build" << std::endl;
        }
    }
}

void MultigroupDiffusion::PostCG(Tessellation3D const& tess, 
                                 std::vector<Conserved3D>& extensives, 
                                 double const dt, 
                                 std::vector<ComputationalCell3D>& cells,
                                 std::vector<double>const& CG_result, 
                                 std::vector<double> const&  full_CG_result) const {

    if(gray){
        PostCGGray(tess, extensives, dt, cells, CG_result, full_CG_result);
    } else {
        assert(current_group < ENERGY_GROUPS_NUM);

        PostCGGroup(current_group, tess, extensives, dt, cells, CG_result, full_CG_result);
    }
}

void MultigroupDiffusion::PostCGGroup(std::size_t const group,
                                      Tessellation3D const& tess, 
                                      std::vector<Conserved3D>& extensives, 
                                      double const dt, 
                                      std::vector<ComputationalCell3D>& cells,
                                      std::vector<double>const& CG_result, 
                                      std::vector<double> const&  full_CG_result) const {
    

    auto const N = tess.GetPointNo();

    for(std::size_t i=0; i<N; ++i){
        double const volume = tess.GetVolume(i) * pow<3>(length_scale_);
        double const full_CG_res_i = std::max(full_CG_result[i], std::numeric_limits<double>::min()*1e100);
        extensives[i].Eg[group] = full_CG_res_i * volume * pow<2>(time_scale_) / (pow<2>(length_scale_) * mass_scale_);
        cells[i].Eg[group] =  extensives[i].Eg[group] / extensives[i].mass;
    }
#ifdef RICH_MPI
    size_t const Ncells_total = cells.size();
    double const min_value = std::numeric_limits<double>::min() * 1e20;
    for(size_t i = N; i < Ncells_total; ++i)
    {
        double const full_CG_res_i = std::max(full_CG_result[i], std::numeric_limits<double>::min()*1e100);
        if(cells[i].density > min_value)
            cells[i].Eg[group] = full_CG_res_i * pow<2>(time_scale_) / (cells[i].density * mass_scale_ / length_scale_);
    }
#endif
}

void MultigroupDiffusion::PostCGFull(Tessellation3D const& tess, 
                                     std::vector<Conserved3D>& extensives, 
                                     double const dt, 
                                     std::vector<ComputationalCell3D>& cells,
                                     std::vector<double>const& CG_result, 
                                     std::vector<double> const&  full_CG_result) const {
    
    auto const N = tess.GetPointNo();
    std::vector<size_t> neighbors;
    face_vec faces;
    Vector3D dummy_v;
    Vector3D dP;

    double Einit = 0.0;
    for(std::size_t i = 0; i < N; ++i){
        Einit += extensives[i].Erad + extensives[i].energy;
    }

#ifdef RICH_MPI
    MPI_Allreduce(MPI_IN_PLACE, &Einit, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
#endif

    int good_end = 1;
    
    double const dt_cgs = dt * time_scale_;
    double const cdt = CG::speed_of_light*dt_cgs;

    bool with_entropy = false;
    size_t entropy_index = ComputationalCell3D::tracerNames.size();
    std::vector<std::string>::const_iterator it = binary_find(ComputationalCell3D::tracerNames.begin(),
						  ComputationalCell3D::tracerNames.end(), string("Entropy"));
    if (it != ComputationalCell3D::tracerNames.end())
    {
        entropy_index = static_cast<size_t>(it - ComputationalCell3D::tracerNames.begin());
        with_entropy = true;
    }

    for(std::size_t i=0; i < N; ++i){
        double const old_e_therm = extensives[i].internal_energy;
        double const volume = tess.GetVolume(i) * pow<3>(length_scale_);
        double Erad_tot = 0;
        double const f = fleck_factor[i];
        double const T  = old_Tm[i];
        double const kp = sigma_absorption_planck[i];
        double const Um = get_radiation_energy_density(T);
        
        double dE_absorption_emission = -volume * f * cdt * kp*Um;
        
        double dE_compton = 0.0;
        
        if(compton_on_){
            generate_S_and_dSdUm_matrices(cells[i], i);
            calculate_compton_quantities(cells[i], i);

            for(std::size_t g=0; g < ENERGY_GROUPS_NUM; ++g){
                
                dE_compton -= get_implicit_compton_contribution_to_b(tess, cells[i], i, g, dt_cgs);
                
                auto const bg = planck_integal_group[i][g];
                double const Gamma_1 = 1.0 / Gammas[i];


                for(std::size_t gt=0; gt < ENERGY_GROUPS_NUM; ++gt){
                    double const full_CG_res_i = std::max(full_CG_result[i * ENERGY_GROUPS_NUM + gt], std::numeric_limits<double>::min()*1e100);
                    
                    double const implicit_conribution_group_j = -volume*bg * (1 - f) * sigma_absorption_group[i][gt] * sigma_absorption_group[i][g] * cdt * Gamma_1;
                    
                    dE_compton += volume*implicit_conribution_group_j * std::max(full_CG_result[i * ENERGY_GROUPS_NUM + gt], std::numeric_limits<double>::min()*1e100);

                    dE_compton += get_implicit_compton_contribution(tess, cells[i], i, g, gt, dt_cgs) * full_CG_res_i;
                    
                }
            }
        }

        for(size_t group = 0; group < ENERGY_GROUPS_NUM; ++group){
            
            double const CG_res = std::max(CG_result[i * ENERGY_GROUPS_NUM + group], std::numeric_limits<double>::min()*1e100);
            
            double const full_CG_res_i = std::max(full_CG_result[i * ENERGY_GROUPS_NUM + group], 
            std::numeric_limits<double>::min()*1e100);
            
            extensives[i].Eg[group] = CG_res * volume * pow<2>(time_scale_) / (pow<2>(length_scale_) * mass_scale_);
            
            cells[i].Eg[group] =  extensives[i].Eg[group] / extensives[i].mass;
            Erad_tot += extensives[i].Eg[group];
            // absorption + emission
            
            // dE_absorption_emission += f * volume * cdt * full_CG_res_i * sigma_absorption_group[i][group];
            dE_absorption_emission += volume * cdt * full_CG_res_i * sigma_absorption_group[i][group];
        }
        
        dE_absorption_emission *= pow<2>(time_scale_) / (pow<2>(length_scale_) * mass_scale_);
        dE_compton *= pow<2>(time_scale_) / (pow<2>(length_scale_) * mass_scale_);
        
        extensives[i].energy += dE_absorption_emission + dE_compton;
        extensives[i].internal_energy += dE_absorption_emission + dE_compton; 
        extensives[i].Erad = Erad_tot;
        cells[i].Erad =  extensives[i].Erad / extensives[i].mass;
        cells[i].internal_energy =  extensives[i].internal_energy / extensives[i].mass;
        // if(cells[i].ID == 1322876)
        // {
        //     std::cout<<cells[i]<<std::endl;
        //     std::cout<<"dE_absorption_emission "<<dE_absorption_emission<<" f "<<f<<" kr "<<kr<<" kp "<<kp<<" R "<<abs(tess.GetMeshPoint(i))<<std::endl;
        //     std::cout<<"Erad "<<extensives[i].Erad<<std::endl;
        //     std::cout<<"internal_energy "<<extensives[i].internal_energy<<std::endl;
        // }
   
        tess.GetNeighbors(i, neighbors);
        std::size_t const Nneighbors = neighbors.size();
        faces = tess.GetCellFaces(i);
        auto const r_i = tess.GetMeshPoint(i);
        
        // momentum term
        if(hydro_on_){
            Vector3D dP;
            for(size_t group = 0; group < ENERGY_GROUPS_NUM; ++group){
                Vector3D gradEg;
                double Eg_j, Eg_i = cells_cgs[i].Eg[group] * cells_cgs[i].density;
                for(size_t j=0; j<Nneighbors; ++j){
                    size_t const neighbor_j = neighbors[j];
                    Vector3D const r_ij = normalize(r_i - tess.GetMeshPoint(neighbor_j));
                    double const A_ij = tess.GetArea(faces[j]) * pow<2>(length_scale_);
                    bool const is_outside = tess.IsPointOutsideBox(neighbor_j);
                    if(!is_outside)
                        gradEg += A_ij * 0.5 * (Eg_i + cells_cgs[neighbor_j].Eg[group] * cells_cgs[neighbor_j].density) * r_ij;
                    else{
                        Vector3D dummy_v;
                        boundary_calculator.getOutsideValuesGroup(group, tess, i, neighbor_j, cells_cgs, Eg_i, Eg_j, dummy_v);
                        gradEg += A_ij * 0.5 * (Eg_i + Eg_j) * r_ij;
                    }
                }
                gradEg *= 1.0 / volume;
                double const D = coefficient_calculator.CalcDiffusionCoefficientGroup(cells_cgs[i], group);
                double const flux_limit = CG::CalcSingleFluxLimiter(gradEg, D, Eg_i);
                dP += (flux_limit / 3) * gradEg;
            }
            dP *= dt_cgs * time_scale_ / (length_scale_ * mass_scale_);
            extensives[i].momentum += dP;
            double mass_i = extensives[i].mass;
    
            double const new_Ek = 0.5 * ScalarProd(extensives[i].momentum, extensives[i].momentum) / mass_i;

            extensives[i].energy = extensives[i].internal_energy + new_Ek;
        }
        // EOS
        try{
            if(minimum_temperature_ > 0){
                double const min_e_therm = eos_.dT2e(cells[i].density, minimum_temperature_, cells[i].tracers, ComputationalCell3D::tracerNames);
                if(min_e_therm > cells[i].internal_energy){
                    double const delta_e = min_e_therm - cells[i].internal_energy;
                    cells[i].internal_energy += delta_e;
                    extensives[i].energy += delta_e * extensives[i].mass;
                    extensives[i].internal_energy += delta_e * extensives[i].mass;
                }
            }
            cells[i].temperature = eos_.de2T(cells[i].density, cells[i].internal_energy, cells[i].tracers, ComputationalCell3D::tracerNames);
            cells[i].pressure = eos_.de2p(cells[i].density, cells[i].internal_energy, cells[i].tracers, ComputationalCell3D::tracerNames);
            cells[i].velocity = extensives[i].momentum / extensives[i].mass;    
            if(with_entropy)
            {
                double new_entropy = eos_.dp2s(cells[i].density, cells[i].pressure, cells[i].tracers, ComputationalCell3D::tracerNames);
                cells[i].tracers[entropy_index] = new_entropy;
                extensives[i].tracers[entropy_index] = new_entropy * extensives[i].mass;
            }
        } catch(UniversalError &eo){
            eo.addEntry("MultigroupDiffusion, cell index", i);
            eo.addEntry("ID", cells[i].ID);
            eo.addEntry("density", cells[i].density);
            eo.addEntry("internal_energy", cells[i].internal_energy);
            eo.addEntry("dE_absorption_emission", dE_absorption_emission);
            eo.addEntry("internal_energy extensive", extensives[i].internal_energy);
            eo.addEntry("f", f);
            eo.addEntry("kp", kp);
            eo.addEntry("Um", Um);
            eo.addEntry("old T", old_Tm[i]);
            eo.addEntry("volume", volume);
            eo.addEntry("cdt", cdt);
            for(size_t group = 0; group < ENERGY_GROUPS_NUM; ++group){
                eo.addEntry("Eg[" + std::to_string(group) + "]", full_CG_result[i * ENERGY_GROUPS_NUM + group]);
                eo.addEntry("sigma_absorption_group[" + std::to_string(group) + "]", sigma_absorption_group[i][group]);
            }
            throw eo;
        }
    }

    int rank = 0;
#ifdef RICH_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    bool was_bad = false;
    if(good_end == 0){
        std::cout<<"Zero good_end rank "<<rank<<std::endl;
        was_bad = true; 
    }
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &good_end, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    if(was_bad){
        std::cout<<"rank "<<rank<<" good_end "<<good_end<<std::endl;
    }
#endif

    if(good_end = 0){
        throw UniversalError("Negative energy in PostCGFull");
    }

    double Efinal = 0;
    for(std::size_t i=0; i<N; ++i){
        Efinal += extensives[i].Erad + extensives[i].energy;
    }

#ifdef RICH_MPI
    MPI_Allreduce(MPI_IN_PLACE, &Efinal, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif

#ifdef DEBUG
    if(rank == 0){
        std::cout << std::setprecision(14) << "Einit = " << Einit << ", Efinal = " << Efinal << std::endl;
    }
#endif
}


void MultigroupDiffusion::PostCGGray(Tessellation3D const& tess, 
                                     std::vector<Conserved3D>& extensives, 
                                     double const dt, 
                                     std::vector<ComputationalCell3D>& cells,
                                     std::vector<double>const& CG_result, 
                                     std::vector<double> const&  full_CG_result) const {
    
    auto const N = tess.GetPointNo();
    std::vector<size_t> neighbors;
    face_vec faces;
    Vector3D dummy_v;
    Vector3D dP;

    double Einit = 0.0;
    for(std::size_t i = 0; i < N; ++i){
        Einit += extensives[i].Erad + extensives[i].energy;
    }

#ifdef RICH_MPI
    MPI_Allreduce(MPI_IN_PLACE, &Einit, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
#endif

    int good_end = 1;
    
    double const dt_cgs = dt * time_scale_;
    double const cdt = CG::speed_of_light*dt_cgs;

    for(std::size_t i=0; i < N; ++i){
        double const old_e_therm = extensives[i].internal_energy;
        double const volume = tess.GetVolume(i) * pow<3>(length_scale_);

        double const CG_res = std::max(CG_result[i], std::numeric_limits<double>::min()*1e100);
        double const full_CG_res_i = std::max(full_CG_result[i], std::numeric_limits<double>::min()*1e100);
        
        extensives[i].Erad = CG_res * volume * pow<2>(time_scale_) / (pow<2>(length_scale_) * mass_scale_);

        double const T  = old_Tm[i];
        double const kp = sigma_absorption_planck[i];
        double const kr = sigma_absorption_average[i];
        double const f = fleck_factor[i];
        double const Um = get_radiation_energy_density(T);

        // absorption + emission
        double dE_absorption_emission = volume * f * cdt * (kr*full_CG_res_i - kp*Um);
        dE_absorption_emission *= pow<2>(time_scale_) / (pow<2>(length_scale_) * mass_scale_);
        
        if(cells[i].ID == 1322876)
        {
            std::cout<<cells[i]<<std::endl;
            std::cout<<"dE_absorption_emission "<<dE_absorption_emission<<" f "<<f<<" kr "<<kr<<" kp "<<kp<<" R "<<abs(tess.GetMeshPoint(i))<<std::endl;
            std::cout<<"Erad "<<extensives[i].Erad<<std::endl;
            std::cout<<"internal_energy "<<extensives[i].internal_energy<<std::endl;
        }
        
        extensives[i].energy = extensives_temp[i].energy;
        extensives[i].energy += dE_absorption_emission;
        
        extensives[i].internal_energy = extensives_temp[i].internal_energy;
        extensives[i].internal_energy += dE_absorption_emission;

        tess.GetNeighbors(i, neighbors);
        std::size_t const Nneighbors = neighbors.size();
        faces = tess.GetCellFaces(i);
        auto const r_i = tess.GetMeshPoint(i);
        
        // relativity term
        double dE_relativity = 0.0;
        if(mix_frame_on_){
            
            double const v_limiter = std::min(1.0, 0.1 * CG::speed_of_light / (fastabs(cells_cgs[i].velocity) + 1e-2));

            for(std::size_t j=0; j<Nneighbors; ++j){
                std::size_t const neighbor_j = neighbors[j];
                auto r_ij = r_i - tess.GetMeshPoint(neighbor_j);
                r_ij *= 1.0 / abs(r_ij);
                double const A_ij = tess.GetArea(faces[j]) * pow<2>(length_scale_);

                double const momentum_relativity_coefficient_j = -0.5 * dt_cgs * A_ij * ScalarProd(cells_cgs[i].velocity, r_ij) / 3.0;

                double relativity_term_neighbor_j = 0.0;
                if(tess.GetFaceNeighbors(faces[j]).first == i){
                    relativity_term_neighbor_j = momentum_relativity_coefficient_j * v_limiter * 2.0 * f * sigma_ratio_lambda_face_gray[faces[j]].first;
                } else {
                    assert(tess.GetFaceNeighbors(faces[j]).second == i);
                    relativity_term_neighbor_j = momentum_relativity_coefficient_j * v_limiter * 2.0 * f * sigma_ratio_lambda_face_gray[faces[j]].second;
                }

                if(!tess.IsPointOutsideBox(neighbor_j)){
                    double const full_CG_res_j = std::max(full_CG_result[neighbor_j], std::numeric_limits<double>::min()*1e100);

                    dE_relativity -= relativity_term_neighbor_j * full_CG_res_j;
                } else {
                    double Er_outside;
                    boundary_calculator.getOutsideValuesGray(tess, i, neighbor_j, cells, full_CG_result[i], Er_outside, dummy_v);
                    dE_relativity -= relativity_term_neighbor_j * Er_outside;
                }

                double const relativity_term_i = momentum_relativity_coefficient_j * v_limiter * 2.0 * f * sigma_ratio_lambda_cell_gray[i];

                dE_relativity -= relativity_term_i * full_CG_res_i;
            }

            dE_relativity *= pow<2>(time_scale_) / (pow<2>(length_scale_) * mass_scale_);

            extensives[i].energy += dE_relativity;
            extensives[i].internal_energy += dE_relativity;
        }

        // momentum term
        if(hydro_on_){
            Vector3D lambda_grad_Er(0.0, 0.0, 0.0);
            double dE_momentum = 0.0;
            
            double mass_i = extensives[i].mass;

            double const old_Ek = 0.5 * ScalarProd(extensives[i].momentum, extensives[i].momentum) / mass_i;

            for(std::size_t j=0; j<Nneighbors; ++j){
                auto const neighbor_j = neighbors[j];

                auto r_ij = r_i - tess.GetMeshPoint(neighbor_j);
                r_ij *= 1.0 / abs(r_ij);

                double const A_ij = tess.GetArea(faces[j]) * pow<2>(length_scale_);
                double const lambda_i = lambda_cell_gray[i];

                double lambda_j = 0.0;
                if(tess.GetFaceNeighbors(faces[j]).first == i){
                    lambda_j = lambda_face_gray[faces[j]].first;
                } else {
                    assert(tess.GetFaceNeighbors(faces[j]).second == i);
                    lambda_j = lambda_face_gray[faces[j]].second;
                }
                
                double Er_j = 0.0;
                if(!tess.IsPointOutsideBox(neighbor_j)){
                    double const full_CG_res_j = std::max(full_CG_result[neighbor_j], std::numeric_limits<double>::min()*1e100);
                    Er_j = full_CG_res_j;
                } else {
                    boundary_calculator.getOutsideValuesGray(tess, i, neighbor_j, cells, full_CG_result[i], Er_j, dummy_v);
                }


                lambda_grad_Er += (0.5*A_ij * (lambda_i * full_CG_res_i + lambda_j * Er_j) / 3.0) * r_ij;
            }

            
            dP = (dt_cgs * time_scale_ / (length_scale_ * mass_scale_))  * lambda_grad_Er;
            double const E_dE = ScalarProd(dP, extensives[i].momentum) / mass_i;
            
            extensives[i].momentum += dP;
    
            double const new_Ek = 0.5 * ScalarProd(extensives[i].momentum, extensives[i].momentum) / mass_i;

            double const dE_dP = -new_Ek + old_Ek + E_dE;
            extensives[i].Erad += dE_dP;

            if(extensives[i].Erad < 0 && dE_dP < 0 && std::abs(dE_dP) < extensives[i].energy * 0.01) {
                 extensives[i].Erad -= dE_dP;            
            }

            extensives[i].energy = extensives[i].internal_energy + new_Ek;
        }


        // other terms
        cells[i].Erad = extensives[i].Erad / extensives[i].mass;
        cells[i].internal_energy = extensives[i].internal_energy / extensives[i].mass;
        if(!(cells[i].internal_energy > 0))// ||  cells[i].ID == 79650)
        {
            UniversalError eo("negative internal energy in PostCGGray");
            eo.addEntry("Cell", i);
            eo.addEntry("ID", cells[i].ID);
            eo.addEntry("Internal Energy", extensives[i].internal_energy);
            eo.addEntry("Mass", extensives[i].mass);
            eo.addEntry("dE_relativity", dE_relativity);
            eo.addEntry("dE_absorption_emission", dE_absorption_emission);
            eo.addEntry("kr*full_CG_res_i", kr*full_CG_res_i);
            eo.addEntry("kp*Um", kp*Um);
            eo.addEntry("kr", kr);
            eo.addEntry("kp", kp);
            throw eo;
        }

        try{
            cells[i].temperature = eos_.de2T(cells[i].density, cells[i].internal_energy, cells[i].tracers, ComputationalCell3D::tracerNames);
            cells[i].pressure = eos_.de2p(cells[i].density, cells[i].internal_energy, cells[i].tracers, ComputationalCell3D::tracerNames);

            cells[i].velocity = extensives[i].momentum / extensives[i].mass;    

        } catch(UniversalError &eo){
            reportError(eo);
            good_end = 0;
            break;
        }
    }

    int rank = 0;
#ifdef RICH_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    bool was_bad = false;
    if(good_end == 0){
        std::cout<<"Zero good_end rank "<<rank<<std::endl;
        was_bad = true; 
    }
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &good_end, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    if(was_bad){
        std::cout<<"rank "<<rank<<" good_end "<<good_end<<std::endl;
    }
#endif

    if(good_end = 0){
        throw UniversalError("Negative energy in PostCGGray");
    }

    double Efinal = 0;
    for(std::size_t i=0; i<N; ++i){
        Efinal += extensives[i].Erad + extensives[i].energy;
    }

#ifdef RICH_MPI
    MPI_Allreduce(MPI_IN_PLACE, &Efinal, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif


#ifdef DEBUG
    if(rank == 0){
        std::cout << std::setprecision(14) << "Einit = " << Einit << ", Efinal = " << Efinal << std::endl;
    }
#endif
}

void MultigroupDiffusion::calculate_fleck_factor(Tessellation3D const& tess, std::vector<ComputationalCell3D> const& cells, double dt_cgs) const
{
    size_t const N = tess.GetPointNo();
    for(size_t i = 0; i < N; ++i)
    {
        double const sigma_planck = sigma_absorption_planck[i];
        double const T = old_Tm[i];
        double cv = eos_.dT2cv(cells[i].density, T, cells[i].tracers, ComputationalCell3D::tracerNames);
        
        // TODO: What is energy ratio (see Diffusion.cpp same line)
        cv *= mass_scale_ / (pow<2>(time_scale_)*length_scale_);
        double const cv_bar = cv / get_radiation_cv(T);
        double const f = CG::FleckFactor(dt_cgs, 1.0/cv_bar, sigma_planck);

        if(f < 0){
            throw UniversalError("Negative fleck factor");
        }
        fleck_factor[i] = f;
    }
}

void MultigroupDiffusion::calculate_group_absorption_and_scattering_coefficients(Tessellation3D const& tess,
                                                                                 std::vector<ComputationalCell3D> const& cells,
                                                                                 double const dt) const {
    auto const N = tess.GetPointNo();
    sigma_absorption_group.resize(N);
    sigma_scattering_group.resize(N);
    for(std::size_t i=0; i < N; ++i){
        for(std::size_t g=0; g < ENERGY_GROUPS_NUM; ++g){
            auto const& cell = cells[i];
            sigma_absorption_group[i].resize(ENERGY_GROUPS_NUM);
            sigma_scattering_group[i].resize(ENERGY_GROUPS_NUM);

            sigma_absorption_group[i][g] = std::min(coefficient_calculator.CalcAbsorptionCoefficientGroup(cell, g),
                CG::max_coupling_strength / (CG::speed_of_light * dt));

            if(sigma_absorption_group[i][g] < 0.){
                throw UniversalError("negative absorption coefficient");
            }

            sigma_scattering_group[i][g] = coefficient_calculator.CalcScatteringCoefficientGroup(cell, g);

            if(sigma_scattering_group[i][g] < 0.){
                throw UniversalError("negative scattering coefficient");
            }
        }
    }
}

void MultigroupDiffusion::calculate_planck_integrals(Tessellation3D const& tess,
                                                     std::vector<ComputationalCell3D> const& cells) const {
    
    auto const N = tess.GetPointNo();

    planck_integal_group.resize(N);
    for(std::size_t i=0; i<N; ++i){
        planck_integal_group[i].resize(ENERGY_GROUPS_NUM);
        double const kT = CG::boltzmann_constant * old_Tm[i];
        double planck_sum = 0.0;
        for(std::size_t g=0; g<ENERGY_GROUPS_NUM; ++g){

            double const a = energy_groups_boundary[g] / kT;
            double const b = energy_groups_boundary[g+1] / kT;

            double const bg = planck_integral::planck_integral(a, b);

            planck_integal_group[i][g] = bg;
            planck_sum += bg;
        }

        auto const& cell = cells_cgs[i];
        if(planck_sum < (1. - 1e-4) && not displayed_warning_){
            displayed_warning_ = true;
            std::cout << "bad groups! planckian not covered well! cell " << i << " T " << old_Tm[i] <<" ID "<<cells[i].ID<<std::endl;
            std::cout << "bad planck_sum " << planck_sum << std::endl;
            // throw UniversalError("bad groups! planckian not covered well!");
        }
    }
}

void MultigroupDiffusion::calculate_gray_absorption_and_scattering_coefficients(Tessellation3D const& tess,
                                                                                std::vector<ComputationalCell3D> const& cells) const {
    auto const N = tess.GetPointNo();
    std::fill(sigma_absorption_planck.begin(), sigma_absorption_planck.end(), 0.0);
    std::fill(sigma_absorption_average.begin(), sigma_absorption_average.end(), 0.0);
    std::fill(sigma_scattering_gray.begin(), sigma_scattering_gray.end(), 0.0);

    for(std::size_t i=0; i<N; ++i){
        double sum_U = 0.0;
        auto const& cell = cells[i];
        for(std::size_t g=0; g < ENERGY_GROUPS_NUM; ++g){
            double const sigma = sigma_absorption_group[i][g];
            double const sigma_scattering = sigma_scattering_group[i][g];
            double const bg = planck_integal_group[i][g];
            // the change of units is not really important since we are averaging with Ug as weights i.e. the units cancel. But it is for consistency.
            double const Ug = cell.Eg[g] * cell.density * pow<2>(length_scale_) / pow<2>(time_scale_);
            
            sigma_absorption_planck[i] += sigma * bg;
            sigma_absorption_average[i] += sigma * Ug;
            sigma_scattering_gray[i] += sigma_scattering * Ug;

            sum_U += Ug;
        }

        if(sum_U > std::numeric_limits<double>::min()*1e40) {
            sigma_absorption_average[i] /= sum_U;
            sigma_scattering_gray[i] /= sum_U;
        }
    }
}

void MultigroupDiffusion::solve_doppler_shift(Tessellation3D const& tess,
                                              std::vector<ComputationalCell3D>& cells,
                                              double const dt, std::vector<Conserved3D>& extensives) const {

    double const dt_cgs = dt * time_scale_;
    auto const N = tess.GetPointNo();
    
    double max_change_doppler = 0.0;
    std::size_t max_group = 0;
    std::size_t max_cell = 0;
    double old_energy = 0.0;

    std::vector<std::size_t> neighbors;
    face_vec faces;
    Vector3D dummy_v;
    Vector3D grad_Eg(0.0, 0.0, 0.0);
    ComputationalCell3D dummy_cell;

    if(mix_frame_on_){

    // #ifdef RICH_MPI    
    //     MPI_exchange_data2(tess, new_Eg, true);
    //     MPI_exchange_data2(tess, R2[0],  true);
    //     MPI_exchange_data2(tess, R2[1],  true);
    //     MPI_exchange_data2(tess, R2[2],  true);
    // #endif

        std::vector<double> lambda_g_dummy(N, 0.0);
        calculate_lambda_g_and_R2_g(0, tess, cells, lambda_g_dummy, R2[2]);

        struct
        {
            double val;
            int mpi_id;
        }max_data;

        for(std::size_t g=0; g < ENERGY_GROUPS_NUM; ++g){
            // for the discretization we need R2 for the groups g-1, g, g+1
            // R2[0] -> g-1
            // R2[1] -> g
            // R2[2] -> g+1 
            std::swap(R2[1], R2[0]);
            std::swap(R2[2], R2[1]);

            if(g+1 < ENERGY_GROUPS_NUM){
                calculate_lambda_g_and_R2_g(g+1, tess, cells, lambda_g_dummy, R2[2]);
            }

            for(std::size_t i=0; i < N; ++i){
                tess.GetNeighbors(i, neighbors);
                auto const Nneighbors = neighbors.size();
                
                faces = tess.GetCellFaces(i);

                double const Eg_i = cells_cgs[i].Eg[g] * cells_cgs[i].density;
                Vector3D const& velocity_i = cells_cgs[i].velocity;
                
                Vector3D const r_i = tess.GetMeshPoint(i);

                double divergence = 0.0;
                for(std::size_t j=0; j<Nneighbors; ++j){
                    std::size_t neighbor_j = neighbors[j];

                    bool const is_outside = tess.IsPointOutsideBox(neighbor_j);

                    Vector3D r_ij = r_i - tess.GetMeshPoint(neighbor_j);
                    r_ij *= 1.0 / abs(r_ij);
                    
                    double const A_ij = tess.GetArea(faces[j]) * pow<2>(length_scale_);

                    Vector3D velocity_j;
                    if(!is_outside){
                        velocity_j = cells_cgs[neighbor_j].velocity;
                    } else {
                        double dummyEg;
                        boundary_calculator.getOutsideValuesGroup(g, tess, i, neighbor_j, cells, Eg_i, dummyEg, velocity_j);
                        velocity_j *= length_scale_ / time_scale_;
                    }

                    double const v_mid = ScalarProd(r_ij, velocity_i + velocity_j) / 2.0;

                    
                    double const nu_g   = energy_groups_center[g];
                    double const dnu_g  = energy_groups_width[g];

                    if(v_mid > 0.0){
                        double const v_ij = ScalarProd(r_ij, velocity_j);
                        
                        std::size_t which_R2 = neighbor_j;
                        if(g+1 < ENERGY_GROUPS_NUM){
                            double const nu_g1  = energy_groups_center[g+1];
                            double const dnu_g1 = energy_groups_width[g+1];
                            
                            double Eg1_j;
                            if(!is_outside){
                                Eg1_j = cells_cgs[neighbor_j].Eg[g+1] * cells_cgs[neighbor_j].density;
                            } else {
                                double const Eg1_i = cells_cgs[i].Eg[g+1] * cells_cgs[i].density;
                                boundary_calculator.getOutsideValuesGroup(g+1, tess, i, neighbor_j, cells, Eg1_i, Eg1_j, dummy_v);

                                which_R2 = i;
                            }
                            
                            divergence += A_ij * v_ij * 0.5 * nu_g1 * Eg1_j * (1.0 - R2[2][which_R2]) / dnu_g1;
                        } 
                        
                        double Eg_j;
                        if(!is_outside){
                            Eg_j = cells_cgs[neighbor_j].Eg[g] * cells_cgs[neighbor_j].density;
                            which_R2 = neighbor_j;
                        } else {
                            boundary_calculator.getOutsideValuesGroup(g, tess, i, neighbor_j, cells, Eg_i, Eg_j, dummy_v);
                            which_R2 = i;
                        }

                        divergence -= A_ij * v_ij * 0.5 * nu_g * Eg_j * (1.0 - R2[1][which_R2]) / dnu_g;
                    } else {
                        double const v_ij = ScalarProd(r_ij, velocity_i);
                        
                        if (g > 0){    
                            double const nu_gm1  = energy_groups_center[g-1];
                            double const dnu_gm1 = energy_groups_width[g-1];
                            
                            double const Egm1_i = cells_cgs[i].Eg[g-1] * cells_cgs[i].density;
                            

                            divergence -= A_ij * v_ij * 0.5 * nu_gm1 * Egm1_i * (1.0 - R2[0][i]) / dnu_gm1;
                        }

                        double const Eg_i  = cells_cgs[i].Eg[g] * cells_cgs[i].density;

                        divergence += A_ij * v_ij * 0.5 * nu_g * Eg_i * (1.0 - R2[1][i]) / dnu_g;
                    }    
                }

                double const change = std::abs(dt_cgs*divergence) / (cells_cgs[i].Eg[g] * extensives[i].mass * mass_scale_);
                if(change > max_change_doppler){
                    max_change_doppler = change;   
                    max_group = g;
                    max_cell = i;
                    old_energy = cells_cgs[i].Eg[g] / cells_cgs[i].density;
                }

                double const dEg = dt_cgs * divergence;
                
                extensives[i].Eg[g] += dEg * pow<2>(time_scale_) / (mass_scale_ * pow<2>(length_scale_));
            }
        }
        int rank = 0;
    #ifdef RICH_MPI
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    #endif   
        max_data.val = max_change_doppler;
        max_data.mpi_id = rank;
    #ifdef RICH_MPI
        MPI_Allreduce(MPI_IN_PLACE, &max_data, 1, MPI_DOUBLE_INT, MPI_MAXLOC, MPI_COMM_WORLD);
    #endif
    if(rank == max_data.mpi_id)
        std::cout << "Maximum change in energy group " << max_group << " in cell " << max_cell << " due to Doppler effect: " << max_change_doppler <<" ID "<<cells[max_cell].ID<<" old Eg:"<<cells[max_cell].Eg[max_group]<<" new Eg:"<<
        extensives[max_cell].Eg[max_group] / extensives[max_cell].mass<<" T "<<cells[max_cell].temperature<<std::endl;

        for(std::size_t n=0; n<D.size(); ++n){
            std::fill(D[n].begin(), D[n].end(), 0.0);
        }
        
        // ADD SECOND DOPPLER SHIFT ELEMENT
        for(std::size_t i=0; i < N; ++i){
            D[2][i] = coefficient_calculator.CalcDiffusionCoefficientGroup(cells_cgs[i], 0);
        }
        D[1] = D[2];

        for(std::size_t g=0; g < ENERGY_GROUPS_NUM; ++g){
            std::swap(D[1], D[0]);
            std::swap(D[2], D[1]);

            if(g+1 < ENERGY_GROUPS_NUM){
                for(std::size_t i=0; i < N; ++i){
                    D[2][i] = coefficient_calculator.CalcDiffusionCoefficientGroup(cells_cgs[i], g+1);
                }
            } else {
                D[2] = D[1];
            }
            

            std::size_t const gp = g+1 < ENERGY_GROUPS_NUM ? g+1 : g;
            std::size_t const gm = g > 0 ? g-1 : g;

            double const nu_g_p = energy_groups_boundary[g+1];
            double const nu_g_m = energy_groups_boundary[g];
            
            double const dnu_g = energy_groups_width[g];
            double const dnu_g_p = energy_groups_width[gp];
            double const dnu_g_m = energy_groups_width[gm];

            for(std::size_t i=0; i < N; ++i){
                
                double const sigma_g = coefficient_calculator.CalcScatteringCoefficientGroup(cells_cgs[i], g);
                
                double const sigma_g1 = coefficient_calculator.CalcScatteringCoefficientGroup(cells_cgs[i], gp);

                double const sigma_gm1 = coefficient_calculator.CalcScatteringCoefficientGroup(cells_cgs[i], gm);
                
                double const sigma_p = 0.5*(sigma_g1 + sigma_g);
                double const sigma_m = 0.5*(sigma_g + sigma_gm1);

                double const Dp = 0.5*(D[2][i] + D[1][i]);
                double const Dm = 0.5*(D[1][i] + D[0][i]);

                tess.GetNeighbors(i, neighbors);
                std::size_t const Nneighbors = neighbors.size();
                
                faces = tess.GetCellFaces(i);
                
                double const Eg_i =  cells_cgs[i].Eg[g] * cells_cgs[i].density;
                double const Egp_i = cells_cgs[i].Eg[gp] * cells_cgs[i].density;
                double const Egm_i = cells_cgs[i].Eg[gm] * cells_cgs[i].density;

                double const Enu_gp_i = 0.5 * (Egp_i / dnu_g_p + Eg_i / dnu_g);
                double const Enu_gm_i = 0.5 * (Eg_i / dnu_g    + Egm_i / dnu_g_m);
                
                Vector3D const r_i = tess.GetMeshPoint(i);

                Vector3D grad_Eg_p(0.0, 0.0, 0.0);
                Vector3D grad_Eg_m(0.0, 0.0, 0.0);
                Vector3D grad_Eg;
                for(std::size_t j=0; j < Nneighbors; ++j){
                    std::size_t const neighbor_j = neighbors[j];

                    double const A_ij = tess.GetArea(faces[j]) * pow<2>(length_scale_);

                    Vector3D const r_ij = normalize(r_i - tess.GetMeshPoint(neighbor_j));
                    
                    double Egp_j;
                    double Egm_j;
                    double Eg_j;
                    if(!tess.IsPointOutsideBox(neighbor_j)){
                        Eg_j =  cells_cgs[neighbor_j].Eg[g] * cells_cgs[neighbor_j].density;
                        Egp_j = cells_cgs[neighbor_j].Eg[gp] * cells_cgs[neighbor_j].density;
                        Egm_j = cells_cgs[neighbor_j].Eg[gm] * cells_cgs[neighbor_j].density;
                    } else {
                        boundary_calculator.getOutsideValuesGroup(g, tess, i, neighbor_j, cells, Eg_i, Eg_j, dummy_v);
                        boundary_calculator.getOutsideValuesGroup(gp, tess, i, neighbor_j, cells, Egp_i, Egp_j, dummy_v);
                        boundary_calculator.getOutsideValuesGroup(gm, tess, i, neighbor_j, cells, Egm_i, Egm_j, dummy_v);
                    }

                    double const Enu_gp_j = 0.5 * (Egp_j / dnu_g_p + Eg_j / dnu_g);
                    double const Enu_gm_j = 0.5 * (Eg_j / dnu_g    + Egm_j / dnu_g_m);
                
                    grad_Eg_p += r_ij * (0.5*A_ij*(Enu_gp_i + Enu_gp_j));
                    grad_Eg_m += r_ij * (0.5*A_ij*(Enu_gm_i + Enu_gm_j));
                    grad_Eg += r_ij * (0.5*A_ij*(Eg_j + Eg_i));
                }
                double const lambda = flux_limiter_ ? CG::CalcSingleFluxLimiter(grad_Eg * (1.0 / (pow<3>(length_scale_) * tess.GetVolume(i))), D[1][i], Eg_i) : 1;
                double const dEp = dt_cgs*nu_g_p * sigma_p * Dp * ScalarProd(cells_cgs[i].velocity, grad_Eg_p) / CG::speed_of_light;

                double const dEm = -dt_cgs * nu_g_m * sigma_m * Dm * ScalarProd(cells_cgs[i].velocity, grad_Eg_m) / CG::speed_of_light;

                double const dEg = lambda * (dEp + dEm) / cells_cgs[i].density * pow<2>(time_scale_) / (mass_scale_ * length_scale_);
                
                extensives[i].Eg[g] += dEg * pow<2>(time_scale_) / (mass_scale_ * pow<2>(length_scale_));
                cells[i].Eg[g] = extensives[i].Eg[g] / extensives[i].mass;
            }
        }
    } else {
            for(std::size_t i=0; i < N; ++i){
                double const volume = tess.GetVolume(i) * pow<3>(length_scale_);
                
                faces = tess.GetCellFaces(i);
                tess.GetNeighbors(i, neighbors);

                std::size_t const Nneighbors = neighbors.size();

                Vector3D const r_i = tess.GetMeshPoint(i);
                double div_velocity = 0.0;
                for(std::size_t j=0; j<Nneighbors; ++j){
                    std::size_t const neighbors_j = neighbors[j];

                    Vector3D const r_ij = normalize(r_i - tess.GetMeshPoint(neighbors_j));

                    double const A_ij = tess.GetArea(faces[j]) * pow<2>(length_scale_);

                    Vector3D velocity_j;
                    if(!tess.IsPointOutsideBox(neighbors_j)){
                        velocity_j = cells_cgs[neighbors_j].velocity;
                    } else {
                        double dummy_Eg_i, dummy_Eg_j;
                        boundary_calculator.getOutsideValuesGray(tess, i, neighbors_j, cells, dummy_Eg_i, dummy_Eg_j, velocity_j);
                    }

                    Vector3D const v_mid = 0.5 * (cells_cgs[i].velocity + velocity_j);
                    div_velocity -= A_ij * ScalarProd(v_mid, r_ij);
                }

                for(std::size_t g=0; g<ENERGY_GROUPS_NUM; ++g){
                    double const Eg_i = cells_cgs[i].Eg[g] * cells_cgs[i].density;
 
                    std::size_t gp = g + 1 == ENERGY_GROUPS_NUM ? g : g + 1;
                    std::size_t gm = g == 0 ? g : g - 1;

                    double const dnu_gp = energy_groups_width[gp];
                    double const dnu_g = energy_groups_width[g];
                    double const dnu_gm = energy_groups_width[gm];

                    double nu_gp = energy_groups_boundary[gp];
                    // double nu_g = energy_groups_center[g];
                    double nu_gm = energy_groups_boundary[g];
                    if(g ==0)
                    {
                        if(div_velocity > 0)
                            nu_gp = 0;
                        else
                            nu_gm = 0;
                    }
                    if(g == (ENERGY_GROUPS_NUM - 1))
                    {
                        if(div_velocity < 0)
                            nu_gm = 0;
                        else
                            nu_gp = 0;
                    }

                    double dnuE_dnu;
                    if(div_velocity < 0.0){
                        double const Egm_i = cells_cgs[i].Eg[gm] * cells_cgs[i].density;
                        dnuE_dnu = (nu_gp*Eg_i / dnu_g - nu_gm*Egm_i / dnu_gm);
                    } else {
                        double const Egp_i = cells_cgs[i].Eg[gp] * cells_cgs[i].density;
                        dnuE_dnu = (nu_gp*Egp_i / dnu_gp - nu_gm*Eg_i / dnu_g);
                    }

                    double const dEg_j = div_velocity * dnuE_dnu * dt_cgs * pow<2>(time_scale_) / (mass_scale_ * pow<2>(length_scale_));

                    extensives[i].Eg[g] += dEg_j;
                    cells[i].Eg[g] = extensives[i].Eg[g] / extensives[i].mass;
                }
            }
        }
#ifdef RICH_MPI
    ComputationalCell3D cdummy;
	MPI_exchange_data(tess, cells, true, &cdummy);	
#endif
} 

void MultigroupDiffusion::calculate_lambda_g_and_R2_g(std::size_t const group,
                                                      Tessellation3D const& tess,
                                                      std::vector<ComputationalCell3D> const& cells,
                                                      std::vector<double>& lambda_g,
                                                      std::vector<double>& R2_g) const {
    
    std::size_t const N = tess.GetPointNo();
    
    std::vector<std::size_t> neighbors;
    face_vec faces;
    
    Vector3D dummy_v;

    lambda_g.resize(N, 0.0);
    R2_g.resize(N, 0.0);

    if(flux_limiter_){
        std::fill(lambda_g.begin(), lambda_g.end(), 1.0);
        std::fill(R2_g.begin(), R2_g.end(), 1.0/3.0);
    }

    Vector3D grad_Eg(0.0, 0.0, 0.0);
    for(std::size_t i=0; i < N; ++i){
        double const volume = tess.GetVolume(i) * pow<3>(length_scale_);
        
        faces = tess.GetCellFaces(i);
        tess.GetNeighbors(i, neighbors);

        std::size_t const Nneighbors = neighbors.size();

        double const Eg_i = cells_cgs[i].Eg[group] * cells_cgs[i].density;
        Vector3D const r_i = tess.GetMeshPoint(i);

        grad_Eg.Set(0.0, 0.0, 0.0);
        for(std::size_t j=0; j<Nneighbors; ++j){
            std::size_t const neighbor_j = neighbors[j];

            Vector3D const r_ij = normalize(r_i - tess.GetMeshPoint(neighbor_j));

            double const A_ij = tess.GetArea(faces[j]) * pow<2>(length_scale_);

            double Eg_mid;
            if(!tess.IsPointOutsideBox(neighbor_j)){
                double const Eg_j = cells_cgs[neighbor_j].Eg[group] * cells_cgs[neighbor_j].density;
                Eg_mid = 0.5 * (Eg_i + Eg_j);
            } else {
                boundary_calculator.getOutsideValuesGroup(group, tess, i, neighbor_j, cells, Eg_i, Eg_mid, dummy_v);
                Eg_mid += Eg_i;
                Eg_mid *= 0.5;
            }

            grad_Eg += r_ij * (A_ij * Eg_mid);
        }

        grad_Eg *= -1.0 / volume;

        double const Dg = coefficient_calculator.CalcDiffusionCoefficientGroup(cells_cgs[i], group);
        
        double const lam = CG::CalcSingleFluxLimiter(grad_Eg, Dg, Eg_i);
        lambda_g[i] = lam;
        
        double const lam_R_g = lam * abs(grad_Eg) * Dg / (CG::speed_of_light * Eg_i);
        R2_g[i] = lam/3.0 + pow<2>(lam_R_g);
    }
}