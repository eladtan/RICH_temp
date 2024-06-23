#include "Diffusion.hpp" // for CalcSingleFluxLimiter and FleckFactor
#include "MultigroupDiffusion.hpp"
#include "planck_integral/planck_integral.hpp"

using boost::math::pow;

MultigroupDiffusion::MultigroupDiffusion(std::vector<double> const& energy_groups_center_, 
                                         std::vector<double> const& energy_groups_boundary_,
                                         MultigroupDiffusionCoefficientCalculator const& coefficient_calc,
                                         MultigroupDiffusionBoundaryCalculator const& boundary_calc,
                                         EquationOfState const& eos,
                                         std::vector<std::string> const zero_cells,
                                         bool const flux_limiter,
                                         bool const hydro_on,
                                         bool const compton_on):
                                                                energy_groups_center(energy_groups_center_),
                                                                energy_groups_boundary(energy_groups_boundary_),
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
                                                                R2_group(ENERGY_GROUPS_NUM, std::vector<double>()),
                                                                cell_flux_limiter_group(ENERGY_GROUPS_NUM, std::vector<double>()),
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
                                                                max_abs_grad_E(),
                                                                max_neighbor_abs_grad_E(),
                                                                grad(),
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
}

bool MultigroupDiffusion::prestep(Tessellation3D const& tess,
                                  std::vector<ComputationalCell3D> const& cells) const {
    auto const N = tess.GetPointNo();

    sigma_absorption_group = std::vector<std::vector<double>>(ENERGY_GROUPS_NUM, std::vector<double>(N, 0.0));
    sigma_scattering_group = std::vector<std::vector<double>>(ENERGY_GROUPS_NUM, std::vector<double>(N, 0.0));
    planck_integal_group = std::vector<std::vector<double>>(ENERGY_GROUPS_NUM, std::vector<double>(N, 0.0));
    R2_group = std::vector<std::vector<double>>(ENERGY_GROUPS_NUM, std::vector<double>(N, 0.0));
    cell_flux_limiter_group = std::vector<std::vector<double>>(ENERGY_GROUPS_NUM, std::vector<double>(N, 0.0));
    
    new_Eg.resize(N, 0.0);
    new_Eg_full.resize(N, 0.0);
    
    sigma_absorption_planck.resize(N, 0.0);
    sigma_absorption_average.resize(N, 0.0);
    sigma_scattering_gray.resize(N, 0.0);
    fleck_factor.resize(N, 0.0);

    new_Er.resize(N, 0.0);
    new_Er_full.resize(N, 0.0);

    old_Er.resize(N, 0.0);

    max_abs_grad_E.resize(N, 0.0);
    max_neighbor_abs_grad_E.resize(N, 0.0);

    for(std::size_t i=0; i < N; ++i){
        old_Er[i] = cells[i].Erad * cells[i].density;
    }

    for(std::size_t g=0; g<ENERGY_GROUPS_NUM; ++g){
        old_Eg[g].resize(N, 0.0);
        for(std::size_t i=0; i < N; ++i){
            old_Eg[g][i] = cells[i].Eg[g] * cells[i].density;
        }
    }

    auto const Nfaces = tess.GetTotalFacesNumber();
    grad.resize(Nfaces);

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
#ifdef RICH_MPI
	MPI_Allreduce(MPI_IN_PLACE, &max_Er, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
#endif	

    auto const N = tess.GetPointNo();            
    size_t const Nzero = zero_cells_.size();
	std::vector<size_t> zero_indeces;
	for(size_t i = 0; i < Nzero; ++i)
		zero_indeces.push_back(binary_index_find(ComputationalCell3D::stickerNames, zero_cells_[i]));
	double max_diff = std::numeric_limits<double>::min() * 100;
	int max_loc = 0;
	for(size_t i = 0; i < N; ++i)
	{
		bool to_calc = true;
		for(size_t j = 0; j < Nzero; ++j)
			if(cells[i].stickers[zero_indeces[j]])
				to_calc = false;
		if(not to_calc)
			continue;
		double const equlibrium_factor = std::abs(cells[i].temperature - std::pow(new_Er[i] / CG::radiation_constant, 0.25)) < 0.02 * cells[i].temperature ? 0.05 : 1;
		double diff = equlibrium_factor * std::abs(cells[i].Erad * cells[i].density - old_Er[i]) / (cells[i].Erad * cells[i].density + 0.02 * max_Er);
		if(fleck_factor[i] < 0.5)
			diff *= 0.1;
		if(diff > max_diff)
		{
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
		std::cout<<"Radiation time step ID "<<cells[max_loc].ID<<" old Er "<<old_Er[max_loc]<<" new Er "<<cells[max_loc].Erad * cells[max_loc].density<<
		" diff "<<max_diff<<" Tgas "<<cells[max_loc].temperature<<" Trad "<<std::pow(new_Er[max_loc] / CG::radiation_constant, 0.25)<<" max_Er "<<max_Er<<" rank "<<rank<<" density "<<cells[max_loc].density<<
		" width "<<tess.GetWidth(max_loc)<<" Tgas_old "<<cells[max_loc].temperature<<std::endl;
		PrintDebugData(max_loc);
	}

    return dt * 0.15 / max_diff;
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

#ifdef RICH_MPI
	ComputationalCell3D cdummy;
	MPI_exchange_data(tess, cells_cgs, true, &cdummy);	
#endif

    calculate_group_absorption_and_scattering_coefficients(tess, cells_cgs);
    calculate_planck_integrals(tess, cells_cgs);

    std::size_t constexpr max_iter=1;
    for(std::size_t iter=1; iter <= max_iter; ++iter){    
        gray = false;
        for(std::size_t g=0; g<ENERGY_GROUPS_NUM; ++g){
            current_group=g;
            new_Eg = CG::BiCGSTAB(tolerance, total_iters, tess, cells, dt, *this, time, new_Eg_full);

            PostCG(tess, extensives, dt, cells, new_Eg, new_Eg_full);
        }
        
        calculate_gray_absorption_and_scattering_coefficients(tess, cells);

        gray = true;
        new_Er = CG::BiCGSTAB(tolerance, total_iters, tess, cells, dt, *this, time, new_Er_full);

        PostCG(tess, extensives, dt, cells, new_Er, new_Er_full);
    }

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
    if(gray){
        BuildMatrixGray(tess, A, A_indeces, cells, dt, b, x0, current_time);
    } else {
        assert(current_group < ENERGY_GROUPS_NUM);
        BuildMatrixGroup(current_group, tess, A, A_indeces, cells, dt, b, x0, current_time);
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
        auto const cell_cgs = cells_cgs[i];
        
        double const Eg_i = cell_cgs.Eg[group]*cell_cgs.density;
        // build the initial guess
        x0[i] = Eg_i;

        auto const volume_cgs = tess.GetVolume(i) * pow<3>(length_scale_);

        // build `b` vector, first term
        b[i] = volume_cgs * Eg_i;

        // second term
        auto const bg = planck_integal_group[group][i];
        auto const Um = get_radiation_energy_density(cell_cgs.temperature);
        auto const cdtkgbg = cdt*sigma_absorption_group[group][i]*bg;
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
        double const T = cells_cgs[i].temperature;

        double cdtkg = cdt * sigma_absorption_group[group][i];

        A[i].push_back(volume*(1.0 + cdtkg));

        if(A[i][0] < 0){
            std::cout << "Negative A[i][i] in matrix build" << std::endl;
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
            double const Eg_i = cells_cgs[i].Eg[group] * cells_cgs[i].density;

            for(std::size_t j=0; j < Nneighbors; ++j){
                std::size_t const neighbor_j = neighbors[j];
                if(neighbor_j < Nlocal || !tess.IsPointOutsideBox(neighbor_j)){
                    double const Eg_j = cells_cgs[neighbor_j].Eg[group] * cells_cgs[neighbor_j].density;
                    auto const abs_dE = std::abs(Eg_i - Eg_j);
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
                    double const max_T = std::max(T_i, T_j);

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

                        // double const gradE_magnitude = std::max(std::abs(fastabs(gradient)*dEg), std::numeric_limits<double>::min()*1e40);
                        // double grad_factor = 1.0;

                        max_neighbor_abs_grad_E[i] = std::max(max_neighbor_abs_grad_E[i], max_abs_grad_E[neighbor_j]);

                        // grad_factor = std::max(0.15 * (max_abs_grad_E[i] + max_abs_grad_E[neighbor_j])/gradE_magnitude, 1.0);

                        // lambda = CG::CalcSingleFluxLimiter(gradient*dEg*grad_factor, D_ij, 0.5*(Eg_i + Eg_j));
                        
                        // TODO: add grad_factor to both here and to gray
                        lambda = CG::CalcSingleFluxLimiter(gradient*dEg, D_ij, 0.5*(Eg_i + Eg_j));
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

                boundary_calculator.getOutSideValuesGroup(group, tess, cells_cgs, i, neighbor_j, new_Eg, Eg_j, dummy_v);
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
        b[i] = volume_cgs * Er_i;

        // fleck factor
        double const sigma_planck = sigma_absorption_planck[i];
        double const T = cell_cgs.temperature;
        double cv = eos_.dT2cv(cells[i].density, T, cells[i].tracers, ComputationalCell3D::tracerNames);
        
        // TODO: What is energy ratio (see Diffusion.cpp same line)
        cv *= mass_scale_ / (pow<2>(time_scale_)*length_scale_);
        double const cv_bar = cv / get_radiation_cv(T);
        double const f = CG::FleckFactor(dt_cgs, 1.0/cv_bar, sigma_planck);

        if(f < 0){
            throw UniversalError("Negative fleck factor");
        }
        fleck_factor[i] = f;
        
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
        double const T = cells_cgs[i].temperature;

        double cdtkrf = cdt*sigma_absorption_average[i]*fleck_factor[i];

        A[i].push_back(volume*(1.0 + cdtkrf));

        if(A[i][0] < 0){
            std::cout << "Negative A[i][i] in matrix build" << std::endl;
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
                std::size_t neighbor_j = neighbors[j];
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
                    double const max_T = std::max(T_i, T_j);

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
                            lambda_g = CG::CalcSingleFluxLimiter(gradient*dEg, Dg_ij, 0.5*(Eg_old_i + Eg_old_j));
                        }

                        double const lambda_gD = lambda_g * Dg_ij;
                        
                        // cell_cgs holds the old Eg but after the group step we need to use cells.
                        double const Eg_i = cells[i].Eg[g] * cells[i].density * pow<2>(length_scale_) / pow<2>(time_scale_);
                        double const Eg_j = cells[neighbor_j].Eg[g] * cells[neighbor_j].density * pow<2>(length_scale_) / pow<2>(time_scale_);

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

                boundary_calculator.getOutsideValuesGray(tess, i, neighbor_j, cells_cgs, new_Er, Er_j, dummy_v);
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
        cells[i].Eg[group] = std::max(full_CG_result[i], std::numeric_limits<double>::min()*1e100) * pow<2>(time_scale_) / (cells[i].density * mass_scale_ / length_scale_);
    }
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

        extensives[i].Erad = CG_result[i] * volume * pow<2>(time_scale_) / (pow<2>(length_scale_) * mass_scale_);

        double const T = cells[i].temperature;
        double const kp = sigma_absorption_planck[i];
        double const kr = sigma_absorption_average[i];
        double const Um = get_radiation_energy_density(T);

        double dE = volume * fleck_factor[i] * cdt * (kr*full_CG_result[i] - kp*Um);

        dE *= pow<2>(time_scale_) / (pow<2>(length_scale_) * mass_scale_);

        extensives[i].energy += dE;
        extensives[i].internal_energy += dE;

        // other terms

        cells[i].Erad = extensives[i].Erad / extensives[i].mass;
        cells[i].internal_energy = extensives[i].internal_energy / extensives[i].mass;

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

    if(rank == 0){
        std::cout << std::setprecision(14) << "Einit = " << Einit << ", Efinal = " << Efinal << ", rel_error = " << std::abs((Einit - Efinal))/Efinal << std::endl;
    }
}

void MultigroupDiffusion::calculate_group_absorption_and_scattering_coefficients(Tessellation3D const& tess,
                                                                                 std::vector<ComputationalCell3D> const& cells) const {
    auto const N = tess.GetPointNo();
    for(std::size_t g=0; g < ENERGY_GROUPS_NUM; ++g){
        for(std::size_t i=0; i < N; ++i){
            auto const& cell = cells[i];

            sigma_absorption_group[g][i] = coefficient_calculator.CalcAbsorptionCoefficientGroup(cell, g);

            if(sigma_absorption_group[g][i] < 0.){
                throw UniversalError("negative absorption coefficient");
            }

            sigma_scattering_group[g][i] = coefficient_calculator.CalcScatteringCoefficientGroup(cell, g);

            if(sigma_scattering_group[g][i] < 0.){
                throw UniversalError("negative scattering coefficient");
            }
        }
    }
}

void MultigroupDiffusion::calculate_planck_integrals(Tessellation3D const& tess,
                                                     std::vector<ComputationalCell3D> const& cells) const {
    
    auto const N = tess.GetPointNo();

    for(std::size_t i=0; i<N; ++i){
        auto const& cell = cells[i];
        double const kT = CG::boltzmann_constant * cell.temperature;
        double planck_sum = 0.0;
        for(std::size_t g=0; g<ENERGY_GROUPS_NUM; ++g){

            double const a = energy_groups_boundary[g] / kT;
            double const b = energy_groups_boundary[g+1] / kT;

            double const bg = planck_integral(a, b);

            planck_integal_group[g][i] = bg;
            planck_sum += bg;
        }

        if(planck_sum < (1. - 1e-4) and 
           get_radiation_energy_density(cell.temperature) > 1e-3*cell.internal_energy*cell.density){
            throw UniversalError("bad groups! planckian not covered well!");
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
            double const sigma = sigma_absorption_group[g][i];
            double const bg = planck_integal_group[g][i];
            // the change of units is not really important since we are averaging with Ug as weights i.e. the units cancel. But it is for consistency.
            double const Ug = cell.Eg[g] * cell.density * pow<2>(length_scale_) / pow<2>(time_scale_);
            
            sigma_absorption_planck[i] += sigma * bg;
            sigma_absorption_average[i] += sigma * Ug;
            sigma_scattering_gray[i] += sigma * Ug;

            sum_U += Ug;
        }

        if(sum_U > std::numeric_limits<double>::min()*1e40) {
            sigma_absorption_average[i] /= sum_U;
            sigma_scattering_gray[i] /= sum_U;
        }
    }
}