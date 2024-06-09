#include "MultigroupDiffusion.hpp"
#include "planck_integral/planck_integral.hpp"

using boost::math::pow;

MultigroupDiffusion::MultigroupDiffusion(std::vector<double> const& energy_groups_center_, 
                                         std::vector<double> const& energy_groups_boundary_,
                                         MultigroupDiffusionCoefficientCalculator const& coefficient_calc,
                                         EquationOfState const& eos,
                                         std::vector<std::string> const zero_cells,
                                         bool const flux_limiter,
                                         bool const hydro_on,
                                         bool const compton_on):
                                                                coefficient_calculator(coefficient_calc),
                                                                energy_groups_center(energy_groups_center_),
                                                                energy_groups_boundary(energy_groups_boundary_),
                                                                current_group(0),
                                                                gray(false),
                                                                cells_temp(),
                                                                cells_cgs(),
                                                                extensives_temp(),
                                                                sigma_absorption_group(ENERGY_GROUPS_NUM, std::vector<double>()),
                                                                sigma_scattering_group(ENERGY_GROUPS_NUM, std::vector<double>()),
                                                                planck_integal_group(ENERGY_GROUPS_NUM, std::vector<double>()),
                                                                D_group(ENERGY_GROUPS_NUM, std::vector<double>()),
                                                                R2_group(ENERGY_GROUPS_NUM, std::vector<double>()),
                                                                cell_flux_limiter_group(ENERGY_GROUPS_NUM, std::vector<double>()),
                                                                new_Eg(),
                                                                new_Eg_full(),
                                                                new_Er(),
                                                                new_Er_full(),
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

MultigroupDiffusion::~MultigroupDiffusion() {}

bool MultigroupDiffusion::prestep(Tessellation3D const& tess) const {
    auto const N = tess.GetPointNo();

    sigma_absorption_group = std::vector<std::vector<double>>(ENERGY_GROUPS_NUM, std::vector<double>(N, 0.0));
    planck_integal_group = std::vector<std::vector<double>>(ENERGY_GROUPS_NUM, std::vector<double>(N, 0.0));
    D_group = std::vector<std::vector<double>>(ENERGY_GROUPS_NUM, std::vector<double>(N, 0.0));
    R2_group = std::vector<std::vector<double>>(ENERGY_GROUPS_NUM, std::vector<double>(N, 0.0));
    cell_flux_limiter_group = std::vector<std::vector<double>>(ENERGY_GROUPS_NUM, std::vector<double>(N, 0.0));
    
    new_Eg.resize(N, 0.0);
    new_Eg_full.resize(N, 0.0);
    new_Er.resize(N, 0.0);
    new_Er_full.resize(N, 0.0);

    return true;
}

bool MultigroupDiffusion::poststep() const {
    std::vector<ComputationalCell3D>().swap(cells_temp);
    std::vector<ComputationalCell3D>().swap(cells_cgs);
    std::vector<Conserved3D>().swap(extensives_temp);
    return true;
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
    calculate_group_diffusion_coefficients(tess, cells_cgs);

    std::size_t constexpr max_iter=1;
    for(std::size_t iter=1; iter <= max_iter; ++iter){    
        gray = false;
        for(std::size_t g=0; g<ENERGY_GROUPS_NUM; ++g){
            current_group=g;
            new_Eg = CG::BiCGSTAB(tolerance, total_iters, tess, cells, dt, *this, time, new_Er_full);

            PostCG(tess, extensives, dt, cells, new_Eg, new_Eg_full);
        }
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
        
        // build the initial guess
        x0[i] = cell_cgs.Eg[group];

        auto const volume_cgs = tess.GetVolume(i) * pow<3>(length_scale_);

        // build `b` vector, first term
        b[i] = volume_cgs * cell_cgs.Eg[group];

        // second term
        auto const bg = planck_integal_group[group][i];
        auto const Um = get_radiation_energy_density(cell_cgs.temperature);
        auto const cdtkgbg = cdt*sigma_absorption_group[group][i]*bg;
        b[i] += volume_cgs*cdtkgbg*Um;
    }

#ifdef RICH_MPI
    MPI_exchange_data2(tess, D_group[group], true);
#endif

    // Find maximum number of neighbors and allocate data
    // THIS SHOULD BE IN PRESTEP BUT BiCGSTAB CREATES A NEW MATRIX EVERY TIME IT IS CALLED. 
    // MAYBE MATRIX BUILDER SHOULD HOLD A MATRIX AS AN ATTRIBUTE
    
    // In Diffusion max_neighbors is used at the end it seems unessecary
    std::size_t max_neighbors = 0;
    for(std::size_t i=0; i < Nlocal; ++i){
        max_neighbors = std::max(max_neighbors, tess.GetNeighbors(i).size());
    }
    ++max_neighbors;

    A.clear(); 
    A.resize(Nlocal);
    A_indeces.clear();
    A_indeces.resize(Nlocal);
    
}

void MultigroupDiffusion::BuildMatrixGray(Tessellation3D const& tess, 
                                          mat& A, 
                                          size_t_mat& A_indeces, 
                                          std::vector<ComputationalCell3D> const& cells, 
                                          double const dt, 
                                          std::vector<double>& b, 
                                          std::vector<double>& x0, 
                                          double const current_time) const {

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

}

void MultigroupDiffusion::PostCGGray(Tessellation3D const& tess, 
                                     std::vector<Conserved3D>& extensives, 
                                     double const dt, 
                                     std::vector<ComputationalCell3D>& cells,
                                     std::vector<double>const& CG_result, 
                                     std::vector<double> const&  full_CG_result) const {

}

void MultigroupDiffusion::calculate_group_absorption_and_scattering_coefficients(Tessellation3D const& tess,
                                                                                 std::vector<ComputationalCell3D> const& cells) const {
    auto const N = tess.GetPointNo();
    for(std::size_t g=0; g < ENERGY_GROUPS_NUM; ++g){
        for(std::size_t i=0; i < N; ++i){
            auto const& cell = cells[i];

            sigma_absorption_group[g][i] = coefficient_calculator.CalcDiffusionCoefficientGroup(cell, g);

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

            planck_sum += bg;
        }

        if(planck_sum < (1. - 1e-4) and get_radiation_energy_density(cell.temperature) > 1e-3*cell.internal_energy*cell.density){
            throw UniversalError("bad groups! planckian not covered well!");
        }
    }
}

void MultigroupDiffusion::calculate_group_diffusion_coefficients(Tessellation3D const& tess,
                                                                 std::vector<ComputationalCell3D> const& cells) const {

    auto const N = tess.GetPointNo();

    for(std::size_t g=0; g<ENERGY_GROUPS_NUM; ++g){
        for(std::size_t i=0; i<N; ++i){
            D_group[g][i] = coefficient_calculator.CalcDiffusionCoefficientGroup(cells[i], g);

            if(D_group[g][i] < 0.0){
                throw UniversalError("negative group diffusion coefficient");
            }
        }
    }
}