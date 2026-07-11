#include "Diffusion.hpp" // for CalcSingleFluxLimiter and FleckFactor
#include "MultigroupDiffusion.hpp"
#include "misc/memory_debug.hpp"
#include "misc/memory_profile.hpp"
// TODO: make a units namespace used by all the program 
#include "CMMC/src/units/units.hpp"
#include "CMMC/src/planck_integral/planck_integral.hpp"

using boost::math::pow;

void fill_zero(std::vector<double>& vec) {
    std::fill(vec.begin(), vec.end(), 0.0);
}

void fill_zero(std::vector<std::vector<double>>& mat) {
    for (std::vector<double>& row : mat) {
        std::fill(row.begin(), row.end(), 0.0);
    }
}

void resize_group_matrix(std::vector<std::vector<double>>& mat, std::size_t cells) {
    mat.resize(ENERGY_GROUPS_NUM);
    for (std::size_t g = 0; g < ENERGY_GROUPS_NUM; ++g) {
        mat[g].resize(cells);
        std::fill(mat[g].begin(), mat[g].end(), 0.0);
    }
}

std::vector<double> compton_temperatures() {
    std::vector<double> tmp_grid = linspace(-2, 4, 128);

    for (size_t i = 0; i < tmp_grid.size(); ++i) {
        tmp_grid[i] = std::pow(10.0, tmp_grid[i]);
    }

    tmp_grid.insert(tmp_grid.begin(), 0.005);
    tmp_grid.insert(tmp_grid.begin(), 0.001);
    tmp_grid.insert(tmp_grid.begin(), 0.0001);
    // tmp_grid = {1e-2, 0.1, 0.2, 0.3, 0.8, 1.5, 3.0, 4.0, 5.0, 7.5, 10.0, 13.0, 18.0, 20.0, 21.};
    for (auto& temp : tmp_grid) {
        temp *= units::kev_kelvin;
    }

    return tmp_grid;
}

std::vector<double> get_energy_groups_width(std::vector<double> const& energy_groups_boundary) {
    std::vector<double> energy_groups_width(energy_groups_boundary.size()-1, std::numeric_limits<double>::signaling_NaN());

    for (std::size_t g=0; g<energy_groups_boundary.size()-1; ++g) {
        energy_groups_width[g] = energy_groups_boundary[g+1] - energy_groups_boundary[g];
    }

    return energy_groups_width;
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
                                         double const minimum_temperature,
                                         bool const protections_on,
                                         bool const cooling_time_limiter_on) :
    RadiationDriver(eos,
        zero_cells,
        flux_limiter,
        hydro_on,
        compton_on),
    coefficient_calculator(coefficient_calc),
    boundary_calculator(boundary_calc),
    energy_groups_center(energy_groups_center_),
    energy_groups_boundary(energy_groups_boundary_),
    energy_groups_width(get_energy_groups_width(energy_groups_boundary)),
    cells_cgs(),
    sigma_absorption_group(ENERGY_GROUPS_NUM, std::vector<double>()),
    sigma_scattering_group(ENERGY_GROUPS_NUM, std::vector<double>()),
    planck_integal_group(ENERGY_GROUPS_NUM, std::vector<double>()),
    sigma_absorption_planck(),
    fleck_factor(),
    new_Eg(),
    new_Eg_full(),
    old_Eg(ENERGY_GROUPS_NUM, std::vector<double>()),
    old_Er(),
    old_Tm(),
    grad(),
    doppler_on_(doppler_on),
    minimum_temperature_(minimum_temperature),
    displayed_warning_(false),
    compton_matrix_gen(
        energy_groups_center_,
        energy_groups_boundary_,
        compton_on ? 100000 : 10,
        true, // num of samples
        1),
    tau(ENERGY_GROUPS_NUM, std::vector<double>(ENERGY_GROUPS_NUM, 0.0)),
    dtau_dUm(ENERGY_GROUPS_NUM, std::vector<double>(ENERGY_GROUPS_NUM, 0.0)),
    S(ENERGY_GROUPS_NUM, std::vector<double>(ENERGY_GROUPS_NUM, 0.0)),
    dSdUm(ENERGY_GROUPS_NUM, std::vector<double>(ENERGY_GROUPS_NUM, 0.0)),
    n(ENERGY_GROUPS_NUM, 0.0),
    cell_id_of_compton_matrices(std::numeric_limits<std::size_t>::max()),
    Gammas(),
    use_n_zero(),
    protections_on_(protections_on),
    cooling_time_limiter_on_(cooling_time_limiter_on) {

    if (energy_groups_center.size() != ENERGY_GROUPS_NUM) {
        std::cout << "bad energy_groups_center.size()" << std::endl;
        exit(1);
    }

    if (energy_groups_boundary.size() != ENERGY_GROUPS_NUM + 1) {
        std::cout << "bad energy_groups_boundary.size()" << std::endl;
        exit(1);
    }

    for (std::size_t g=0; g < ENERGY_GROUPS_NUM; ++g) {
        if (energy_groups_boundary[g] >= energy_groups_boundary[g+1]) {
            std::cout << "bad energy_groups_boundary" << std::endl;
            exit(1);
        }

        if (energy_groups_boundary[g] >= energy_groups_center[g] or energy_groups_boundary[g+1] <= energy_groups_center[g]) {
            std::cout << "bad energy_groups_boundary and energy_groups_center" << std::endl;
            exit(1);
        }
    }
    if(compton_on)
        compton_matrix_gen.set_tables(compton_temperatures());
}

bool MultigroupDiffusion::prestep(Tessellation3D const& tess,
                                  std::vector<ComputationalCell3D> const& cells) const {
    MEMORY_PROFILE_SCOPE("multigroup diffusion prestep");
    auto const N = tess.GetPointNo();

    resize_group_matrix(sigma_absorption_group, N);
    resize_group_matrix(sigma_scattering_group, N);
    resize_group_matrix(planck_integal_group, N);

    new_Eg.resize(N, 0.0);
    new_Eg_full.resize(N, 0.0);

    sigma_absorption_planck.resize(N, 0.0);
    fleck_factor.resize(N, 0.0);

    old_Er.resize(N, 0.0);
    old_Tm.resize(N, 0.0);

    for (std::size_t i=0; i < N; ++i) {
        old_Er[i] = cells[i].Erad * cells[i].density;
        old_Tm[i] = cells[i].temperature;
    }

    old_Eg.resize(N);
    for (std::size_t i=0; i < N; ++i) {
        old_Eg[i].resize(ENERGY_GROUPS_NUM, 0.0);

        for (std::size_t g=0; g<ENERGY_GROUPS_NUM; ++g) {
            old_Eg[i][g] = cells[i].Eg[g] * cells[i].density;
        }
    }

    auto const Nfaces = tess.GetTotalFacesNumber();
    grad.resize(Nfaces);

    // temporary vectors
    std::vector<std::size_t> neighbors;
    face_vec faces;

    // create gradient per face
    for (std::size_t i=0; i < N; ++i) {
        tess.GetNeighbors(i, neighbors);
        faces = tess.GetCellFaces(i);

        Vector3D CM_i = tess.GetCellCM(i);

        auto const Nneighbors = neighbors.size();
        for (std::size_t j=0; j < Nneighbors; ++j) {
            std::size_t const neighbor_j = neighbors[j];

            if (!tess.IsPointOutsideBox(neighbor_j)) {
                if (i < neighbor_j) {
                    Vector3D const CM_ij = CM_i - tess.GetCellCM(neighbor_j);
                    grad[faces[j]] = CM_ij * (1.0 / (length_scale_*ScalarProd(CM_ij, CM_ij)));
                }
            }
        }
    }

    Gammas.resize(N, 0.0);
    use_n_zero.resize(N, false);
    compton_limiter_scale_.assign(N, 1.0);

    return true;
}

bool MultigroupDiffusion::poststep() const {
    cells_cgs.clear();

    return true;
}

double MultigroupDiffusion::calculate_dt(double const dt,
                                         Tessellation3D& tess,
                                         std::vector<ComputationalCell3D>& cells) const {
    int rank = 0;
#ifdef RICH_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif

    double max_Er = old_Er.size() > 0 ? *std::max_element(old_Er.begin(), old_Er.end()) : std::numeric_limits<double>::min();
    double max_rhoT = 0;
    auto const N = tess.GetPointNo();

    for (size_t i=0; i < N; ++i) max_rhoT = std::max(max_rhoT, cells[i].density * cells[i].temperature);

#ifdef RICH_MPI
    MPI_Allreduce(MPI_IN_PLACE, &max_Er, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &max_rhoT, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
#endif	

    size_t const Nzero = zero_cells_.size();
    std::vector<size_t> zero_indeces;
    for (size_t i = 0; i < Nzero; ++i) {
        zero_indeces.push_back(binary_index_find(ComputationalCell3D::stickerNames, zero_cells_[i]));
    }

    double max_diff = std::numeric_limits<double>::min() * 100;
    int max_which = 0;
    int max_loc = 0;
    double equlibrium_factor_final = 0, final_Erad_eq = 0;
    for (size_t i = 0; i < N; ++i)
    {
        int which_one = 0;
        bool to_calc = true;
        for (size_t j = 0; j < Nzero; ++j) {
            if (cells[i].stickers[zero_indeces[j]]) {
                to_calc = false;
            }
        }

        if (not to_calc)
            continue;

        double const new_Er_cell = cells[i].Erad * cells[i].density;
        double const equlibrium_factor = std::abs(cells[i].temperature - std::pow(new_Er_cell / CG::radiation_constant, 0.25)) < 0.05 * cells[i].temperature ? 0.5 : 1;
        double diff = equlibrium_factor * std::abs(new_Er_cell - old_Er[i]) / (new_Er_cell+0.02*max_Er);
        bool const Erad_equib = std::abs(new_Er_cell - old_Er[i]) < 0.075 * old_Er[i];
        double temp_diff = cells[i].density * equlibrium_factor* std::abs(cells[i].temperature - old_Tm[i]) / (cells[i].density * cells[i].temperature+1e-3*max_rhoT);

        if (Erad_equib) temp_diff *= 0.25;

        if (fleck_factor[i] < 0.8) temp_diff *= std::pow(0.2 + fleck_factor[i], 3.0);

        if (temp_diff > diff) {
            which_one = 1;
            diff = temp_diff;
        }

        for (std::size_t g=0; g<ENERGY_GROUPS_NUM; ++g) {
            double temp = 0.2 * (7 * equlibrium_factor / 6 - (1.0 / 6.0))* std::abs(cells[i].Eg[g]*cells[i].density - old_Eg[i][g]) / (cells[i].Eg[g]*cells[i].density + 0.01/ENERGY_GROUPS_NUM*max_Er + cells[i].Erad * cells[i].density / ENERGY_GROUPS_NUM);

            if (Erad_equib) temp *= 0.25;

            if (temp > diff) {
                which_one = 2 + g;
                diff = temp;
            }
        }

        if (diff > max_diff) {
            max_which = which_one;
            max_diff = diff;
            max_loc = i;
            equlibrium_factor_final = equlibrium_factor;
            final_Erad_eq = Erad_equib;
        }
    }

    struct {
        double val;
        int mpi_id;
    } max_data;

    max_data.mpi_id = rank;
    max_data.val = max_diff;

#ifdef RICH_MPI
    MPI_Allreduce(MPI_IN_PLACE, &max_data, 1, MPI_DOUBLE_INT, MPI_MAXLOC, MPI_COMM_WORLD);
    max_diff = max_data.val;
    MPI_exchange_data(tess, cells, true);
#endif
    if (rank == max_data.mpi_id) {
        // max_loc /= ENERGY_GROUPS_NUM;
        std::cout<<"Radiation time step ID "<<cells[max_loc].ID<<" old Er "<<old_Er[max_loc]<<" new Er "<<cells[max_loc].Erad * cells[max_loc].density<<
            " diff "<<max_diff<<" Tgas "<<cells[max_loc].temperature<<" Trad "<<std::pow(cells[max_loc].density * cells[max_loc].Erad * mass_scale_ / (length_scale_ * pow<2>(time_scale_) * CG::radiation_constant), 0.25)<<" max_Er "<<max_Er<<" max_rhoT "<<max_rhoT<<" rank "<<rank<<" density "<<cells[max_loc].density<<
            " width "<<tess.GetWidth(max_loc)<<" Tgas_old "<<old_Tm[max_loc]<<" loc="<<tess.GetMeshPoint(max_loc)<<std::endl;
        std::cout<<"kp="<<sigma_absorption_planck[max_loc]<<" fleck factor "<<fleck_factor[max_loc]<<" which one "<<max_which<<" equlibrium_factor "<<equlibrium_factor_final<<" final_Erad_eq "<<final_Erad_eq<<std::endl;

        if (max_which >= 2) std::cout << "Group number "<<max_which - 2<<" New_Eg="<<cells[max_loc].Eg[max_which - 2]*cells[max_loc].density<<" old_Eg="<<old_Eg[max_loc][max_which - 2]<<std::endl;
#ifdef DEBUG
        for (size_t j = 0; j < ENERGY_GROUPS_NUM; ++j) {
            std::cout<<"Eg["<<j<<"]="<<cells[max_loc].Eg[j]*cells[max_loc].density*mass_scale_ / (length_scale_ * pow<2>(time_scale_)) <<" old Eg["<<j<<"]="<<old_Eg[max_loc][j]*mass_scale_ / (length_scale_ * pow<2>(time_scale_))<<" energy(keV) "<<energy_groups_center[j] / units::kev<<" bg="<<
            planck_integral::planck_energy_density_group_integral(energy_groups_boundary[j], energy_groups_boundary[j+1], cells[max_loc].temperature)<<
            " bg_old="<<
            planck_integral::planck_energy_density_group_integral(energy_groups_boundary[j], energy_groups_boundary[j+1], old_Tm[max_loc])<< ", sigma[g]=" << sigma_absorption_group[max_loc][j] << ", cdt*sigma_g=" <<sigma_absorption_group[max_loc][j]*CG::speed_of_light*dt*time_scale_<< std::endl;
        }
#endif
    }

    return std::min(dt * 0.15 / max_diff, dt*1.4);
}

bool MultigroupDiffusion::step(double const tolerance,
                               int& total_iters,
                               Tessellation3D const& tess,
                               std::vector<ComputationalCell3D>& cells,
                               std::vector<Conserved3D>& extensives,
                               double const dt,
                               double const time) const {

    auto const N = tess.GetPointNo();

    cells_cgs = cells;
    for (std::size_t i=0; i<N; ++i) {
        cells_cgs[i].density *= mass_scale_ / pow<3>(length_scale_);
        cells_cgs[i].internal_energy *= pow<2>(length_scale_) / pow<2>(time_scale_);
        cells_cgs[i].Erad *= pow<2>(length_scale_) / pow<2>(time_scale_);
        cells_cgs[i].velocity *= length_scale_ / time_scale_;
        for (std::size_t g=0; g<ENERGY_GROUPS_NUM; ++g) {
            cells_cgs[i].Eg[g] *= pow<2>(length_scale_) / pow<2>(time_scale_);
        }
    }

    int rank = 0;
#ifdef RICH_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_exchange_data(tess, cells_cgs, true);
#endif

    calculate_group_absorption_and_scattering_coefficients(tess, cells_cgs, dt * time_scale_);
    calculate_planck_integrals(tess, cells_cgs);
    calculate_planck_absorption_coefficient(tess, cells);
    calculate_fleck_factor(tess, cells, dt * time_scale_);

    std::size_t tot_iters = 0;
    bool good_end = false;
    new_Eg = CG::BiCGSTAB(tolerance, total_iters, tess, cells, dt, *this, time, new_Eg_full, good_end, cg_workspace_);
    MEMORY_DEBUG_PRINT("multigroup: after BiCGSTAB");
    if (not good_end)
        return false;
    tot_iters += total_iters;

    if (rank == 0) std::cout << "Total iterations: " << tot_iters << std::endl;

    PostCG(tess, extensives, dt, cells, new_Eg, new_Eg_full);
    MEMORY_DEBUG_PRINT("multigroup: after PostCG");

#ifdef RICH_MPI
    MPI_exchange_data(tess, cells, true);
#endif

    return true;
}

double  MultigroupDiffusion::get_doppler_slope(ComputationalCell3D const& cell, size_t const g, bool const expansion) const
{
    MEMORY_PROFILE_SCOPE("multigroup diffusion step");
    if (g == 0 or (g + 1) == ENERGY_GROUPS_NUM) {
        return 0.0;
    }

    double const dw_left = expansion ? energy_groups_width[g] : energy_groups_width[g - 1];
    double const dw_right = expansion ? energy_groups_width[g + 1] : energy_groups_width[g];

    double const slope_left = (cell.Eg[g] * cell.density / energy_groups_width[g] - cell.Eg[g - 1] * cell.density / energy_groups_width[g - 1]) / dw_left;
    double const slope_right = (cell.Eg[g + 1] * cell.density / energy_groups_width[g + 1] - cell.Eg[g] * cell.density / energy_groups_width[g]) / dw_right;

    double const r = slope_left / (slope_right + std::max({ slope_right, slope_left, std::numeric_limits<double>::min() * 1e50 })*1e-16);

    double const slope = std::max(std::max(0.0, std::min(2 * r, 1.0)), std::min(r, 2.0));

    return slope;
}

void MultigroupDiffusion::BuildMatrix(Tessellation3D const& tess,
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
    for (std::size_t i=0; i < Nlocal; ++i) {
        auto const& cell_cgs = cells_cgs[i];
        auto const volume_cgs = tess.GetVolume(i) * pow<3>(length_scale_);
        double const f = fleck_factor[i];

        auto const Um = get_radiation_energy_density(cell_cgs.temperature);
        double aT4_np1 = f * Um;
        for (size_t group=0; group<ENERGY_GROUPS_NUM; ++group) {
            aT4_np1 += (1 - f) * sigma_absorption_group[i][group] * cell_cgs.Eg[group] * cell_cgs.density / sigma_absorption_planck[i];
        }

        for (size_t group=0; group<ENERGY_GROUPS_NUM; ++group) {
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
    for (std::size_t i=0; i < Nlocal; ++i) {
        bool const do_compton = compton_on_ && (sigma_absorption_planck[i] * dt_cgs * CG::speed_of_light < compton_optical_depth_turn_off);
        if (do_compton) {
            generate_S_and_dSdUm_matrices(cells[i], i, dt_cgs, !use_n_zero[i]);
        }

        double const f = fleck_factor[i];
        double const volume = tess.GetVolume(i) * pow<3>(length_scale_);
        for (size_t group=0; group<ENERGY_GROUPS_NUM; ++group) {
            auto const bg = planck_integal_group[i][group];
            double const Gamma_1 = 1.0 / Gammas[i];

            double const cdtkg = cdt * sigma_absorption_group[i][group];
            double const implicit_self_contribution = -(1 - f) * cdtkg * Gamma_1 * bg * sigma_absorption_group[i][group];

            double implicit_self_compton_contribution = 0.0;

            if (do_compton) {
                double const implicit_compton_contribution_to_b = get_implicit_compton_contribution_to_b(tess, cells[i], i, group, dt_cgs);
                b[i * ENERGY_GROUPS_NUM + group] += implicit_compton_contribution_to_b;

                implicit_self_compton_contribution = get_implicit_compton_contribution(tess, cells[i], i, group, group, dt_cgs);
            }

            A[i * ENERGY_GROUPS_NUM + group].push_back(volume*(1.0 + cdtkg + implicit_self_contribution) + implicit_self_compton_contribution);
            A_indeces[i * ENERGY_GROUPS_NUM + group].push_back(i * ENERGY_GROUPS_NUM + group);

            for (size_t group_j=0; group_j<ENERGY_GROUPS_NUM; ++group_j) {
                if (group_j!= group) {
                    double const implicit_conribution_group_j = -volume*bg * (1 - f) * sigma_absorption_group[i][group_j] * sigma_absorption_group[i][group] * cdt * Gamma_1;

                    double implicit_compton_contribution_group_j = 0.0;
                    if (do_compton) {
                        implicit_compton_contribution_group_j = get_implicit_compton_contribution(tess, cells[i], i, group, group_j, dt_cgs);
                    }

                    A[i * ENERGY_GROUPS_NUM + group].push_back(implicit_conribution_group_j + implicit_compton_contribution_group_j);

                    A_indeces[i * ENERGY_GROUPS_NUM + group].push_back(i * ENERGY_GROUPS_NUM + group_j);
                }
            }
            double A0 = A[i * ENERGY_GROUPS_NUM + group][0];
            if (A0 < 0) {
                std::cout <<"Negative A[i][i] in matrix build cell ID " <<cells[i].ID<<" group "<<group<<std::endl;
                std::cout<<cells[i]<<std::endl;
            }
        }
    }

    // calculate R2
    std::vector<std::size_t> neighbors;
    face_vec faces;
    Vector3D dummy_v;
    std::vector<std::vector<double>> R2;
    std::vector<Vector3D> grad_temp_array(ENERGY_GROUPS_NUM);
    if (flux_limiter_) {
        R2.resize(Nlocal);
        for (std::size_t i=0; i < Nlocal; ++i) {

            R2[i].resize(ENERGY_GROUPS_NUM, 0);
            tess.GetNeighbors(i, neighbors);
            faces = tess.GetCellFaces(i);
            double const cell_width = std::max(tess.GetWidth(i) * length_scale_, 1e-200);

            auto const Nneighbors = neighbors.size();
            double Er_i = cells_cgs[i].Erad * cells_cgs[i].density;
            Vector3D const r_i = tess.GetMeshPoint(i);
            for (std::size_t j=0; j < Nneighbors; ++j) {
                std::size_t const neighbor_j = neighbors[j];
                Vector3D const r_ij = normalize(r_i - tess.GetMeshPoint(neighbor_j));
                if (neighbor_j < Nlocal || !tess.IsPointOutsideBox(neighbor_j)) {
                    double const Er_j = cells_cgs[neighbor_j].Erad * cells_cgs[neighbor_j].density;
                    auto const abs_dE = std::abs(Er_i - Er_j);
                    auto const abs_grad_E = abs_dE * fastabs(grad[faces[j]]);

                }
                for (size_t g = 0; g < ENERGY_GROUPS_NUM; ++g) {
                    if (j == 0) grad_temp_array[g].Set(0, 0, 0);

                    Er_i = cells_cgs[i].Eg[g] * cells_cgs[i].density;
                    if (neighbor_j < Nlocal || !tess.IsPointOutsideBox(neighbor_j)) {
                        double const Er_j = cells_cgs[neighbor_j].Eg[g] * cells_cgs[neighbor_j].density;
                        grad_temp_array[g] += (tess.GetArea(faces[j]) * pow<2>(length_scale_) * 0.5 * (Er_j + Er_i)) * r_ij;
                    } else {
                        grad_temp_array[g] += (tess.GetArea(faces[j]) * pow<2>(length_scale_) * 0.5 * (Er_i + Er_i)) * r_ij;
                    }
                }
            }
            for (size_t g = 0; g < ENERGY_GROUPS_NUM; ++g) {
                double const Dg = calcEffectiveDiffusionCoefficient(cells_cgs[i], i, g);
                Vector3D grad_for_limiter = grad_temp_array[g] / (tess.GetVolume(i) * pow<3>(length_scale_));
                double const Eg_i = cells_cgs[i].Eg[g] * cells_cgs[i].density;
                double const min_grad = std::abs(Eg_i) / (1000.0 * cell_width);
                double const grad_abs = std::abs(fastabs(grad_for_limiter));
                if (grad_abs < min_grad) {
                    if (grad_abs > 0)
                        grad_for_limiter *= min_grad / grad_abs;
                    else
                        grad_for_limiter = Vector3D(min_grad, 0, 0);
                }

                double const lambda  =  CG::CalcSingleFluxLimiter(grad_for_limiter, Dg, Eg_i) / 3;
                double const sigma_t =  CG::speed_of_light / (3 * Dg) + 1e-100;
                double const R_g = abs(grad_temp_array[g]) / (tess.GetVolume(i) * pow<3>(length_scale_) * sigma_t * cells_cgs[i].Eg[g] * cells_cgs[i].density + 1e-200);
                if (abs(grad_temp_array[g]) < 1e-100) {
                    R2[i][g] = 1.0 / 3.0;
                } else {
                    R2[i][g] = lambda + lambda * lambda * R_g * R_g;
                }

                if (cells[i].ID==-1) {
                    std::cout<<"R2["<<i<<"]["<<g<<"] = "<<R2[i][g]<<" density "<<cells[i].density<<" T "<<cells[i].temperature<<" location "<<tess.GetMeshPoint(i)<<" lambda "<<lambda<<" R_g "<<R_g<<" sigma_t "<<sigma_t
                        <<" grad_Eg "<<abs(grad_temp_array[g]) / (tess.GetVolume(i) * pow<3>(length_scale_))<<" Eg "<<cells_cgs[i].Eg[g] * cells_cgs[i].density<<std::endl;
                }
            }
        }
    } else {
        R2.resize(Nlocal);
        for (std::size_t i=0; i < Nlocal; ++i) {
            R2[i].resize(ENERGY_GROUPS_NUM, 1.0/3.0);
            std::fill(
                R2[i].begin(),
                R2[i].end(),
                1.0/3.0 // set all values to 1/3.0 for the diffusion term
            );
        }
    }

    // Add the diffusion terms
    for (std::size_t i=0; i < Nlocal; ++i) {
        faces = tess.GetCellFaces(i);

        tess.GetNeighbors(i, neighbors);
        std::size_t const Nneighbors = neighbors.size();

        Vector3D const r_i = tess.GetMeshPoint(i);

        auto& cell_i = cells_cgs[i]; // reference and not const reference is because we change cell_i temperature to calculate the diffusion coefficient 

        for (std::size_t j=0; j < Nneighbors; ++j) {
            std::size_t const neighbor_j = neighbors[j];

            auto r_ij = r_i - tess.GetMeshPoint(neighbor_j);

            double const abs_r_ij = abs(r_ij);
            r_ij *= 1.0 / abs_r_ij; // normalize the vector perpendicular to the face between cells i and j

            double Eg_j = 0;
            bool const outside_point = tess.IsPointOutsideBox(neighbor_j);
            ComputationalCell3D* cell_j = outside_point ? nullptr : &cells_cgs[neighbor_j];
            for (size_t group=0; group<ENERGY_GROUPS_NUM; ++group) {
                double const Eg_i = cell_i.Eg[group] * cell_i.density;

                if (!outside_point) {
                    Eg_j = cell_j->Eg[group] * cell_j->density;

                    // since the diffusion terms are symmetric we only go update the matrix if i < j
                    if (i < neighbor_j) {
                        auto const& face_j = faces[j];
                        Vector3D const& gradient = grad[face_j];

                        // calculate the diffusion coefficient on the boundary using the maximal temperature of the cells
                        double const T_i = cell_i.temperature;
                        double const T_j = cell_j->temperature;
                        double const max_T = std::pow(0.5 * (pow<4>(T_i) + pow<4>(T_j)), 0.25);

                        cell_j->temperature = max_T;
                        cell_i.temperature = max_T;

                        double const D_i = calcEffectiveDiffusionCoefficient(cell_i, i, group);
                        double const D_j = calcEffectiveDiffusionCoefficient(*cell_j, neighbor_j, group);

                        cell_i.temperature = T_i;
                        cell_j->temperature = T_j;

                        double const D_ij = 2.0 * D_i * D_j / (D_i + D_j);

                        double lambda = 1.0;
                        if (flux_limiter_) {
                            double const dEg = Eg_i - Eg_j;

                            // double const gradE_magnitude = std::max(std::abs(fastabs(gradient)*dEg), std::numeric_limits<double>::min()*1e40);
                            double const grad_factor = 1;//std::max(0.15 * (max_abs_grad_E[i] + max_abs_grad_E[neighbor_j])/gradE_magnitude, 1.0);
                            Vector3D grad_for_limiter = gradient * dEg * grad_factor;
                            double const cell_width = std::max(tess.GetWidth(i) * length_scale_, 1e-200);
                            double const E_mid = 0.5 * (Eg_i + Eg_j);
                            double const min_grad = std::abs(E_mid) / (1000.0 * cell_width);
                            double const grad_abs = std::abs(fastabs(grad_for_limiter));
                            if (grad_abs < min_grad) {
                                if (grad_abs > 0)
                                    grad_for_limiter *= min_grad / grad_abs;
                                else
                                    grad_for_limiter = Vector3D(min_grad, 0, 0);
                            }

                            lambda = CG::CalcSingleFluxLimiter(grad_for_limiter, D_ij, E_mid);
                        }
                        double const lambdaD = lambda*D_ij;

                        double const A_j = tess.GetArea(face_j) * pow<2>(length_scale_);
                        double const flux  = dt_cgs * lambdaD * ScalarProd(gradient, r_ij) * A_j;

                        A[i * ENERGY_GROUPS_NUM + group][0] += flux;
                        A[i * ENERGY_GROUPS_NUM + group].push_back(-flux);
                        A_indeces[i * ENERGY_GROUPS_NUM + group].push_back(neighbor_j * ENERGY_GROUPS_NUM + group);

                        if (neighbor_j < Nlocal) { // check that neighboring cell is not boundary
                            A[neighbor_j * ENERGY_GROUPS_NUM + group][0] += flux;
                            A[neighbor_j * ENERGY_GROUPS_NUM + group].push_back(-flux);
                            A_indeces[neighbor_j * ENERGY_GROUPS_NUM + group].push_back(i * ENERGY_GROUPS_NUM + group);
                        }
                    }
                } else { // boundary condition
                    if (i < neighbor_j) {
                        boundary_calculator.setBoundaryValuesGroup(group, tess, i, neighbor_j, dt_cgs, cells_cgs, tess.GetArea(faces[j])*pow<2>(length_scale_), A[i * ENERGY_GROUPS_NUM + group][0], b[i * ENERGY_GROUPS_NUM + group], faces[j]);
                    }
                }
            }
        }
    }

    // Add velocity term
    for (std::size_t i=0; i<Nlocal; ++i) {

        faces = tess.GetCellFaces(i);
        tess.GetNeighbors(i, neighbors);
        std::size_t const Nneighbors = neighbors.size();
        double div_V = 0;
        Vector3D const r_i = tess.GetMeshPoint(i);
        for (std::size_t j=0; j<Nneighbors; ++j) {
            std::size_t const neighbor_j = neighbors[j];

            auto const r_ij = normalize(r_i - tess.GetMeshPoint(neighbor_j));

            double const A_ij = tess.GetArea(faces[j]) * pow<2>(length_scale_);
            Vector3D velocity_j;
            bool const is_outside = tess.IsPointOutsideBox(neighbor_j);
            if (!is_outside) {
                velocity_j = cells_cgs[neighbor_j].velocity;
            } else {
                double dummyEg_i, dummy_Eg_j;
                boundary_calculator.getOutsideValuesGroup(0, tess, i, neighbor_j, cells_cgs, dummyEg_i, dummy_Eg_j, velocity_j);
            }

            div_V -= 0.5*ScalarProd(cells_cgs[i].velocity+velocity_j, r_ij) * A_ij;
            if (hydro_on_ or doppler_on_) {
                for (size_t group=0; group<ENERGY_GROUPS_NUM; ++group) {
                    A[i * ENERGY_GROUPS_NUM + group][0] += -0.5*ScalarProd(cells_cgs[i].velocity+velocity_j, r_ij) * A_ij * dt_cgs * (0.5 - 0.5 * R2[i][group]);
                }
            }
        }

        if (doppler_on_) {
            // double const coeff = -div_V * dt_cgs / 3;
            double const coeff = -div_V * dt_cgs;
            for (std::size_t g=1; g<ENERGY_GROUPS_NUM; ++g) {
                if (div_V < 0) {
                    double const slope_left = get_doppler_slope(cells_cgs[i], g - 1, false);
                    size_t const gm = g - 1;
                    auto it = std::find(A_indeces[i * ENERGY_GROUPS_NUM + g].begin(), A_indeces[i * ENERGY_GROUPS_NUM + g].end(), i * ENERGY_GROUPS_NUM + gm);
                    if (it == A_indeces[i * ENERGY_GROUPS_NUM + g].end()) {
                        throw UniversalError("Not found [i*ENERGY_GROUPS_NUM + g - 1] in A_indeces (1)");
                    }
                    std::size_t const gm_index = static_cast<std::size_t>(it - A_indeces[i * ENERGY_GROUPS_NUM + g].begin());

                    if (A_indeces[i * ENERGY_GROUPS_NUM + g][gm_index] != i*ENERGY_GROUPS_NUM + gm) {
                        throw UniversalError("Not found [i*ENERGY_GROUPS_NUM + gm] in A_indeces (2)");
                    }

                    it = std::find(A_indeces[i * ENERGY_GROUPS_NUM + gm].begin(), A_indeces[i * ENERGY_GROUPS_NUM + gm].end(), i * ENERGY_GROUPS_NUM + g);
                    if (it == A_indeces[i * ENERGY_GROUPS_NUM + gm].end()) {
                        throw UniversalError("Not found [i*ENERGY_GROUPS_NUM + g] in A_indeces (1)");
                    }
                    std::size_t const g_index = static_cast<std::size_t>(it - A_indeces[i * ENERGY_GROUPS_NUM + gm].begin());

                    double coeff_left = 1 / energy_groups_width[gm] - 0.5 * slope_left  / energy_groups_width[gm];
                    coeff_left *= 0.5 - 0.5*R2[i][gm];
                    coeff_left *= coeff * energy_groups_boundary[g];

                    double coeff_right = 0.5 * slope_left  / energy_groups_width[g];
                    coeff_right *= 0.5 - 0.5*R2[i][g];
                    coeff_right *= coeff * energy_groups_boundary[g];

                    A[i * ENERGY_GROUPS_NUM + gm][0] += coeff_left;
                    A[i * ENERGY_GROUPS_NUM + g][0]  -= coeff_right;

                    A[i * ENERGY_GROUPS_NUM + g][gm_index] -= coeff_left;
                    A[i * ENERGY_GROUPS_NUM + gm][g_index] += coeff_right;
                } else {
                    double const slope_right = get_doppler_slope(cells_cgs[i], g, true);
                    size_t const gm = g - 1;
                    auto it = std::find(A_indeces[i * ENERGY_GROUPS_NUM + gm].begin(), A_indeces[i * ENERGY_GROUPS_NUM + gm].end(), i * ENERGY_GROUPS_NUM + g);
                    if (it == A_indeces[i * ENERGY_GROUPS_NUM + gm].end()) {
                        throw UniversalError("Not found [i*ENERGY_GROUPS_NUM + g] in A_indeces (1)");
                    }
                    std::size_t const g_index = static_cast<std::size_t>(it - A_indeces[i * ENERGY_GROUPS_NUM + gm].begin());

                    double coeff_right = 1 / energy_groups_width[g];
                    double coeff_right_right = 0;
                    if ((g + 1) < ENERGY_GROUPS_NUM) {
                        coeff_right += 0.5 * slope_right  / energy_groups_width[g + 1];
                        coeff_right_right = -0.5 * slope_right * energy_groups_width[g] / (energy_groups_width[g + 1] * energy_groups_width[g + 1]);
                    }
                    coeff_right *= coeff * (0.5 - 0.5*R2[i][g]) * energy_groups_boundary[g];
                    coeff_right_right *= coeff * energy_groups_boundary[g];

                    coeff_right_right *= (g + 1) < ENERGY_GROUPS_NUM ? 0.5 - 0.5*R2[i][g+1] : 0.5 - 0.5*R2[i][g];

                    A[i * ENERGY_GROUPS_NUM + g][0] -= coeff_right;
                    A[i * ENERGY_GROUPS_NUM + gm][g_index] += coeff_right;
                    if (g + 1 < ENERGY_GROUPS_NUM) {
                        size_t const gp = g + 1;
                        it = std::find(A_indeces[i * ENERGY_GROUPS_NUM + g].begin(), A_indeces[i * ENERGY_GROUPS_NUM + g].end(), i * ENERGY_GROUPS_NUM + gp);
                        std::size_t gp_index = static_cast<std::size_t>(it - A_indeces[i * ENERGY_GROUPS_NUM + g].begin());
                        if (A_indeces[i * ENERGY_GROUPS_NUM + g][gp_index] != i*ENERGY_GROUPS_NUM + gp) {
                            throw UniversalError("Not found [i*ENERGY_GROUPS_NUM + gp] in A_indeces (2)");
                        }
                        A[i * ENERGY_GROUPS_NUM + g][gp_index] -= coeff_right_right;

                        it = std::find(A_indeces[i * ENERGY_GROUPS_NUM + gm].begin(), A_indeces[i * ENERGY_GROUPS_NUM + gm].end(), i * ENERGY_GROUPS_NUM + gp);
                        gp_index = static_cast<std::size_t>(it - A_indeces[i * ENERGY_GROUPS_NUM + gm].begin());
                        if (A_indeces[i * ENERGY_GROUPS_NUM + gm][gp_index] != i*ENERGY_GROUPS_NUM + gp) {
                            throw UniversalError("Not found [i*ENERGY_GROUPS_NUM + gp] in A_indeces (2)");
                        }
                        A[i * ENERGY_GROUPS_NUM + gm][gp_index] += coeff_right_right;
                    }
                }
            }
        }
    }

    // Find maximum number of neighbors and allocate data
    // THIS SHOULD BE IN PRESTEP BUT BiCGSTAB CREATES A NEW MATRIX EVERY TIME IT IS CALLED. 
    // MAYBE MATRIX BUILDER SHOULD HOLD A MATRIX AS AN ATTRIBUTE
    std::size_t max_neighbors = 0;
    for (std::size_t i=0; i < Nlocal; ++i) {
        for (size_t group=0; group<ENERGY_GROUPS_NUM; ++group) {
            max_neighbors = std::max(max_neighbors, A[i * ENERGY_GROUPS_NUM + group].size());
        }
    }
    ++max_neighbors;

    for (std::size_t i=0; i < Nlocal; ++i) {
        for (size_t group=0; group<ENERGY_GROUPS_NUM; ++group) {
            A[i * ENERGY_GROUPS_NUM + group].resize(max_neighbors, 0);
            A_indeces[i * ENERGY_GROUPS_NUM + group].resize(max_neighbors, max_size_t);

            if (A[i * ENERGY_GROUPS_NUM + group][0] < 0) {
                std::cout << "Negative A in matrix build, ID " <<cells[i].ID<<" R2 "<<R2[i][group]<<" group "<<group<<std::endl;
            }
        }
    }
}
void MultigroupDiffusion::PostCG(Tessellation3D const& tess,
                                 std::vector<Conserved3D>& extensives,
                                 double const dt,
                                 std::vector<ComputationalCell3D>& cells,
                                 std::vector<double> const& CG_result,
                                 std::vector<double> const& full_CG_result) const {

    auto const N = tess.GetPointNo();
    std::vector<size_t> neighbors;
    face_vec faces;
    Vector3D dummy_v;
    Vector3D dP;

    double Einit = 0.0;
    for (std::size_t i = 0; i < N; ++i) {
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
    std::vector<std::string>::const_iterator it = binary_find(
        ComputationalCell3D::tracerNames.begin(),
        ComputationalCell3D::tracerNames.end(),
        string("Entropy")
    );

    if (it != ComputationalCell3D::tracerNames.end())
    {
        entropy_index = static_cast<size_t>(it - ComputationalCell3D::tracerNames.begin());
        with_entropy = true;
    }

    double min_T_E_added = 0;
    double d_Ek = 0;
    for (std::size_t i=0; i < N; ++i) {
        double const old_e_therm = extensives[i].internal_energy;
        double const volume = tess.GetVolume(i) * pow<3>(length_scale_);
        double Erad_tot = 0;
        double const f = fleck_factor[i];
        double const T  = old_Tm[i];
        double const kp = sigma_absorption_planck[i];
        double const Um = get_radiation_energy_density(T);

        double dE_absorption_emission = -volume * f * cdt * kp*Um;

        double dE_compton = 0.0;
        bool const do_compton = compton_on_ && (sigma_absorption_planck[i] * dt_cgs * CG::speed_of_light < compton_optical_depth_turn_off);
        if (do_compton) {
            generate_S_and_dSdUm_matrices(cells[i], i, dt_cgs, !use_n_zero[i]);

            for (std::size_t g=0; g < ENERGY_GROUPS_NUM; ++g) {

                dE_compton -= get_implicit_compton_contribution_to_b(tess, cells[i], i, g, dt_cgs);

                for (std::size_t gt=0; gt < ENERGY_GROUPS_NUM; ++gt) {
                    double const CG_res_i = std::max(CG_result[i * ENERGY_GROUPS_NUM + gt], std::numeric_limits<double>::min()*1e100);

                    dE_compton += get_implicit_compton_contribution(tess, cells[i], i, g, gt, dt_cgs) * CG_res_i;
                }
            }
        }
        double const Gamma_1 = 1.0 / Gammas[i];

        for (size_t group = 0; group < ENERGY_GROUPS_NUM; ++group) {

            double const CG_res = std::max(CG_result[i * ENERGY_GROUPS_NUM + group], std::numeric_limits<double>::min()*1e100);

            double const full_CG_res_i = std::max(
                full_CG_result[i * ENERGY_GROUPS_NUM + group],
                std::numeric_limits<double>::min()*1e100
            );

            extensives[i].Eg[group] = full_CG_res_i * volume * pow<2>(time_scale_) / (pow<2>(length_scale_) * mass_scale_);

            cells[i].Eg[group] =  extensives[i].Eg[group] / extensives[i].mass;
            Erad_tot += extensives[i].Eg[group];
            // absorption + emission must use raw sub_x (CG_result)
            dE_absorption_emission += volume * cdt * CG_res * sigma_absorption_group[i][group];
            auto const bg = planck_integal_group[i][group];
            for (std::size_t gt=0; gt < ENERGY_GROUPS_NUM; ++gt) {

                double const implicit_conribution_group_j = -volume*bg * (1 - f) * sigma_absorption_group[i][gt] * sigma_absorption_group[i][group] * cdt * Gamma_1;

                dE_absorption_emission += implicit_conribution_group_j * std::max(CG_result[i * ENERGY_GROUPS_NUM + gt], std::numeric_limits<double>::min()*1e100);
            }
        }

        dE_absorption_emission *= pow<2>(time_scale_) / (pow<2>(length_scale_) * mass_scale_);
        dE_compton *= pow<2>(time_scale_) / (pow<2>(length_scale_) * mass_scale_);

        extensives[i].energy += dE_absorption_emission + dE_compton;
        extensives[i].internal_energy += dE_absorption_emission + dE_compton;
        extensives[i].Erad = Erad_tot;
        cells[i].Erad =  extensives[i].Erad / extensives[i].mass;
        cells[i].internal_energy =  extensives[i].internal_energy / extensives[i].mass;

        tess.GetNeighbors(i, neighbors);
        std::size_t const Nneighbors = neighbors.size();
        faces = tess.GetCellFaces(i);
        auto const r_i = tess.GetMeshPoint(i);

        // momentum term
        if (hydro_on_) {
            size_t print_id = -1;
            Vector3D dP;
            for (size_t group = 0; group < ENERGY_GROUPS_NUM; ++group) {
                Vector3D gradEg, gradEg_new;
                double Eg_j, Eg_i = cells_cgs[i].Eg[group] * cells_cgs[i].density;
                double Eg_j_new, Eg_i_new = std::max(CG_result[i * ENERGY_GROUPS_NUM + group], std::numeric_limits<double>::min()*1e100);
                for (size_t j=0; j<Nneighbors; ++j) {
                    size_t const neighbor_j = neighbors[j];
                    Vector3D const r_ij = normalize(r_i - tess.GetMeshPoint(neighbor_j));
                    double const A_ij = tess.GetArea(faces[j]) * pow<2>(length_scale_);
                    bool const is_outside = tess.IsPointOutsideBox(neighbor_j);
                    if (!is_outside) {
                        gradEg += A_ij * 0.5 * (Eg_i + cells_cgs[neighbor_j].Eg[group] * cells_cgs[neighbor_j].density) * r_ij;
                        gradEg_new += A_ij * 0.5 * (Eg_i_new + std::max(CG_result[neighbor_j * ENERGY_GROUPS_NUM + group], std::numeric_limits<double>::min()*1e100)) * r_ij;
                    } else {
                        Vector3D dummy_v;
                        boundary_calculator.getOutsideValuesGroup(group, tess, i, neighbor_j, cells_cgs, Eg_i, Eg_j, dummy_v);
                        gradEg += A_ij * 0.5 * (Eg_i + Eg_j) * r_ij;
                        boundary_calculator.getOutsideValuesGroup(group, tess, i, neighbor_j, cells_cgs, Eg_i_new, Eg_j_new, dummy_v);
                        gradEg_new += A_ij * 0.5 * (Eg_i_new + Eg_j_new) * r_ij;
                    }
                }
                gradEg *= 1.0 / volume;
                double const D = calcEffectiveDiffusionCoefficient(cells_cgs[i], i, group);
                double const flux_limit = CG::CalcSingleFluxLimiter(gradEg, D, Eg_i);
                if(cells[i].ID == print_id)
                    std::cout<<"Group "<<group<<" flux_limit "<<flux_limit<<" sigma rossland "<<units::clight / (3 * D)<<" Eg_i "<<Eg_i<<" gradEg "<<gradEg<<std::endl;
                dP += (flux_limit / 3) * gradEg_new;
            }
            dP *= dt_cgs * time_scale_ / (length_scale_ * mass_scale_);
            double mass_i = extensives[i].mass;
            double old_Ek = 0.5 * ScalarProd(extensives[i].momentum, extensives[i].momentum) / mass_i;
            if(cells[i].ID == print_id)
                std::cout<<"dP "<<dP<<" cell momentum "<<extensives[i].momentum<<std::endl;
            extensives[i].momentum += dP;

            double const new_Ek = 0.5 * ScalarProd(extensives[i].momentum, extensives[i].momentum) / mass_i;

            d_Ek += new_Ek - old_Ek;

            extensives[i].energy = extensives[i].internal_energy + new_Ek;
        }
        // EOS
        try {
            if (minimum_temperature_ > 0) {
                double const min_e_therm = eos_.dT2e(cells[i].density, minimum_temperature_, cells[i].tracers, ComputationalCell3D::tracerNames);
                if (cells[i].internal_energy < min_e_therm) {
                    double const Trad = std::pow(cells[i].Erad * cells[i].density * mass_scale_ / (units::arad * length_scale_ * pow<2>(time_scale_)), 0.25);
                    if (cells[i].temperature > Trad &&
                        ((extensives[i].Erad * cells[i].temperature > 1e2 * old_e_therm * Trad) ||
                            (fleck_factor[i] < 0.75 &&
                                extensives[i].Erad > min_e_therm *extensives[i].mass)) &&
                        Trad > minimum_temperature_) {
                        double const min_e_therm2 = eos_.dT2e(cells[i].density, Trad, cells[i].tracers, ComputationalCell3D::tracerNames);
                        double const delta_e = min_e_therm2 - cells[i].internal_energy;
                        cells[i].internal_energy += delta_e;
                        double const dE_change = delta_e * extensives[i].mass;
                        double const ratio = (extensives[i].Erad - dE_change) / extensives[i].Erad;
                        if (ratio > 0) {
                            min_T_E_added += dE_change;
                            extensives[i].energy += dE_change;
                            extensives[i].internal_energy += dE_change;
                            extensives[i].Erad *= ratio;
                            cells[i].Erad *= ratio;
                            for (size_t k = 0; k < ENERGY_GROUPS_NUM; ++k) {
                                extensives[i].Eg[k] *= ratio;
                                cells[i].Eg[k] *= ratio;
                            }
                        }
                    }
                }
                if (min_e_therm > cells[i].internal_energy) {
                    if (cells[i].temperature < 2e4) {
                        double const delta_e = min_e_therm - cells[i].internal_energy;
                        cells[i].internal_energy += delta_e;
                        min_T_E_added += delta_e * extensives[i].mass;
                        extensives[i].energy += delta_e * extensives[i].mass;
                        extensives[i].internal_energy += delta_e * extensives[i].mass;
                    }
                    if (cells[i].internal_energy < 0) {
                        UniversalError eo("Negative thermal energy in MG diffusion");
                        double const Trad = std::pow(cells[i].Erad * cells[i].density * mass_scale_ / (units::arad * length_scale_ * pow<2>(time_scale_)), 0.25);
                        eo.addEntry("ID", cells[i].ID);
                        eo.addEntry("density", cells[i].density);
                        eo.addEntry("T", old_Tm[i]);
                        eo.addEntry("Trad", Trad);
                        eo.addEntry("dE_absorption_emission", dE_absorption_emission);
                        eo.addEntry("dE_compton", dE_compton);
                        eo.addEntry("Vx", cells[i].velocity.x);
                        eo.addEntry("Vy", cells[i].velocity.y);
                        eo.addEntry("Vz", cells[i].velocity.z);
                        for (size_t k = 0; k < ENERGY_GROUPS_NUM; ++k) {
                            eo.addEntry("Eg[" + std::to_string(k) + "]", old_Eg[i][k]);
                        }
                        throw eo;
                    }
                }
            }
            cells[i].temperature = eos_.de2T(cells[i].density, cells[i].internal_energy, cells[i].tracers, ComputationalCell3D::tracerNames);
            cells[i].pressure = eos_.de2p(cells[i].density, cells[i].internal_energy, cells[i].tracers, ComputationalCell3D::tracerNames);
            cells[i].velocity = extensives[i].momentum / extensives[i].mass;
            if (with_entropy) {
                double new_entropy = eos_.dp2s(cells[i].density, cells[i].pressure, cells[i].tracers, ComputationalCell3D::tracerNames);
                cells[i].tracers[entropy_index] = new_entropy;
                extensives[i].tracers[entropy_index] = new_entropy * extensives[i].mass;
            }
        }
        catch (UniversalError& eo) {
            eo.addEntry("MultigroupDiffusion, cell index", i);
            eo.addEntry("ID", cells[i].ID);
            eo.addEntry("density", cells[i].density);
            eo.addEntry("internal_energy", cells[i].internal_energy);
            eo.addEntry("dE_absorption_emission", dE_absorption_emission);
            eo.addEntry("dE_compton", dE_compton);
            eo.addEntry("internal_energy extensive", extensives[i].internal_energy);
            eo.addEntry("f", f);
            eo.addEntry("kp", kp);
            eo.addEntry("Um", Um);
            eo.addEntry("old T", old_Tm[i]);
            eo.addEntry("volume", volume);
            eo.addEntry("cdt", cdt);
            eo.addEntry("Erad", cells[i].Erad * cells[i].density);
            eo.addEntry("old_e_therm", old_e_therm);
            for (size_t group = 0; group < ENERGY_GROUPS_NUM; ++group) {
                eo.addEntry("Eg[" + std::to_string(group) + "]_full", full_CG_result[i * ENERGY_GROUPS_NUM + group]);
                eo.addEntry("Eg[" + std::to_string(group) + "]", CG_result[i * ENERGY_GROUPS_NUM + group]);
                eo.addEntry("sigma_absorption_group[" + std::to_string(group) + "]", sigma_absorption_group[i][group]);
                eo.addEntry("sigma_rossland_group[" + std::to_string(group) + "]", units::clight / (3 * calcEffectiveDiffusionCoefficient(cells_cgs[i], i, group)));
            }
            throw eo;
        }
    }

    int rank = 0;
#ifdef RICH_MPI
    MPI_Allreduce(MPI_IN_PLACE, &min_T_E_added, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &d_Ek, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    bool was_bad = false;
    if (good_end == 0) {
        std::cout<<"Zero good_end rank "<<rank<<std::endl;
        was_bad = true;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &good_end, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    if (was_bad) {
        std::cout<<"rank "<<rank<<" good_end "<<good_end<<std::endl;
    }
#endif

    if (good_end == 0) {
        throw UniversalError("Negative energy in PostCG");
    }

    double Efinal = 0;
    for (std::size_t i=0; i<N; ++i) {
        Efinal += extensives[i].Erad + extensives[i].energy;
    }

#ifdef RICH_MPI
    MPI_Allreduce(MPI_IN_PLACE, &Efinal, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif

// #ifdef DEBUG
    if (rank == 0) {
        std::cout << std::setprecision(14) << "Einit = " << Einit << ", Efinal = " << Efinal <<" min_T_E_added = "<<min_T_E_added<<" d_Ek "<<d_Ek<<std::endl;
        std::cout << std::setprecision(16) << "|Einit-Efinal|/Einit = " << std::abs(Einit - Efinal) / Einit << std::endl;
    }
// #endif
}

void MultigroupDiffusion::calculate_fleck_factor(Tessellation3D const& tess, std::vector<ComputationalCell3D> const& cells, double dt_cgs) const
{
    size_t const N = tess.GetPointNo();
    bool printed_warning = false;
    for (size_t i = 0; i < N; ++i) {
        double const sigma_planck = sigma_absorption_planck[i];
        double const T = old_Tm[i];
        double cv = eos_.dT2cv(cells[i].density, T, cells[i].tracers, ComputationalCell3D::tracerNames);

        // TODO: What is energy ratio (see Diffusion.cpp same line)
        cv *= mass_scale_ / (pow<2>(time_scale_)*length_scale_);
        double const cv_bar = cv / get_radiation_cv(T);

        double Gamma = sigma_planck;
        bool negative_upsilon = false;
        double upsilon = 0;
        bool did_compton = false;
        if (compton_on_ &&
            (sigma_planck * dt_cgs * CG::speed_of_light < compton_optical_depth_turn_off)) {
            did_compton = true;
            generate_S_and_dSdUm_matrices(cells[i], i, dt_cgs);
            upsilon = calculate_Upsilon(cells[i]);

            if (upsilon < 0.0) negative_upsilon = true;
            Gamma += upsilon;
        }

        double f = CG::FleckFactor(dt_cgs, 1.0/cv_bar, Gamma);

        if (f < 0 || (negative_upsilon && std::abs(upsilon) > 0.1 * sigma_planck)) {
            if (!printed_warning && f < 0.5) {
                std::cout << "Warning: Negative fleck factor in cell " << cells[i].ID 
                        << ", recalculating with n=0" << std::endl;
                std::cout << "  Original: Gamma=" << Gamma << ", upsilon=" << upsilon 
                        << ", f=" << f << std::endl;
            }
            double old_f = f;
            use_n_zero[i] = true;
            generate_S_and_dSdUm_matrices(cells[i], i, dt_cgs, false);
            upsilon = calculate_Upsilon(cells[i]);
            Gamma = sigma_planck + upsilon;
            f = CG::FleckFactor(dt_cgs, 1.0/cv_bar, Gamma);
            if(not printed_warning && 0.5 > old_f) {
                std::cout << "  With n=0: Gamma=" << Gamma << ", upsilon=" << upsilon 
                        << ", f=" << f << std::endl;
                printed_warning = true;
            }
        } else {
            use_n_zero[i] = false;
        }

        if (did_compton && sigma_scattering_group[i][0] < 1e-50)
            fillComptonScatteringRates(i, tau, n);

        fleck_factor[i] = f;
        Gammas[i] = Gamma;
    }

    // Second pass: fill Compton scattering rates for cells skipped by the
    // main loop (Compton turned off by optical depth, or ghost cells i >= N).
    // We check group 0 as a proxy -- if it was filled above, all groups were.
    if (compton_on_) {
        auto const Ntotal = cells.size();
        double constexpr fac_n = pow<3>(units::clight) / (8.0 * M_PI * units::planck_constant);
        std::vector<std::vector<double>> local_tau(ENERGY_GROUPS_NUM, std::vector<double>(ENERGY_GROUPS_NUM, 0.0));
        std::vector<std::vector<double>> local_dtau(ENERGY_GROUPS_NUM, std::vector<double>(ENERGY_GROUPS_NUM, 0.0)); // required by get_tau_matrix, unused
        std::vector<double> local_n(ENERGY_GROUPS_NUM, 0.0);
        for (std::size_t i = 0; i < Ntotal; ++i) {
            if (sigma_scattering_group[i][0] > 1e-50)
                continue;
            if (i >= N && tess.IsPointOutsideBox(i))
                continue;
            auto const& cell = cells[i];
            if (cell.temperature <= 0.0 || cell.density <= 0.0)
                continue;
            double const T = std::min(compton_matrix_gen.get_maximum_temperature_grid() * 0.9999,
                                      (i < N) ? old_Tm[i] : cell.temperature);
            double const rho_cgs = cell.density * mass_scale_ / pow<3>(length_scale_);
            compton_matrix_gen.get_tau_matrix(T, rho_cgs, 1.0, 1.0, local_tau, local_dtau);

            for (std::size_t g = 0; g < ENERGY_GROUPS_NUM; ++g) {
                double const dnu = energy_groups_width[g] / units::planck_constant;
                double const nu = energy_groups_center[g] / units::planck_constant;
                double const Eg = cell.Eg[g] * cell.density * mass_scale_ /
                                  (length_scale_ * pow<2>(time_scale_));
                local_n[g] = std::min(100.0, fac_n * Eg / (pow<3>(nu) * dnu));
            }

            fillComptonScatteringRates(i, local_tau, local_n);
        }
    }
}

void MultigroupDiffusion::calculate_group_absorption_and_scattering_coefficients(Tessellation3D const& tess,
                                                                                 std::vector<ComputationalCell3D> const& cells,
                                                                                 double const dt) const {
    auto const N = tess.GetPointNo();
    auto const Ntotal = cells.size();
    sigma_absorption_group.resize(Ntotal);
    sigma_scattering_group.resize(Ntotal);
    std::vector<std::size_t> cooling_neighbors;
    face_vec cooling_faces;
    for (std::size_t i=0; i < N; ++i) {
        double const Trad = std::pow(cells[i].Erad * cells[i].density / CG::radiation_constant, 0.25);
        double cv = eos_.dT2cv(cells[i].density * pow<3>(length_scale_) / mass_scale_, cells[i].temperature) * mass_scale_ / (pow<2>(time_scale_)*length_scale_);
        double const volume = tess.GetVolume(i) * length_scale_ * length_scale_ * length_scale_;
        double const cell_width = std::max(tess.GetWidth(i) * length_scale_, 1e-200);

        sigma_absorption_group[i].resize(ENERGY_GROUPS_NUM);
        sigma_scattering_group[i].resize(ENERGY_GROUPS_NUM);

        auto const& cell = cells[i];
        double const kT_1 = 1.0 / (CG::boltzmann_constant * cell.temperature);
        double const Um = get_radiation_energy_density(cell.temperature);
        for (std::size_t g=0; g < ENERGY_GROUPS_NUM; ++g) {

            sigma_absorption_group[i][g] = std::min(coefficient_calculator.CalcAbsorptionOpacity(cell, energy_groups_center[g]),
                CG::max_coupling_strength / (CG::speed_of_light * dt));
            if (protections_on_) {
                if (Trad > 1.1 * cells[i].temperature && cv < 0.1 * get_radiation_cv(Trad)) {
                    sigma_absorption_group[i][g] = std::min(sigma_absorption_group[i][g],
                        cv * Trad / (CG::speed_of_light * dt * cells[i].Erad * cells[i].density));
                }
            }

            double const a = energy_groups_boundary[g] * kT_1;
            double const b = energy_groups_boundary[g+1] * kT_1;
            double const bg = planck_integral::planck_integral(a, b);
            if (cell.density > 1e-12 && CG::speed_of_light * dt * sigma_absorption_group[i][g] * (cell.Eg[g] * cell.density - bg * Um) > 2 * cv * cell.temperature)
            {
                double const new_sigma =  2 * cv * cell.temperature / (CG::speed_of_light * dt * (cell.Eg[g] * cell.density - bg * Um));
                sigma_absorption_group[i][g] = std::min(sigma_absorption_group[i][g], new_sigma);//std::max(cell.density * 0.34 * 0.1, new_sigma));
            }

            if (sigma_absorption_group[i][g] < 0.) {
                throw UniversalError("negative absorption coefficient");
            }

            sigma_scattering_group[i][g] = coefficient_calculator.CalcScatteringOpacity(cell, energy_groups_center[g]);

            if (sigma_scattering_group[i][g] < 0.) {
                throw UniversalError("negative scattering coefficient");
            }
        }

        if(cooling_time_limiter_on_)
        {
            tess.GetNeighbors(i, cooling_neighbors);
            cooling_faces = tess.GetCellFaces(i);
            auto &neighbors = cooling_neighbors;
            auto &faces = cooling_faces;

            double div_v = 0;
            for(std::size_t j = 0; j < neighbors.size(); ++j)
            {
                if(j >= faces.size())
                    continue;
                std::size_t const neigh = neighbors[j];
                Vector3D const r_ij = normalize(tess.GetMeshPoint(i) - tess.GetMeshPoint(neigh));
                Vector3D vel_j = cells[i].velocity;
                if(neigh < N || !tess.IsPointOutsideBox(neigh))
                    vel_j = cells[neigh].velocity;
                div_v -= 0.5 * ScalarProd(cells[i].velocity + vel_j, r_ij) *
                         tess.GetArea(faces[j]) * length_scale_ * length_scale_;
            }
            div_v /= std::max(volume, 1e-200);

            double const speed = fastabs(cells[i].velocity);
            double const compression_speed = std::max(-div_v, 0.0) * cell_width;
            if(speed > 1.0 && compression_speed > 0.25 * speed && compression_speed * speed > cells[i].internal_energy * 0.25)
            {
                double const hydro_time = 1.0 / std::max(-div_v, 1e-200);
                double const T_local = std::max(cells[i].temperature, 1.0);
                double const inv_kT = 1.0 / (CG::boltzmann_constant * T_local);
                double const Um_local = get_radiation_energy_density(T_local);
                double planck_exchange = 0.0;
                double compton_exchange = 0.0;
                for(std::size_t g = 0; g < ENERGY_GROUPS_NUM; ++g)
                {
                    double const a = energy_groups_boundary[g] * inv_kT;
                    double const b = energy_groups_boundary[g + 1] * inv_kT;
                    double const bg = planck_integral::planck_integral(a, b);
                    double const Eg_density = cells[i].Eg[g] * cells[i].density;

                    planck_exchange += CG::speed_of_light * sigma_absorption_group[i][g] * (bg * Um_local - Eg_density);
                }

                // Use the full multigroup Compton operator for the gas-radiation
                // exchange estimate instead of a gray (Tgas-Trad) approximation.
                if(compton_on_)
                {
                    ComputationalCell3D cell_for_compton = cells[i];
                    cell_for_compton.density *= pow<3>(length_scale_) / mass_scale_;
                    cell_for_compton.internal_energy *= pow<2>(time_scale_) / pow<2>(length_scale_);
                    cell_for_compton.Erad *= pow<2>(time_scale_) / pow<2>(length_scale_);
                    cell_for_compton.velocity *= time_scale_ / length_scale_;
                    for(std::size_t g = 0; g < ENERGY_GROUPS_NUM; ++g)
                    {
                        cell_for_compton.Eg[g] *= pow<2>(time_scale_) / pow<2>(length_scale_);
                    }

                    generate_S_and_dSdUm_matrices(cell_for_compton, i, dt, true);
                    for(std::size_t g = 0; g < ENERGY_GROUPS_NUM; ++g)
                    {
                        double const Eg_density = cells[i].Eg[g] * cells[i].density;
                        for(std::size_t gt = 0; gt < ENERGY_GROUPS_NUM; ++gt)
                        {
                            compton_exchange -= CG::speed_of_light * S[g][gt] * Eg_density;
                        }
                    }
                }

                double const net_cooling_power = planck_exchange + compton_exchange;
                if(net_cooling_power > 0)
                {
                    double const thermal_energy = std::max(cells[i].internal_energy * cells[i].density, 1e-200);
                    double const cool_time = thermal_energy / net_cooling_power;
                    double const target_cool_time = 2.0 * hydro_time;
                    if(cool_time < target_cool_time)
                    {
                        double const target_cooling_power = thermal_energy / target_cool_time;
                        double const opacity_scale = std::max(target_cooling_power / std::max(net_cooling_power, 1e-200), 1e-6);
                        for(std::size_t g = 0; g < ENERGY_GROUPS_NUM; ++g)
                        {
                            sigma_absorption_group[i][g] *= opacity_scale;
                        }
                        if(compton_on_)
                            compton_limiter_scale_[i] = opacity_scale;
                    }
                }
            }
        }
    }

    for (std::size_t i = N; i < Ntotal; ++i) {
        sigma_absorption_group[i].resize(ENERGY_GROUPS_NUM, 0.0);
        sigma_scattering_group[i].resize(ENERGY_GROUPS_NUM, 0.0);
        if (tess.IsPointOutsideBox(i))
            continue;
        auto const& cell = cells[i];
        if (cell.temperature <= 0.0)
            continue;
        for (std::size_t g = 0; g < ENERGY_GROUPS_NUM; ++g) {
            sigma_absorption_group[i][g] = std::min(
                coefficient_calculator.CalcAbsorptionOpacity(cell, energy_groups_center[g]),
                CG::max_coupling_strength / (CG::speed_of_light * dt));
            sigma_scattering_group[i][g] = coefficient_calculator.CalcScatteringOpacity(cell, energy_groups_center[g]);
        }
    }
}

void MultigroupDiffusion::calculate_planck_integrals(Tessellation3D const& tess,
                                                     std::vector<ComputationalCell3D> const& cells) const {

    auto const N = tess.GetPointNo();

    planck_integal_group.resize(N);
    for (std::size_t i=0; i<N; ++i) {
        planck_integal_group[i].resize(ENERGY_GROUPS_NUM);
        double const kT = CG::boltzmann_constant * old_Tm[i];
        double planck_sum = 0.0;
        for (std::size_t g=0; g<ENERGY_GROUPS_NUM; ++g) {

            double const a = energy_groups_boundary[g] / kT;
            double const b = energy_groups_boundary[g+1] / kT;

            double const bg = planck_integral::planck_integral(a, b);

            planck_integal_group[i][g] = bg;
            planck_sum += bg;
        }

        if (planck_sum < (1. - 1e-2) && not displayed_warning_) {
            displayed_warning_ = true;
            std::cout << "bad groups! planckian not covered well! cell " << i << " T " << old_Tm[i] <<" ID "<<cells[i].ID<<std::endl;
            std::cout << "bad planck_sum " << planck_sum << std::endl;
            // throw UniversalError("bad groups! planckian not covered well!");
        }
    }
}

void MultigroupDiffusion::calculate_planck_absorption_coefficient(Tessellation3D const& tess,
                                                                                std::vector<ComputationalCell3D> const& cells) const {
    auto const N = tess.GetPointNo();
    std::fill(sigma_absorption_planck.begin(), sigma_absorption_planck.end(), 0.0);

    for (std::size_t i=0; i<N; ++i) {
        auto const& cell = cells[i];
        for (std::size_t g=0; g < ENERGY_GROUPS_NUM; ++g) {
            double const sigma = sigma_absorption_group[i][g];
            double const bg = planck_integal_group[i][g];

            sigma_absorption_planck[i] += sigma * bg;
        }
    }
}

void MultigroupDiffusion::generate_S_and_dSdUm_matrices(ComputationalCell3D const& cell, std::size_t const cell_index, double const dt_cgs, bool const calculate_n) const {
    cell_id_of_compton_matrices = cell.ID;

    double constexpr fac = pow<3>(units::clight) / (8.0*M_PI*units::planck_constant);

    for (std::size_t g=0; g < ENERGY_GROUPS_NUM; ++g) {
        if (calculate_n) {
            double const dnu = energy_groups_width[g]/units::planck_constant;
            double const nu = energy_groups_center[g]/units::planck_constant;

            double const Eg = cell.Eg[g] * cell.density * mass_scale_ / (length_scale_ * pow<2>(time_scale_));

            n[g] = std::min(100.0, fac * Eg / (pow<3>(nu)*dnu));
        } else {
            n[g] = 0.0;
        }
    }

    double const A = 1.0;
    double const Z = 1.0;
    double const T = std::min(compton_matrix_gen.get_maximum_temperature_grid() * 0.9999, old_Tm[cell_index]);
    compton_matrix_gen.get_tau_matrix(T, cell.density*mass_scale_/pow<3>(length_scale_), A, Z, tau, dtau_dUm);

    // transform dtau_dT to dtau_dUm
    // double const beta = 1.0 / (4.0*CG::radiation_constant*pow<3>(T));
    // for (std::size_t g=0; g < ENERGY_GROUPS_NUM; ++g) {
    //     for (auto& val : dtau_dUm[g]) {
    //         val *= beta;
    //     }
    // }

    auto const [up_scattering_last, down_scattering_last] = compton_matrix_gen.get_last_group_upscattering_and_downscattering(T, cell.density*mass_scale_/pow<3>(length_scale_), A, Z);

    fill_zero(S);
    fill_zero(dSdUm);

    for (std::size_t g=0; g < ENERGY_GROUPS_NUM; ++g) {
        for (std::size_t gt=0; gt < ENERGY_GROUPS_NUM; ++gt) {
            if (g+1 == ENERGY_GROUPS_NUM and gt+1 == ENERGY_GROUPS_NUM) {
                S[g][g] += (up_scattering_last - down_scattering_last)*(1.0 + n[g]);
                dSdUm[g][g] += dtau_dUm[g][g] * (1.0 + n[g]);
                continue;
            }

            // in scattering
            double const in_scattering_factor = energy_groups_center[g] / energy_groups_center[gt] * (1.0 + n[g]);
            double const in_scattering_factor_dsdum = energy_groups_center[g] / energy_groups_center[gt] * (1.0 + n[g]);
            S[gt][g] += tau[gt][g] * in_scattering_factor;
            dSdUm[gt][g] += dtau_dUm[gt][g] * in_scattering_factor_dsdum;

            // out scattering
            double const out_scattering_factor = 1.0 + n[gt];
            S[g][g] -= tau[g][gt] * out_scattering_factor;
            dSdUm[g][g] -= dtau_dUm[g][gt] * (1 + n[gt]);
        }
    }
    double const Um = CG::radiation_constant*pow<4>(cell.temperature);
    double const Um_factor = 1.0 / (4 * CG::radiation_constant*pow<3>(cell.temperature));
    for (std::size_t g=0; g < ENERGY_GROUPS_NUM; ++g) {
        for (std::size_t gt=0; gt < ENERGY_GROUPS_NUM; ++gt) {
            dSdUm[g][gt] *= Um_factor;
        }
    }

    if (protections_on_) {
        double dE = 0;
        for (std::size_t g=0; g < ENERGY_GROUPS_NUM; ++g) {
            for (std::size_t gt=0; gt < ENERGY_GROUPS_NUM; ++gt) {
                dE -= S[g][gt] * cell.Eg[g] * cell.density * mass_scale_ / (length_scale_ * pow<2>(time_scale_));
            }
        }
        dE *= dt_cgs * units::clight;

        double dE_dT = 0;
        for (std::size_t g=0; g < ENERGY_GROUPS_NUM; ++g) {
            for (std::size_t gt=0; gt < ENERGY_GROUPS_NUM; ++gt) {
                dE_dT -= dSdUm[g][gt] * cell.Eg[g] * cell.density * mass_scale_ / (length_scale_ * pow<2>(time_scale_));
            }
        }
        dE_dT *= dt_cgs * units::clight*Um;
        double const E_cell = cell.internal_energy * cell.density * mass_scale_ / (length_scale_ * pow<2>(time_scale_));
        double const Trad = std::pow(cell.Erad * cell.density * mass_scale_ / (units::arad * length_scale_ * pow<2>(time_scale_)), 0.25);
        if (dE > E_cell) {
            if ((cell.internal_energy * Trad < 0.1 * cell.Erad * cell.temperature) && Trad > cell.temperature) {
                double max_dE = cell.density * cell.internal_energy * (Trad - cell.temperature) / cell.temperature;
                max_dE *= mass_scale_ / (length_scale_ * pow<2>(time_scale_));
                double const reduce_factor = max_dE / dE;
                if (reduce_factor < 1) {
                    for (std::size_t g=0; g < ENERGY_GROUPS_NUM; ++g) {
                        for (std::size_t gt=0; gt < ENERGY_GROUPS_NUM; ++gt) {
                            S[gt][g] *= reduce_factor;
                            dSdUm[gt][g] *= reduce_factor;
                        }
                    }
                }
            }
        } else {
            double const dE_factor = 0.4;
            if (dE < -dE_factor * E_cell) {
                double const reduce_factor = std::abs(dE_factor * E_cell / dE);
                if (reduce_factor < 1)
                {
                    for (std::size_t g=0; g < ENERGY_GROUPS_NUM; ++g) {
                        for (std::size_t gt=0; gt < ENERGY_GROUPS_NUM; ++gt) {
                            S[gt][g] *= reduce_factor;
                            dSdUm[gt][g] *= reduce_factor;
                        }
                    }
                }
            }
        }
    }

    if(cooling_time_limiter_on_ && compton_limiter_scale_[cell_index] < 1.0)
    {
        double const scale = compton_limiter_scale_[cell_index];
        for(std::size_t g = 0; g < ENERGY_GROUPS_NUM; ++g)
        {
            for(std::size_t gt = 0; gt < ENERGY_GROUPS_NUM; ++gt)
            {
                S[g][gt] *= scale;
                dSdUm[g][gt] *= scale;
            }
        }
    }
}

double MultigroupDiffusion::calculate_Upsilon(ComputationalCell3D const& cell) const {
    assert(cell_id_of_compton_matrices == cell.ID);

    double Upsilon = 0.0;

    for (std::size_t gt=0; gt < ENERGY_GROUPS_NUM; ++gt) {
        for (std::size_t gtt=0; gtt < ENERGY_GROUPS_NUM; ++gtt) {
            Upsilon += dSdUm[gt][gtt] * cell.Eg[gt] * cell.density * mass_scale_ / (length_scale_ * pow<2>(time_scale_));
        }
    }

    return Upsilon;
}

double MultigroupDiffusion::get_implicit_compton_contribution(Tessellation3D const& tess, ComputationalCell3D const& cell, std::size_t const cell_index, std::size_t const g, std::size_t const gt, double const dt_cgs) const {
    assert(cell_id_of_compton_matrices == cell.ID);

    double const volume = tess.GetVolume(cell_index) * pow<3>(length_scale_);

    double const cdt = CG::speed_of_light*dt_cgs;

    double const T = old_Tm[cell_index];
    double const cv = eos_.dT2cv(cell.density, T, cell.tracers, ComputationalCell3D::tracerNames)*mass_scale_ / (pow<2>(time_scale_)*length_scale_);

    double const cv_bar = cv / get_radiation_cv(T);
    double const cdt_cv_bar = cdt / cv_bar;

    double const kg = sigma_absorption_group[cell_index][g];
    double const kgbg = kg*planck_integal_group[cell_index][g];

    double const f = fleck_factor[cell_index];


    double implicit_contribution = 0.0;

    double const coeff_1 = volume*cdt*cdt_cv_bar*kgbg*f;

    double const Um_old = get_radiation_energy_density(T);

    implicit_contribution -= volume*cdt*S[gt][g];
    double sum_dSdUm_Egtt = 0.0;
    for (std::size_t gtt=0; gtt < ENERGY_GROUPS_NUM; ++gtt) {
        sum_dSdUm_Egtt += dSdUm[gtt][g]*cell.Eg[gtt]*cell.density*mass_scale_ / (length_scale_ * pow<2>(time_scale_));
    }

    double const A = volume*cdt*cdt_cv_bar*f*(kgbg + sum_dSdUm_Egtt);

    implicit_contribution -= volume*cdt*cdt_cv_bar*f*sum_dSdUm_Egtt*sigma_absorption_group[cell_index][gt];
    for (std::size_t gtt=0; gtt < ENERGY_GROUPS_NUM; ++gtt) {
        implicit_contribution += A*S[gt][gtt];
    }

    return implicit_contribution;
}

double MultigroupDiffusion::get_implicit_compton_contribution_to_b(Tessellation3D const& tess, ComputationalCell3D const& cell, std::size_t const cell_index, std::size_t const g, double const dt_cgs) const {
    assert(cell_id_of_compton_matrices == cell.ID);

    double const volume = tess.GetVolume(cell_index) * pow<3>(length_scale_);

    double const cdt = CG::speed_of_light*dt_cgs;

    double const T = old_Tm[cell_index];
    double const cv = eos_.dT2cv(cell.density, T, cell.tracers, ComputationalCell3D::tracerNames)*mass_scale_ / (pow<2>(time_scale_)*length_scale_);

    double const cv_bar = cv / get_radiation_cv(T);
    double const cdt_cv_bar = cdt / cv_bar;

    double const kg = sigma_absorption_group[cell_index][g];
    double const kgbg = kg*planck_integal_group[cell_index][g];
    double const kp = sigma_absorption_planck[cell_index];

    double const f = fleck_factor[cell_index];
    double const Um_old = get_radiation_energy_density(T);

    double sum_dSdUm_Egtt = 0.0;
    for (std::size_t gtt=0; gtt < ENERGY_GROUPS_NUM; ++gtt) {
        sum_dSdUm_Egtt += dSdUm[gtt][g]*cell.Eg[gtt]*cell.density*mass_scale_ / (length_scale_ * pow<2>(time_scale_));
    }

    // you can do a simplifcation with the volume*cdt*kgbg*f*Um_old already in the b 
    double const contribution_to_b = volume*cdt*kp*Um_old*(kgbg/kp*(1.0 - (1.0+cdt_cv_bar*kp)*f) - cdt_cv_bar*f*sum_dSdUm_Egtt);

    return contribution_to_b;
}

double MultigroupDiffusion::calcEffectiveDiffusionCoefficient(
    ComputationalCell3D const& cell,
    std::size_t cell_index,
    std::size_t group) const
{
    double D = coefficient_calculator.CalcDiffusionCoefficient(cell, energy_groups_center[group]);
    if (compton_on_ && !coefficient_calculator.ComptonIncludedInTransport()) {
        if (cell_index < sigma_scattering_group.size() &&
            group < sigma_scattering_group[cell_index].size() &&
            sigma_scattering_group[cell_index][group] > 1e-50)
        {
            double sigma_transport = CG::speed_of_light / (3.0 * D);
            sigma_transport += sigma_scattering_group[cell_index][group];
            D = CG::speed_of_light / (3.0 * sigma_transport);
        }
    }
    return D;
}

void MultigroupDiffusion::fillComptonScatteringRates(
    std::size_t cell_index,
    std::vector<std::vector<double>> const& tau_mat,
    std::vector<double> const& occ) const
{
    for (std::size_t g = 0; g < ENERGY_GROUPS_NUM; ++g) {
        double out = 0.0;
        for (std::size_t gt = 0; gt < ENERGY_GROUPS_NUM; ++gt)
            if (gt != g) out += tau_mat[g][gt] * (1.0 + occ[gt]);
        sigma_scattering_group[cell_index][g] = std::max(0.0, out);
    }
}
