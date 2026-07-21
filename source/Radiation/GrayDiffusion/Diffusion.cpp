#include "Diffusion.hpp"
#include <boost/math/special_functions.hpp>
#include <algorithm>
#include <boost/math/tools/roots.hpp>

#ifdef    RICH_MPI
#include "mpi/mpi_commands.hpp"
#endif

namespace {

double FleckFactorCompton(double const dt, double const beta, double const sigma_a, double const sigma_s, double const Erad, double const Cv)
{
    return 1.0 / (1 + beta * dt * sigma_a * CG::speed_of_light + dt * 16 * sigma_s * CG::boltzmann_constant * Erad / (CG::electron_mass * CG::speed_of_light * Cv));
}

double calculate_total_energy(
    Tessellation3D const& tess,
    std::vector<Conserved3D> const& extensives
){     
    auto const N = tess.GetPointNo();
    
    double total_energy = 0.0;
    for(size_t i = 0; i < N; ++i) total_energy += extensives[i].Erad + extensives[i].energy;
    
#ifdef RICH_MPI
    MPI_Allreduce(MPI_IN_PLACE, &total_energy, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif  
    return total_energy;
}

double Trad(double const Er) {
    return std::pow(Er / CG::radiation_constant, 0.25);
}

double Um(double const T) {
    return CG::radiation_constant*T*T*T*T;
}

double radiation_cv(double const T){
    return 4 * CG::radiation_constant * T * T * T;
}

} // namespace 

Diffusion::Diffusion(DiffusionCoefficientCalculator const& D_coefficient_calc, 
                     DiffusionBoundaryCalculator const& boundary_calc,
                     EquationOfState const& eos, 
                     std::vector<std::string> const zero_cells, 
                     bool const flux_limiter, 
                     bool const hydro_on, 
                     bool const compton_on,
                     bool const cooling_time_limiter_on) : 
                                            RadiationDriver(eos, 
                                                            zero_cells, 
                                                            flux_limiter, 
                                                            hydro_on, 
                                                            compton_on),
                                             D_coefficient_calcualtor(D_coefficient_calc),
                                             boundary_calc_(boundary_calc), 
                                             sigma_planck(),
                                             sigma_s(), 
                                             fleck_factor(),
                                             D(),
                                             R2(),
                                             cell_flux_limiter(),
                                             new_Er(),
                                             new_Er_full(),
                                             old_Er(),
                                             do_iterations_on_Um(false),
                                             use_new_Er_for_x0(false),
                                             cooling_time_limiter_on_(cooling_time_limiter_on) {}

double Diffusion::GetSingleFleckFactor(
    ComputationalCell3D const& cell, 
    std::size_t const cell_index, 
    double const dt 
) const 
{
    double const Er = cell.Erad * cell.density * energy_density_scale_;
    
    double const T = cell.temperature;
    double Cv = eos_.dT2cv(cell.density, T, cell.tracers, ComputationalCell3D::tracerNames);
    double const energy_ratio = Cv * T / (cell.internal_energy * cell.density);

    Cv *= energy_density_scale_;
    double const beta = std::max(1.0, 0.5 * energy_ratio) * radiation_cv(T) / Cv;
    
    return compton_on_ ? 
            FleckFactorCompton(dt * time_scale_, beta, sigma_planck[cell_index], sigma_s[cell_index], Er, Cv) 
            : 
            FleckFactor(dt * time_scale_, beta, sigma_planck[cell_index]);
}

bool Diffusion::prestep(Tessellation3D const& tess,
                        std::vector<ComputationalCell3D> const& cells) const {
    auto const N = tess.GetPointNo();
    
    sigma_planck.resize(N, 0.0);
    sigma_planck.shrink_to_fit();
    sigma_s.resize(N, 0.0);
    sigma_s.shrink_to_fit();
    fleck_factor.resize(N, 0.0);
    fleck_factor.shrink_to_fit();
    D.resize(N, 0.0);
    D.shrink_to_fit();
    R2.resize(N, 0.0);
    R2.shrink_to_fit();
    cell_flux_limiter.resize(N, 0.0);
    cell_flux_limiter.shrink_to_fit();
    old_T.resize(N, 0.0);


    new_Er.resize(N, 0.0);
    new_Er.shrink_to_fit();
    new_Er_full.resize(N, 0.0);
    new_Er_full.shrink_to_fit();
    old_Er.resize(N, 0.0);
    old_Er.shrink_to_fit();

    cells_cgs.resize(N);

    for(std::size_t i=0; i < N; ++i){
        old_Er[i] = cells[i].Erad * cells[i].density;

        if(old_Er[i] < 0.0){
            UniversalError eo("negative Erad");
 			eo.addEntry("i", i);
			eo.addEntry("old_Er", old_Er[i]);
			eo.addEntry("ID", cells[i].ID);
			eo.addEntry("density", cells[i].density);
            throw eo;
        }

        old_T[i] = cells[i].temperature;
    }
    
    #ifdef RICH_MPI
    MPI_exchange_data(tess, old_T, true);
    #endif

    set_scales(mass_scale_, length_scale_, time_scale_);

    return true;
}

bool Diffusion::poststep() const {    
    std::vector<ComputationalCell3D>().swap(cells_cgs);
    return true;
}

double Diffusion::calculate_dt(double const dt,
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
		if(fleck_factor[i] < 0.4)
			diff *= 0.2;
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
	MPI_exchange_data(tess, cells, true);	
#endif
	if(rank == max_data.mpi_id)
	{
		std::cout<<"Radiation time step ID "<<cells[max_loc].ID<<" old Er "<<old_Er[max_loc]<<" new Er "<<cells[max_loc].Erad * cells[max_loc].density<<
		" diff "<<max_diff<<" Tgas "<<cells[max_loc].temperature<<" Trad "<<std::pow(new_Er[max_loc] / CG::radiation_constant, 0.25)<<" max_Er "<<max_Er<<" rank "<<rank<<" density "<<cells[max_loc].density<<
		" width "<<tess.GetWidth(max_loc)<<" Tgas_old "<<old_T[max_loc]<<" location "<<tess.GetMeshPoint(max_loc)<<std::endl;
		PrintDebugData(max_loc);
        std::cout<<"Next time step is "<<dt * std::min(1.25, 0.15 / max_diff)<<std::endl;
	}

    return dt * std::min(1.25, 0.15 / max_diff);
}

bool Diffusion::step(double const tolerance, 
                     int& total_iters, 
                     Tessellation3D const& tess, 
                     std::vector<ComputationalCell3D>& cells,
                     std::vector<Conserved3D>& extensives,
                     double const dt,
                     double const time) const {


    int rank = 0;
#ifdef RICH_MPI
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif
    
    std::vector<Conserved3D> extensives_temp(extensives);
    std::vector<ComputationalCell3D> cells_temp(cells);
    
    load_cells_cgs(tess, cells);
    calculate_planck_absorption_coefficient(tess);
    calculate_scattering_coefficient(tess);
    apply_opacity_limiters(tess, cells, dt);
    calculate_cell_diffusion_coefficients(tess);

    calculate_fleck_factor(tess, cells, dt);

    bool good_end = false;
    new_Er = CG::BiCGSTAB(tolerance, total_iters, tess, cells, dt, *this, time, new_Er_full, good_end);
    
    if(not good_end) return false;
    
    try {
        fix_small_negative_Er(tess, cells);
    } catch(UniversalError const& eo){
        reportError(eo);
        return false;
    }

    try {
        PostCG(tess, extensives, dt, cells, new_Er, new_Er_full);
    } catch(UniversalError const& eo) {
        if(rank == 0){
            std::cout<< "PostCG Exception:" << std::endl;
            reportError(eo);
        }
        
        extensives = extensives_temp;
        cells = cells_temp;

        double const iterations_CG_eps = 1e-16;
        do_iterations_on_Um = true;
        int total_iters_2=0;

        bool const res_iterations = iterations(
            iterations_CG_eps,
            total_iters_2,
            tess,
            cells,
            extensives,
            dt,
            time
        );

        if(!res_iterations){
            extensives = extensives_temp;
            cells = cells_temp;
        }
        
        return res_iterations;
    }

    return true;
}


void
Diffusion::load_cells_cgs(
    Tessellation3D const& tess, 
    std::vector<ComputationalCell3D> const& cells_not_cgs) const {

    cells_cgs = cells_not_cgs;

    auto const Nlocal = tess.GetPointNo();
    for(std::size_t cell=0; cell < Nlocal; ++cell){
        cells_cgs[cell].density     *= density_scale_;
        cells_cgs[cell].Erad        *= specific_energy_scale_;
        cells_cgs[cell].Erad_dt     *= specific_energy_scale_ / time_scale_;
        cells_cgs[cell].Erad_dt_dt  *= specific_energy_scale_ / (time_scale_*time_scale_);
        cells_cgs[cell].velocity    *= velocity_scale_;
    }

#ifdef RICH_MPI
	MPI_exchange_data(tess, cells_cgs, true);	
#endif
}

void Diffusion::BuildMatrix(
    Tessellation3D const& tess, 
    mat& A, 
    size_t_mat& A_indeces, 
    std::vector<ComputationalCell3D> const& cells,
    double const dt, 
    std::vector<double>& b, 
    std::vector<double>& x0, 
    double const current_time) const
{
    int rank = 0;
#ifdef RICH_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif
    size_t const Nlocal = tess.GetPointNo();

    b.resize(Nlocal, 0);
    x0.resize(Nlocal, 0);

    std::vector<size_t> neighbors;
    face_vec faces;
    
    std::vector<size_t> zero_indeces;
    size_t const Nzero = zero_cells_.size();
    for(size_t i = 0; i < Nzero; ++i)
        zero_indeces.push_back(binary_index_find(ComputationalCell3D::stickerNames, zero_cells_[i]));
    
    double const zero_value = 1e-10;
    std::vector<double> new_Er(Nlocal, 0), Er_for_limit(Nlocal, 0);
    
    double const dt_cgs = dt * time_scale_;
    for(size_t i = 0; i < Nlocal; ++i)
    {
        double const volume = tess.GetVolume(i) * volume_scale_;
        double const cell_width = std::max(tess.GetWidth(i) * length_scale_, 1e-200);
        bool set_to_zero = false;
        
        for(size_t j = 0; j < Nzero; ++j)
            if(cells_cgs[i].stickers[zero_indeces[j]])
                set_to_zero = true;
        
        double const Er = cells_cgs[i].Erad * cells_cgs[i].density * (set_to_zero ? zero_value : 1);
        new_Er[i] = Er;

        double const T = cells_cgs[i].temperature;


        b[i] = volume * Er;

        // cells have the updated temperature while cells_cgs have the ones at the start of the time step
        double const Um_i = Um(cells[i].temperature);
        if(use_new_Er_for_x0){
            x0[i] = new_Er[i];
        } else if(fleck_factor[i] < 0.8 && Um_i > Er)
        {
            double const prefactor = fleck_factor[i] * dt * CG::speed_of_light * sigma_planck[i];
            x0[i] = (Er + prefactor * Um_i) / (1 + prefactor);
        }
        else 
            x0[i] = std::min(2 * Er, std::max(0.5 * Er, Er + cells_cgs[i].Erad_dt * cells_cgs[i].density * dt_cgs  + 0.5 * cells_cgs[i].Erad_dt_dt * cells_cgs[i].density * dt_cgs * dt_cgs));
            
        b[i] += volume * fleck_factor[i] * dt_cgs * CG::speed_of_light * sigma_planck[i] * Um_i;

        Er_for_limit[i] = std::min(Er, std::max(1e-5 * Er, Er + dt_cgs * fleck_factor[i] * sigma_planck[i] * CG::speed_of_light * (Um_i - Er)));
    }

    size_t max_neigh = 0;
    // Find maximum number of neighbors and allocate data
    for(size_t i = 0; i < Nlocal; ++i)
        max_neigh = std::max(max_neigh, tess.GetNeighbors(i).size());
    ++max_neigh;
    
    A.clear();
    A.resize(Nlocal);
    A_indeces.clear();
    A_indeces.resize(Nlocal);

    // Build the matrix
    for(size_t i = 0; i < Nlocal; ++i)
    {
        A_indeces[i].push_back(i);
        double const volume = tess.GetVolume(i) * volume_scale_;
        
        A[i].push_back(volume * (1 + fleck_factor[i] * dt_cgs * CG::speed_of_light * sigma_planck[i]));
        
        if(compton_on_ && cells[i].tracers[1] > 0.5)
        {
            double const T = cells_cgs[i].temperature;
            double const Tr = Trad(new_Er[i]);
	        double const pre_factor = fleck_factor[i] * dt_cgs * 4 * sigma_s[i] * CG::boltzmann_constant / (CG::electron_mass * CG::speed_of_light);
            double const compton_term = pre_factor * (Tr - T);
            double const theta = (fleck_factor[i] < 0.5 || std::abs(compton_term) > 1e-3) ? 1 : 0.1;
            
            A[i][0] += pre_factor * (Tr - (1 - theta) * T) * volume;
            b[i] += pre_factor * volume * theta * T * new_Er[i];
        }
        
        if(A[i][0] < 0){
	        std::cout   << "Negative A in matrix build, density " <<cells_cgs[i].density
                        << " T " << cells_cgs[i].temperature 
                        << " fleck " << fleck_factor[i]
                        << " sig_P " << sigma_planck[i] 
                        << " sig_s " << sigma_s[i] 
                        << " dt " << dt_cgs 
                        << " Erad " << cells_cgs[i].Erad * cells_cgs[i].density
                        << " compton term " << fleck_factor[i] * dt_cgs * 4 * sigma_s[i] * CG::boltzmann_constant / (CG::electron_mass * CG::speed_of_light) 
                        << std::endl;
        }
    }

    std::vector<double> max_R;
    max_R.reserve(Nlocal);
    for(size_t i = 0; i < Nlocal; ++i)
    {
        double max_R_local = 0;
        tess.GetNeighbors(i, neighbors);
        size_t const Nneigh = neighbors.size();
        double const Er = cells_cgs[i].Erad * cells_cgs[i].density;
        Vector3D const CM = tess.GetCellCM(i);
        for(size_t j = 0; j < Nneigh; ++j)
        {
            size_t const neighbor_j = neighbors[j];
            if(neighbor_j < Nlocal || !tess.IsPointOutsideBox(neighbor_j))
            {
                bool set_to_zero = false;
                for(size_t k = 0; k < Nzero; ++k)
                    if(cells_cgs[neighbor_j].stickers[zero_indeces[k]])
                        set_to_zero = true;
                
                double const Er_j = cells_cgs[neighbor_j].Erad * cells_cgs[neighbor_j].density * (set_to_zero ? zero_value : 1);
                Vector3D const cm_ij = CM - tess.GetCellCM(neighbor_j);
                Vector3D const grad_E = cm_ij * (1.0 / (length_scale_ * ScalarProd(cm_ij, cm_ij)));                
                
                max_R_local = std::max(max_R_local, std::abs(fastabs(grad_E) * (Er - Er_j)));
            }
        }
        max_R.push_back(max_R_local);
    }
#ifdef RICH_MPI
    MPI_exchange_data(tess, max_R, true);
#endif
    Vector3D dummy_v;
    std::vector<Vector3D> gradE(Nlocal);
    std::vector<double> max_neighbor_R(Nlocal, 0);
    for(size_t i = 0; i < Nlocal; ++i)
    {
        faces = tess.GetCellFaces(i);
        tess.GetNeighbors(i, neighbors);
        size_t const Nneigh = neighbors.size();
        Vector3D const CM = tess.GetCellCM(i);
        Vector3D const point = tess.GetMeshPoint(i);
        
        gradE[i] = Vector3D(0, 0, 0);
        
        double const Er = cells_cgs[i].Erad * cells_cgs[i].density;
        bool self_zero = false;
        
        for(size_t k = 0; k < Nzero; ++k)
            if(cells_cgs[i].stickers[zero_indeces[k]])
                self_zero = true;
        
        for(size_t j = 0; j < Nneigh; ++j)
        {
            size_t const neighbor_j = neighbors[j];
            Vector3D r_ij = point - tess.GetMeshPoint(neighbor_j);
            double const r_ij_size = abs(r_ij);
            r_ij *= 1.0 / r_ij_size;
            double Er_j = 0;
            
            if(!tess.IsPointOutsideBox(neighbor_j))
            {
                bool set_to_zero = false;
                for(size_t k = 0; k < Nzero; ++k)
                    if(cells_cgs[neighbor_j].stickers[zero_indeces[k]])
                        set_to_zero = true;
                
                Er_j = cells_cgs[neighbor_j].Erad * cells_cgs[neighbor_j].density * (set_to_zero ? zero_value : 1);
                
                if(i < neighbor_j)
                {
                    Vector3D const cm_ij = CM - tess.GetCellCM(neighbor_j);
                    Vector3D const grad_E = cm_ij * (1.0 / (length_scale_ * ScalarProd(cm_ij, cm_ij)));                
                    
                    
                    double const avgT = std::pow(0.5 * (pow<4>(old_T[i]) + pow<4>(old_T[neighbor_j])), 0.25);
                    
                    cells_cgs[i].temperature = avgT;
                    double const D1 =  D_coefficient_calcualtor.CalcDiffusionCoefficient(cells_cgs[i]);
                    cells_cgs[i].temperature = old_T[i];
                    
                    cells_cgs[neighbor_j].temperature = avgT;
                    double const D2 =  D_coefficient_calcualtor.CalcDiffusionCoefficient(cells_cgs[neighbor_j]);
                    cells_cgs[neighbor_j].temperature = old_T[neighbor_j];
                    
                    double mid_D = 2 * D1 * D2 / (D1 + D2);

                    double const grad_magnitude = std::max(std::numeric_limits<double>::min() * 1e40, std::abs(fastabs(grad_E) * (Er - Er_j)));
                    double grad_factor = 1.0;
                    max_neighbor_R[i] = std::max(max_neighbor_R[i], max_R[neighbor_j]);
                    
                    if(grad_magnitude < 0.15 * (max_R[i] + max_R[neighbor_j]))
                        grad_factor = 0.15 * (max_R[i] + max_R[neighbor_j]) / grad_magnitude;
                    
                    double const flux_limiter = flux_limiter_ ? CalcSingleFluxLimiter(grad_E * ((Er - Er_j) * grad_factor), mid_D, 0.5 * (Er + Er_j)) : 1;
                    mid_D *= flux_limiter;
                    
                    double const flux = ((self_zero || set_to_zero) ? tess.GetArea(faces[j]) * area_scale_ * dt_cgs * CG::speed_of_light * 0.5 : ScalarProd(grad_E, r_ij) * tess.GetArea(faces[j]) * area_scale_ * dt_cgs * mid_D); 
                    
                    if(neighbor_j < Nlocal)
                    {
                        A[i][0] += flux;
                        A[i].push_back(-flux);
                        A_indeces[i].push_back(neighbor_j);
                        A[neighbor_j].push_back(-flux);
                        A_indeces[neighbor_j].push_back(i);
                        A[neighbor_j][0] += flux;
                    }                  
                    else
                    {
                        A[i][0] += flux;
                        A[i].push_back(-flux);
                        A_indeces[i].push_back(neighbor_j);
                    }
                }
            }
            else
            {
                if(i < neighbor_j)
                    boundary_calc_.SetBoundaryValues(tess, i, neighbor_j, dt_cgs, cells_cgs, tess.GetArea(faces[j]) * area_scale_, A[i][0], b[i], faces[j]);
                boundary_calc_.GetOutSideValues(tess, cells_cgs, i, neighbor_j, new_Er, Er_j, dummy_v);
            }
            gradE[i] += r_ij * (tess.GetArea(faces[j]) * area_scale_ * 0.5 * (Er + Er_j));
        }
    }
    for(size_t i = 0; i < Nlocal; ++i)
    {
        double const volume = tess.GetVolume(i) * volume_scale_;
        gradE[i] *= -1.0 / volume;
        faces = tess.GetCellFaces(i);
        tess.GetNeighbors(i, neighbors);
        size_t const Nneigh = neighbors.size();
        Vector3D const point = tess.GetMeshPoint(i);
        
        double const Er = cells_cgs[i].Erad * cells_cgs[i].density; 

        double const grad_magnitude = std::max(std::numeric_limits<double>::min() * 1e40, std::abs(fastabs(gradE[i])));
        if(grad_magnitude < 0.5 * max_neighbor_R[i])
            gradE[i] *= 0.5 * max_neighbor_R[i] / grad_magnitude;
        Vector3D grad_for_limiter = gradE[i];
        if(flux_limiter_)
        {
            double const cell_width = std::max(tess.GetWidth(i) * length_scale_, 1e-200);
            double const min_grad = std::abs(Er_for_limit[i]) / (1000.0 * cell_width);
            double const grad_abs = std::abs(fastabs(grad_for_limiter));
            if(grad_abs < min_grad)
            {
                if(grad_abs > 0)
                    grad_for_limiter *= min_grad / grad_abs;
                else
                    grad_for_limiter = Vector3D(min_grad, 0, 0);
            }
        }
        double const flux_limiter = flux_limiter_ ? CalcSingleFluxLimiter(grad_for_limiter, D[i], Er_for_limit[i]) : 1;
        cell_flux_limiter[i] = flux_limiter;
        Vector3D const CM = tess.GetCellCM(i);
        double const v_ratio = std::min(1.0, 0.05 * CG::speed_of_light / (fastabs(cells_cgs[i].velocity) + 1e-2));
        
        for(size_t j = 0; j < Nneigh; ++j)
        {
            size_t const neighbor_j = neighbors[j];
            if(!tess.IsPointOutsideBox(neighbor_j))
            {
                Vector3D r_ij = point - tess.GetMeshPoint(neighbor_j);
                double const r_ij_size = abs(r_ij);
                r_ij *= 1.0 / r_ij_size;
                Vector3D const cm_ij = CM - tess.GetCellCM(neighbor_j);
                Vector3D const grad_E = r_ij * ScalarProd(r_ij, cm_ij) * (1.0 / (length_scale_ * ScalarProd(cm_ij, cm_ij)));   

                double const mid_D = D[i];
                double const momentum_relativity_term = -0.5 * dt_cgs * flux_limiter * tess.GetArea(faces[j]) * area_scale_ * (v_ratio * fleck_factor[i] * 2 * 3 * sigma_planck[i] * mid_D / CG::speed_of_light - 1) * ScalarProd(cells_cgs[i].velocity, r_ij) / 3;
                A[i][0] += momentum_relativity_term;
                
                auto it = std::find(A_indeces[i].begin(), A_indeces[i].end(), neighbor_j);
                if(it == A_indeces[i].end()) throw UniversalError("Key not equal in diffusion");
                
                size_t const neigh_counter = static_cast<size_t>(it - A_indeces[i].begin());
                if(A_indeces[i][neigh_counter] != neighbor_j) throw UniversalError("Key not equal value in diffusion");
                
                A[i][neigh_counter] += momentum_relativity_term;
            }
            else
                boundary_calc_.SetMomentumTermBoundary(tess, i, neighbor_j, dt_cgs, cells_cgs[i],
                    tess.GetArea(faces[j]) * area_scale_, A[i][0], b[i], faces[j], fleck_factor[i],
                    flux_limiter, D[i], sigma_planck[i]);
        }
        
        R2[i] = flux_limiter_ ? flux_limiter / 3 + boost::math::pow<2>(flux_limiter * abs(gradE[i]) * D[i] / (CG::speed_of_light * Er)) : 1.0 / 3.0;
        A[i][0] -= volume * fleck_factor[i] * dt_cgs * 0.5 * (3 - R2[i]) * sigma_planck[i] * std::min(0.01 * CG::speed_of_light * CG::speed_of_light, ScalarProd(cells_cgs[i].velocity, cells_cgs[i].velocity)) / CG::speed_of_light;
    }
    for(size_t i = 0; i < Nlocal; ++i)
    {
        A[i].resize(max_neigh, 0);
        A_indeces[i].resize(max_neigh, max_size_t);
	
        if(A[i][0] < 0)
	        std::cout   << "Negative A in matrix build, density " << cells_cgs[i].density
                        << " T " << cells_cgs[i].temperature 
                        << " fleck " << fleck_factor[i]
                        << " sig_P " << sigma_planck[i] << " dt " << dt_cgs 
                        << " Erad " << cells_cgs[i].Erad * cells_cgs[i].density 
                        << std::endl;
    }
}

void Diffusion::PostCG(
    Tessellation3D const& tess, 
    std::vector<Conserved3D>& extensives, 
    double const dt, 
    std::vector<ComputationalCell3D>& cells,
    std::vector<double>const& full_CG_result, 
    std::vector<double> const& CG_result
) const
{
    double const max_v = 0.1 * CG::speed_of_light * velocity_scale_;
    double const dt_cgs = dt * time_scale_;

    Vector3D dummy_v;
    std::vector<size_t> neighbors;
    face_vec faces;
    size_t const N = tess.GetPointNo();
    bool const entropy = !(std::find(ComputationalCell3D::tracerNames.begin(), ComputationalCell3D::tracerNames.end(), std::string("Entropy")) ==
		ComputationalCell3D::tracerNames.end());
    size_t const entropy_index = static_cast<size_t>(std::find(ComputationalCell3D::tracerNames.begin(),
        ComputationalCell3D::tracerNames.end(), std::string("Entropy")) - ComputationalCell3D::tracerNames.begin());
    std::vector<size_t> zero_indeces;
    size_t const Nzero = zero_cells_.size();
    for(size_t i = 0; i < Nzero; ++i)
        zero_indeces.push_back(binary_index_find(ComputationalCell3D::stickerNames, zero_cells_[i]));

    double const Einit = calculate_total_energy(tess, extensives);

    int good_end = 1;
    for(size_t i = 0; i < N; ++i)
    {
        double const old_e_therm = extensives[i].internal_energy;
        double const volume = tess.GetVolume(i) * volume_scale_;
        extensives[i].Erad = full_CG_result[i] * volume / energy_scale_;
        double const T = cells[i].temperature;
        
        double old_Tr = 0;
        double compton_term = 0;
        double e_absorb = fleck_factor[i] * CG::speed_of_light * dt_cgs * sigma_planck[i] * CG_result[i] * volume;
        double e_emitt = -fleck_factor[i] * CG::speed_of_light * dt_cgs * sigma_planck[i] * Um(T) * volume;
        
        double e_v2 =  dE_v_squared(
            tess, i,
            CG_result[i],
            cells_cgs[i].velocity,
            max_v,
            dt_cgs
        );
        
        double const e_absorb_emitt = dE_absorption_emission(
            tess, i,
            CG_result[i],
            cells[i].temperature,
            dt_cgs
        );

        double dE = e_absorb_emitt + e_v2;        

        if(compton_on_)
        {
            double const old_Er = cells_cgs[i].Erad * cells_cgs[i].density;
            old_Tr = Trad(old_Er);
	        
            compton_term = dE_compton(
                tess, i,
                CG_result[i],
                cells_cgs[i].temperature,
                old_Er,
                dt_cgs
            );

            dE += compton_term;

            compton_term /= energy_scale_;
        }
        
        dE /= energy_scale_;
        e_absorb /= energy_scale_;
        e_emitt /= energy_scale_;
        e_v2 /= energy_scale_;

        extensives[i].energy += dE;
        extensives[i].internal_energy += dE;
        
        if(do_iterations_on_Um and GetSingleFleckFactor(cells[i], i, dt) < 0.1){
            compute_equilibrium_from_energy_sum(tess, i, cells, extensives, volume, CG_result, dt_cgs);
        } else {
            if(is_energy_invalid(extensives[i]))
            {
                good_end = 0;
                print_postcg1_debug(i, cells, extensives, CG_result, full_CG_result, T, dE, old_e_therm, e_emitt, e_v2, e_absorb_emitt, compton_term, old_Tr);
                break;
            }


            tess.GetNeighbors(i, neighbors);
            faces = tess.GetCellFaces(i);
            Vector3D const point = tess.GetMeshPoint(i);
            Vector3D const CM = tess.GetCellCM(i);
            double etherm_mid = extensives[i].internal_energy;
            
            Vector3D gradE;
            double total_relativity = dE_relativity(tess, i, CG_result, dt_cgs, gradE);
            extensives[i].energy += total_relativity;
            extensives[i].internal_energy += total_relativity;
            
            Vector3D dP;
            double Erad_dE = 0;
            double dE_mom = 0;
            if(hydro_on_)
            {
                auto const [dP_val, dE_val, Erad_dE_val] = dP_and_dE_momentum(i, gradE, dt_cgs, extensives[i].momentum, extensives[i].mass);
                dP = dP_val;
                dE_mom = dE_val;
                Erad_dE = Erad_dE_val;
                extensives[i].momentum += dP;
                extensives[i].Erad += dE_mom;
                
                if(extensives[i].Erad < 0 && dE_mom < 0 && std::abs(dE_mom) < extensives[i].energy * 0.01)
                    extensives[i].Erad -= dE_mom;
                
                extensives[i].energy = extensives[i].internal_energy + ScalarProd(extensives[i].momentum, extensives[i].momentum) / (2 * extensives[i].mass);
            }

            if(is_energy_invalid(extensives[i]) || cells[i].Erad < 0)
            {
                print_postcg2_debug(i, tess, cells, extensives, CG_result, full_CG_result, T, max_v, dt_cgs, dP, Erad_dE, e_absorb, e_emitt, e_v2, total_relativity, etherm_mid, gradE, CM, point, neighbors, faces);
                good_end = 0;
            }
        }

        double const old_Edot = cells[i].Erad_dt;
        cells[i].Erad_dt = (extensives[i].Erad / extensives[i].mass - cells[i].Erad) / dt;
        cells[i].Erad_dt_dt = (cells[i].Erad_dt - old_Edot) / dt;
        
        if(!std::isfinite(cells[i].Erad_dt) || !std::isfinite(cells[i].Erad_dt)) {
            std::cout   << "Bad Edot " << cells[i].Erad_dt 
                        << " edot_dt_dt " << cells[i].Erad_dt_dt 
                        << " i " << i 
                        << " ID " << cells[i].ID 
                        << " dt " << dt 
                        << " Erad " << extensives[i].Erad 
                        << " m " << extensives[i].mass 
                        << " old_Edot " << old_Edot << std::endl;
        }

        cells[i].Erad = extensives[i].Erad / extensives[i].mass;
        cells[i].internal_energy = extensives[i].internal_energy / extensives[i].mass;
        
        try
        {
            cells[i].temperature = eos_.de2T(cells[i].density, cells[i].internal_energy, cells[i].tracers, ComputationalCell3D::tracerNames);
            cells[i].pressure = eos_.de2p(cells[i].density, cells[i].internal_energy, cells[i].tracers, ComputationalCell3D::tracerNames);
            cells[i].velocity = extensives[i].momentum / extensives[i].mass;    
            
            if(entropy)
            {
                cells[i].tracers[entropy_index] = eos_.dp2s(cells[i].density, cells[i].pressure, cells[i].tracers, ComputationalCell3D::tracerNames);
                extensives[i].tracers[entropy_index] = cells[i].tracers[entropy_index] * extensives[i].mass;
            }
        }
        catch(UniversalError &eo)
        {
            reportError(eo);
            good_end = 0;
            break;
        }
    }
    int rank = 0;
#ifdef RICH_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    bool was_bad = false;
    if(good_end == 0)
    {
        std::cout << "Zero good_end rank " << rank << std::endl;
        was_bad = true;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &good_end, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    
    if(was_bad) std::cout<<"rank "<<rank<<" good_end "<<good_end<<std::endl;
#endif
    
    if(good_end == 0) throw UniversalError("Negative energy in POSTCG");

    double const Efinal = calculate_total_energy(tess, extensives);

    if(rank == 0) std::cout<<std::setprecision(14) << "Einit " << Einit << " Efinal " << Efinal << std::endl;
}

void Diffusion::calculate_planck_absorption_coefficient(
    Tessellation3D const& tess) const 
{
    auto const N = tess.GetPointNo();
    for(std::size_t i=0; i < N; ++i){    
        sigma_planck[i] = D_coefficient_calcualtor.CalcPlanckOpacity(cells_cgs[i]);

        if(sigma_planck[i] < 0.0) {
            throw UniversalError("Negative Sigma Planck")
                    .addEntry("Sigma Planck", sigma_planck[i])
                    .addEntry("cell ID", cells_cgs[i].ID);
        }
    }
}

void Diffusion::calculate_scattering_coefficient(
    Tessellation3D const& tess
) const
{
    auto const N = tess.GetPointNo();
    for(std::size_t i=0; i < N; ++i){
        sigma_s[i] = D_coefficient_calcualtor.CalcScatteringOpacity(cells_cgs[i]);
        
        if(sigma_s[i] < 0.0) {
            throw UniversalError("Negative Sigma Scattering")
                    .addEntry("Sigma Scattering", sigma_s[i])
                    .addEntry("cell ID", cells_cgs[i].ID);
        }
    }
}

void Diffusion::apply_opacity_limiters(
    Tessellation3D const& tess,
    std::vector<ComputationalCell3D> const& cells,
    double const dt
) const
{
    size_t const Nlocal = tess.GetPointNo();
    double const dt_cgs = dt * time_scale_;

    for(size_t i = 0; i < Nlocal; ++i)
    {
        sigma_planck[i] = std::min(sigma_planck[i], 10.0 / (CG::speed_of_light * dt_cgs));
    }

    if(!cooling_time_limiter_on_)
        return;

    std::vector<size_t> neighbors;
    face_vec faces;
    for(size_t i = 0; i < Nlocal; ++i)
    {
        double const volume = tess.GetVolume(i) * volume_scale_;
        double const cell_width = std::max(tess.GetWidth(i) * length_scale_, 1e-200);
        double const Er = cells_cgs[i].Erad * cells_cgs[i].density;

        faces = tess.GetCellFaces(i);
        tess.GetNeighbors(i, neighbors);
        double div_v = 0;
        for(size_t j = 0; j < neighbors.size(); ++j)
        {
            size_t const neigh = neighbors[j];
            Vector3D const r_ij = normalize(tess.GetMeshPoint(i) - tess.GetMeshPoint(neigh));
            Vector3D vel_j = cells_cgs[i].velocity;
            if(neigh < Nlocal || !tess.IsPointOutsideBox(neigh))
                vel_j = cells_cgs[neigh].velocity;
            div_v -= 0.5 * ScalarProd(cells_cgs[i].velocity + vel_j, r_ij) * tess.GetArea(faces[j]) * length_scale_ * length_scale_;
        }
        div_v /= std::max(volume, 1e-200);

        double const speed = fastabs(cells_cgs[i].velocity);
        double const compression_speed = std::max(-div_v, 0.0) * cell_width;
        if(speed > 1.0 && compression_speed > 0.25 * speed && compression_speed * speed > cells[i].internal_energy * 0.25)
        {
            double const hydro_time = 1.0 / std::max(-div_v, 1e-200);
            double const T_local = std::max(cells_cgs[i].temperature, 1.0);
            double const radiation_eq = CG::radiation_constant * std::pow(T_local, 4);
            double const planck_exchange = CG::speed_of_light * sigma_planck[i] * (radiation_eq - Er);
            double compton_exchange = 0.0;
            if(compton_on_)
            {
                double const Tr = std::pow(std::max(Er / CG::radiation_constant, 1e-200), 0.25);
                compton_exchange = 16.0 * sigma_s[i] * CG::boltzmann_constant * Er *
                                   (T_local - Tr) / (CG::electron_mass * CG::speed_of_light);
            }

            double const net_cooling_power = planck_exchange + compton_exchange;
            if(net_cooling_power > 0)
            {
                double const thermal_energy = std::max(cells_cgs[i].internal_energy * cells_cgs[i].density, 1e-200);
                double const cool_time = thermal_energy / net_cooling_power;
                double const target_cool_time = 2.0 * hydro_time;
                if(cool_time < target_cool_time)
                {
                    double const target_cooling_power = thermal_energy / target_cool_time;
                    double const opacity_scale = std::max(target_cooling_power / std::max(net_cooling_power, 1e-200), 1e-8);
                    sigma_planck[i] *= opacity_scale;
                    if(compton_on_)
                        sigma_s[i] *= opacity_scale;
                }
            }
        }
    }
}

void Diffusion::calculate_fleck_factor(
    Tessellation3D const& tess,
    std::vector<ComputationalCell3D> const& cells,
    double const dt
) const {
    auto const N = tess.GetPointNo();

    for(std::size_t i=0; i < N; ++i){
        fleck_factor[i] = GetSingleFleckFactor(cells[i], i, dt);

        if(fleck_factor[i] < 0){
            throw UniversalError("Negative fleck_factor")
                    .addEntry("fleck factor", fleck_factor[i])
                    .addEntry("cell ID", cells[i].ID);
        }
    }
}

void Diffusion::calculate_cell_diffusion_coefficients(
    Tessellation3D const& tess
) const 
{
    auto const N = tess.GetPointNo();
    
    for (std::size_t i=0; i < N; ++i){
        D[i] = D_coefficient_calcualtor.CalcDiffusionCoefficient(cells_cgs[i]);

        if(D[i] < 0.0){
            throw UniversalError("Negative Diffusion Coefficient")
                    .addEntry("Diffusion Coefficient", D[i])
                    .addEntry("cell ID", cells_cgs[i].ID);
        }
    }

#ifdef RICH_MPI
    MPI_exchange_data(tess, D, true);
#endif    
}

void Diffusion::fix_small_negative_Er(
    Tessellation3D const& tess,
    std::vector<ComputationalCell3D> const& cells
) const 
{
    int rank = 0;
#ifdef RICH_MPI
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif

    double max_Er = *std::max_element(new_Er.begin(), new_Er.end());

#ifdef RICH_MPI
    MPI_Allreduce(MPI_IN_PLACE, &max_Er, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
#endif

   struct MinErData {
        double value;
        int rank;
    };

    MinErData minErData = {1.0, rank};
    size_t min_index = -1;

    std::size_t const N = tess.GetPointNo();
    for(std::size_t i=0; i < N; ++i){
        if(new_Er[i] < 0.0 && std::abs(new_Er[i]) < 1e-9 * max_Er){
            new_Er[i] = std::min(1e-8 * max_Er, CG::radiation_constant * cells[i].temperature * cells[i].temperature * cells[i].temperature * cells[i].temperature);
        }

        if(new_Er[i] < minErData.value) {
            minErData.value = new_Er[i];
            min_index = i;
        }
    }

#ifdef RICH_MPI
		MPI_Allreduce(MPI_IN_PLACE, &minErData, 1, MPI_DOUBLE_INT, MPI_MINLOC, MPI_COMM_WORLD);
#endif

    if(minErData.value < 0) {
        if(rank == minErData.rank) {
            throw UniversalError("Negative Er! Minimal Value")
                    .addEntry("rank", minErData.rank)
                    .addEntry("Index", min_index)
                    .addEntry("Location", tess.GetMeshPoint(min_index))
                    .addEntry("cell data", cells[min_index]);
        }
    }
}

double Diffusion::dE_absorption_emission(
    Tessellation3D const& tess,
    std::size_t i,
    double const Er,
    double const temperature,
    double const dt_cgs
) const 
{
    double const volume = tess.GetVolume(i) * volume_scale_;
    
    return fleck_factor[i] * CG::speed_of_light * dt_cgs * sigma_planck[i] * (Er - Um(temperature)) * volume;
}

double Diffusion::dE_v_squared(
    Tessellation3D const& tess,
    std::size_t i,
    double const Er,
    Vector3D const& velocity_cgs,
    double const max_velocity_cgs,
    double const dt_cgs
) const {
    if (not hydro_on_) return 0.0;

    double const volume = tess.GetVolume(i) * volume_scale_;
    double const v_squared = std::min(max_velocity_cgs * max_velocity_cgs, ScalarProd(velocity_cgs, velocity_cgs));

    return -0.5 * (3 - R2[i]) * fleck_factor[i] * CG::speed_of_light * dt_cgs * sigma_planck[i] *  v_squared * Er / (CG::speed_of_light * CG::speed_of_light) * volume;
}

double Diffusion::dE_compton(
    Tessellation3D const& tess,
    std::size_t i,
    double const Er,
    double const temperature,
    double const old_Er,
    double const dt_cgs
) const
{
    if (not compton_on_) return 0.0;
    
    double const volume = tess.GetVolume(i) * volume_scale_;
    double const old_Tr = Trad(old_Er);
    
    double const pre_factor = fleck_factor[i] * dt_cgs * 4 * sigma_s[i] * CG::boltzmann_constant / (CG::electron_mass * CG::speed_of_light);
    
    double result = pre_factor * (old_Tr - temperature);
    double const theta = (fleck_factor[i] < 0.5 || std::abs(result) > 1e-3) ? 1 : 0.1;
    
    result =  pre_factor * volume * (Er * (old_Tr - temperature * (1 - theta)) - temperature * theta * old_Er);

    return result;
}

bool Diffusion::is_energy_invalid(Conserved3D const& extensive) const
{
    return extensive.Erad < 0 || extensive.internal_energy < 0 || !std::isfinite(extensive.internal_energy);
}

double Diffusion::dE_relativity(
    Tessellation3D const& tess,
    std::size_t i,
    std::vector<double> const& CG_result,
    double const dt_cgs,
    Vector3D& gradE
) const
{
    if(not hydro_on_) return 0.0;
    
    std::vector<size_t> neighbors;
    tess.GetNeighbors(i, neighbors);
    size_t const Nneigh = neighbors.size();
    face_vec faces = tess.GetCellFaces(i);
    Vector3D const point = tess.GetMeshPoint(i);
    Vector3D dummy_v;
    
    double const v_ratio = std::min(1.0, 0.05 * CG::speed_of_light / (fastabs(cells_cgs[i].velocity) + 1e-2));
    double total_relativity = 0;
    gradE = Vector3D(0, 0, 0);
    
    for(size_t j = 0; j < Nneigh; ++j)
    {
        size_t const neighbor_j = neighbors[j];
        Vector3D r_ij = point - tess.GetMeshPoint(neighbor_j);
        double const r_ij_size = abs(r_ij);
        r_ij *= 1.0 / r_ij_size;
        double Er_j = 0;
        if(tess.IsPointOutsideBox(neighbor_j))
            boundary_calc_.GetOutSideValues(tess, cells_cgs, i, neighbor_j, CG_result, Er_j, dummy_v);
        else
            Er_j = CG_result[neighbor_j];

        gradE += (0.5 * tess.GetArea(faces[j]) * (Er_j + CG_result[i])) * r_ij * area_scale_;
        double const momentum_term = (0.5 * dt_cgs * cell_flux_limiter[i] * tess.GetArea(faces[j]) * area_scale_ * ScalarProd(cells_cgs[i].velocity, r_ij) * (Er_j + CG_result[i]) / 3);
        double const relativity_term = -v_ratio * momentum_term * 2 * 3 * sigma_planck[i] * D[i] / CG::speed_of_light / energy_scale_;
        total_relativity += fleck_factor[i] * relativity_term;
    }
    
    return total_relativity;
}

std::tuple<Vector3D, double, double> Diffusion::dP_and_dE_momentum(
    std::size_t i,
    Vector3D const& gradE,
    double const dt_cgs,
    Vector3D const& momentum,
    double const mass
) const
{
    double const old_Ek = 0.5 * ScalarProd(momentum, momentum) / mass;
    Vector3D const dP = (cell_flux_limiter[i] * dt_cgs / 3) * gradE / momentum_scale_;
    double const Erad_dE = ScalarProd(dP, momentum) / mass;
    Vector3D const new_momentum = momentum + dP;
    double const new_Ek = 0.5 * ScalarProd(new_momentum, new_momentum) / mass;
    double const dE = -new_Ek + old_Ek + Erad_dE;
    return {dP, dE, Erad_dE};
}

void Diffusion::compute_equilibrium_from_energy_sum(
    Tessellation3D const& tess,
    std::size_t i,
    std::vector<ComputationalCell3D> const& cells,
    std::vector<Conserved3D>& extensives,
    double const volume,
    std::vector<double> const& CG_result,
    double const dt_cgs
) const
{
    
    if(hydro_on_)
    {
        // Call dE_relativity only to calculate gradE for momentum calculation below
        Vector3D gradE;
        dE_relativity(tess, i, CG_result, dt_cgs, gradE);

        auto const [dP, dE_mom, Erad_dE] = dP_and_dE_momentum(i, gradE, dt_cgs, extensives[i].momentum, extensives[i].mass);
        extensives[i].momentum += dP;
        extensives[i].Erad += dE_mom;
    }

    auto const internal_and_Erad_tot_cgs = (extensives[i].Erad + extensives[i].internal_energy) * energy_scale_;

    if(internal_and_Erad_tot_cgs < 0.0){
        throw UniversalError("Diffusion::compute_equilibrium_from_energy_sum: Negative internal and Erad total")
                .addEntry("internal_and_Erad_tot_cgs", internal_and_Erad_tot_cgs)
                .addEntry("cell ID", cells[i].ID)
                .addEntry("cell", cells[i]);
    } 

    auto func = [&](double const equilibrium_temperature){
        double const internal_energy_cgs = extensives[i].mass * eos_.dT2e(
            cells[i].density, 
            equilibrium_temperature,
            cells[i].tracers,
            ComputationalCell3D::tracerNames
        ) * energy_scale_;

        return Um(equilibrium_temperature) * volume + internal_energy_cgs - internal_and_Erad_tot_cgs;
    };

    boost::math::tools::eps_tolerance<double> tol(26);
    std::uintmax_t it = 150;

    double const T_right_bracket = Trad(internal_and_Erad_tot_cgs / volume) * 1.2;
    double const T_left_bracket = T_right_bracket * 1e-2;
    auto r = boost::math::tools::bisect(
        func,
        T_left_bracket,
        T_right_bracket,
        tol,
        it
    );

    double const equilibrium_temperature = 0.5 * (r.first + r.second);
    extensives[i].internal_energy = eos_.dT2e(
        cells[i].density,
        equilibrium_temperature,
        cells[i].tracers,
        ComputationalCell3D::tracerNames
    ) * extensives[i].mass;

    extensives[i].Erad = tess.GetVolume(i) * Um(equilibrium_temperature) / energy_density_scale_;

    extensives[i].energy = extensives[i].internal_energy;
    if(hydro_on_){
        extensives[i].energy += ScalarProd(extensives[i].momentum, extensives[i].momentum) / (2 * extensives[i].mass);
    }
}

void Diffusion::print_postcg1_debug(
    std::size_t i,
    std::vector<ComputationalCell3D> const& cells,
    std::vector<Conserved3D> const& extensives,
    std::vector<double> const& CG_result,
    std::vector<double> const& full_CG_result,
    double T,
    double dE,
    double old_e_therm,
    double e_emitt,
    double e_v2,
    double e_absorb_emitt,
    double compton_term,
    double old_Tr
) const
{
    std::cout   << "Negative internal energy in postcg1, " 
                << extensives[i].internal_energy 
                << " ID " << cells[i].ID 
                << " T " << T << "\n"
                << " CG_result " << CG_result[i] 
                << " old Er " << cells[i].Erad * cells[i].density * energy_density_scale_
                << " v " << fastabs(cells[i].velocity) 
                << " mass " << extensives[i].mass
                << " dE " << dE 
                << " R2 " << R2[i] 
                << " old_e_therm " << old_e_therm << std::endl;
    
    std::cout   << cells[i] << std::endl;
    std::cout   << extensives[i] << std::endl;

    std::cout   << "max emitt " << -e_emitt
                << " full_CG_result " << full_CG_result[i]
                << " relativity " << -e_v2
                << " fleck " << fleck_factor[i]
                << " sigma_planck " << sigma_planck[i]
                << " other dE " << e_absorb_emitt + e_v2 
                << " density " << cells[i].density
                << " compton_term " << compton_term 
                << " old_Tr " << old_Tr << std::endl;
}

void Diffusion::print_postcg2_debug(
    std::size_t i,
    Tessellation3D const& tess,
    std::vector<ComputationalCell3D> const& cells,
    std::vector<Conserved3D> const& extensives,
    std::vector<double> const& CG_result,
    std::vector<double> const& full_CG_result,
    double T,
    double max_v,
    double dt_cgs,
    Vector3D const& dP,
    double Erad_dE,
    double e_absorb,
    double e_emitt,
    double e_v2,
    double total_relativity,
    double etherm_mid,
    Vector3D const& gradE,
    Vector3D const& CM,
    Vector3D const& point,
    std::vector<size_t> const& neighbors,
    face_vec const& faces
) const
{
    std::cout   << "Negative internal energy is postcg2, " 
                << extensives[i].internal_energy
                << " ID " << cells[i].ID 
                << " T " << T 
                << " CG_result " << CG_result[i]
                << " full_CG_result " << full_CG_result[i] 
                << " v " << fastabs(cells[i].velocity) 
                << " sigma_planck " << sigma_planck[i]
                << " sigma_r " << CG::speed_of_light / (3 * D[i]) 
                << " E_init " << cells[i].Erad*cells[i].density* mass_scale_ / (time_scale_ * time_scale_ * length_scale_)
                << " volume " << tess.GetVolume(i)
                << " Erad " << extensives[i].Erad
                << " mass " << extensives[i].mass
                << " dP " << dP.x << "," << dP.y << "," << dP.z 
                << " momentum " << extensives[i].momentum.x << "," << extensives[i].momentum.y << "," << extensives[i].momentum.z << "\n"
                << "Erad_dE " << Erad_dE 
                << " cell_flux_limiter " << cell_flux_limiter[i] 
                << " e_absorb " << e_absorb 
                << " e_emitt "<< e_emitt 
                << " e_v2 " << e_v2 << "\n"
                << "total relativity " << total_relativity 
                << " etherm_mid " << etherm_mid
                << " fleck_factor " << fleck_factor[i]
                << " gradE " << gradE * (1.0 / tess.GetVolume(i)) << std::endl;

    size_t const Nneigh = neighbors.size();
    Vector3D dummy_v;
    
    for(size_t j = 0; j < Nneigh; ++j)
    {
        size_t const neighbor_j = neighbors[j];
        Vector3D r_ij = point - tess.GetMeshPoint(neighbor_j);
        double const r_ij_size = abs(r_ij);
        r_ij *= 1.0 / r_ij_size;
        double Er_j = 0;
        if(tess.IsPointOutsideBox(neighbor_j))
            boundary_calc_.GetOutSideValues(tess, cells_cgs, i, neighbor_j, CG_result, Er_j, dummy_v);
        else
            Er_j = CG_result[neighbor_j];

        Vector3D const cm_ij = CM - tess.GetCellCM(neighbor_j);
        Vector3D const grad_E = r_ij * ScalarProd(r_ij, cm_ij) * (1.0 / (length_scale_ * ScalarProd(cm_ij, cm_ij)));   
        double mid_D = 0.5 * (D[neighbor_j] + D[i]);
        double const flux_limiter_face = flux_limiter_ ? CalcSingleFluxLimiter(grad_E * (CG_result[i] - Er_j), mid_D, 0.5 * (CG_result[i] + Er_j)) : 1;
        
        double const max_local_v = std::min(max_v, std::max(-max_v, ScalarProd(cells_cgs[i].velocity, r_ij)));
        double const v_ratio = max_local_v / ScalarProd(cells_cgs[i].velocity, r_ij);
        
        double const momentum_term = (0.5 * dt_cgs * cell_flux_limiter[i] * tess.GetArea(faces[j]) * area_scale_ * ScalarProd(cells_cgs[i].velocity, r_ij) * (Er_j + CG_result[i]) / 3);
        double const relativity_term = -v_ratio * fleck_factor[i] * momentum_term * 2 * 3 * sigma_planck[i] * D[i] / CG::speed_of_light / energy_scale_;
        
        std::cout   << "relativity_term " << relativity_term
                    << " flux_limiter_face " << flux_limiter_face
                    << " Er_j " << Er_j
                    << " Er_j_init " << cells[neighbor_j].density * cells[neighbor_j].Erad * energy_density_scale_
                    << " ID " << cells[neighbor_j].ID
                    << " v_ratio " << v_ratio
                    << " Area " << tess.GetArea(faces[j]) << std::endl;
    }

    std::cout   << "Negative internal energy is postcg2, " 
                << extensives[i].internal_energy 
                << " ID " << cells[i].ID 
                << " T " << T 
                << " CG_result " << CG_result[i] 
                << " v " << fastabs(cells[i].velocity) 
                << " mass " << extensives[i].mass << std::endl;
}

bool Diffusion::iterations(
    double const tolerance, 
    int& total_iters, 
    Tessellation3D const& tess, 
    std::vector<ComputationalCell3D>& cells,
    std::vector<Conserved3D>& extensives,
    double const dt,
    double const time
) const 
{   
    int rank = 0;
    int ws = 1;
#ifdef RICH_MPI
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ws);
#endif

    std::fill(
        fleck_factor.begin(), fleck_factor.end(), 
        1.0);

    int const max_iterations = 20;
    double const outer_iterations_eps = 1e3*std::sqrt(tolerance);

    load_cells_cgs(tess, cells);
    
    calculate_planck_absorption_coefficient(tess);
    calculate_scattering_coefficient(tess);
    apply_opacity_limiters(tess, cells, dt);
    calculate_cell_diffusion_coefficients(tess);

    auto const N = tess.GetPointNo();
    
    double total_N_all_processes = static_cast<double>(N);
    #ifdef RICH_MPI
        MPI_Allreduce(MPI_IN_PLACE, &total_N_all_processes, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    #endif

    double error_sie = 0.0;
    double error_Um = 0.0;
    double error = 0.0;
    std::vector<double> error_per_cell(N, 1e200);
    int iter = 0;
    use_new_Er_for_x0 = false;
    for(iter=0; iter < max_iterations; ++iter){
        int num_of_inner_iters = 0;
        // solve for E(Um^k)
        bool good_end = false;
        new_Er = CG::BiCGSTAB(tolerance, num_of_inner_iters, tess, cells, dt, *this, time, new_Er_full, good_end);
        use_new_Er_for_x0 = true;

        total_iters += num_of_inner_iters;

        if(not good_end) {
            use_new_Er_for_x0 = false;
            return false;
        }

        std::vector<double> old_internal_energy(N, 0.0);
        std::vector<double> old_Um(N, 0.0);
        for(std::size_t i=0; i < N; ++i){
            old_Um[i] = Um(cells[i].temperature);
            old_internal_energy[i] = cells[i].internal_energy;
        }

        update_energy_iterations(
            tess,
            cells,
            extensives,
            dt,
            new_Er_full,
            new_Er,
            error,
            error_per_cell
        );

        double max_internal_energy = *std::max_element(old_internal_energy.begin(), old_internal_energy.end());
        double max_Um = *std::max_element(old_Um.begin(), old_Um.end());
        
        #ifdef RICH_MPI
        MPI_Allreduce(MPI_IN_PLACE, &max_internal_energy, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, &max_Um, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        #endif

        error_Um = 0.0;
        error_sie = 0.0;
        for(std::size_t i=0; i<N; ++i){
            error_Um += std::abs(Um(cells[i].temperature) - old_Um[i]) / (0.5*(Um(cells[i].temperature) + old_Um[i]) + 1e-3*max_Um);
            error_sie = std::max(error_sie, std::abs(cells[i].internal_energy - old_internal_energy[i]) / old_internal_energy[i]);
        }

        #ifdef RICH_MPI
        MPI_Allreduce(MPI_IN_PLACE, &error_Um, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, &error_sie, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        #endif

        error_Um /= total_N_all_processes;
        
        
        if (rank == 0)
            std::cout   << "error_nr: " << error
                        << ", error_sie: " << error_sie
                        << ", error_Um: " << error_Um 
                        << ", iteration: " << iter 
                        << ", inner_iterations: " << num_of_inner_iters 
                        << ", total_iters: " << total_iters << std::endl;

        if(error_Um < 1e-3 && error < 1e-1){
            break;
        }
    }

    
    if (rank == 0)
            std::cout   << "error_sie: " << error_sie << "\n"
        << "error_Um: " << error_Um << "\n"
        << "iteration: " << iter << std::endl;

    if(iter == max_iterations){
        if (rank == 0)
            std::cout << "Max iterations reached" << std::endl;
        return false;
    }

    bool good_end = false;
    int num_of_inner_iters = 0;
    new_Er = CG::BiCGSTAB(tolerance, num_of_inner_iters, tess, cells, dt, *this, time, new_Er_full, good_end);

    total_iters += num_of_inner_iters;

    if(not good_end) {
        use_new_Er_for_x0 = false;
        return false;
    }

    PostCG(tess, extensives, dt, cells, new_Er_full, new_Er);
    
    use_new_Er_for_x0 = false;

    return true;
}

bool Diffusion::update_energy_iterations(
    Tessellation3D const& tess,
    std::vector<ComputationalCell3D>& cells,
    std::vector<Conserved3D>& extensives,
    double const dt,
    std::vector<double>& Er_full,
    std::vector<double>& Er,
    double& newton_raphson_error,
    std::vector<double>& error_per_cell
) const
{
    double const max_v = 0.1 * CG::speed_of_light * velocity_scale_;
    Vector3D gradE_dummy;

    auto const N = tess.GetPointNo();
    double const dt_cgs = dt * time_scale_;

    double nr_error_tmp = 0.0;
    for(std::size_t i=0; i < N; ++i){
        double const E_old = extensives[i].internal_energy;
        double const temperature_k = cells[i].temperature;
        
        double const dE_abs_emiss = dE_absorption_emission(
            tess,
            i,
            Er[i],
            temperature_k,
            dt_cgs
        ) / energy_scale_;

        double const dE_v2 = dE_v_squared(
            tess,
            i,
            Er[i],
            cells_cgs[i].velocity,
            max_v,
            dt_cgs
        ) / energy_scale_;

        double const dE_compton_term = dE_compton(
            tess,
            i,
            Er[i],
            cells_cgs[i].temperature,
            cells_cgs[i].Erad * cells_cgs[i].density,
            dt_cgs
        ) / energy_scale_;

        double const dE_relativity_term = dE_relativity(
            tess,
            i,
            Er,
            dt_cgs,
            gradE_dummy
        ) / energy_scale_;

        double const dE_total = dE_abs_emiss + dE_v2 + dE_relativity_term + dE_compton_term;

        double Cv = eos_.dT2cv(cells[i].density, temperature_k, cells[i].tracers, ComputationalCell3D::tracerNames) * energy_density_scale_;
        double const beta = radiation_cv(temperature_k) / Cv;
        double const cdt = CG::speed_of_light * dt_cgs;

        double const E_prev = cells[i].internal_energy * extensives[i].mass;
        
        double const cdtkp = cdt*sigma_planck[i];

        double const compton_in_fleck_term = compton_on_ ? -dt_cgs*4.0*sigma_s[i]*CG::boltzmann_constant/(CG::electron_mass*CG::speed_of_light) * (Trad(cells_cgs[i].Erad * cells_cgs[i].density) - cells_cgs[i].temperature) *beta*cdtkp/(1.0+cdtkp) : 0.0;
        
        double const fleck_like_factor = 1.0 / (1.0 + beta*cdtkp/(1.0+cdtkp) + compton_in_fleck_term);
        double const dE_newton_raphson = (E_old + dE_total - E_prev) * fleck_like_factor;
        
        double const err_cell = std::abs(E_old + dE_total - E_prev) / E_prev;
        
        error_per_cell[i] = err_cell;
        nr_error_tmp = std::max(nr_error_tmp, err_cell); 

        double const E_new = E_prev + dE_newton_raphson;
        
        cells[i].internal_energy = std::max(E_new / extensives[i].mass, 1e-20);
        
        cells[i].temperature = eos_.de2T(cells[i].density, cells[i].internal_energy, cells[i].tracers, ComputationalCell3D::tracerNames);
    }

    #ifdef RICH_MPI
    MPI_Allreduce(MPI_IN_PLACE, &nr_error_tmp, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    #endif
    
    newton_raphson_error = nr_error_tmp;
    
    return true;
}