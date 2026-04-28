#include "Diffusion.hpp"
#include "misc/memory_debug.hpp"
#include "misc/memory_profile.hpp"
#include "misc/utils.hpp"
#include <boost/math/special_functions.hpp>

#ifdef RICH_MPI
#include "mpi/mpi_commands.hpp"
#endif

namespace CG
{
     // LP flux limiter taken from "EQUATIONS AND ALGORITHMS FOR MIXED-FRAME FLUX-LIMITED DIFFUSION RADIATION HYDRODYNAMICS"
    double CalcSingleFluxLimiter(Vector3D const& grad, double const D, double const cell_value)
    {
        double const R = std::max(3 * abs(grad) * D / (cell_value * CG::speed_of_light + 1e-200), 1e-200);
        
        // series expansion
        if(R < 1e-2) return 1 - R * R / 15 + 2 * R * R * R * R /315;
        
        return 3 * (1.0 / std::tanh(R) - 1.0 / R) / R;
    }

    double FleckFactor(double const dt, double const beta, double const sigma_a)
    {
        return 1.0 / (1 + beta * dt * sigma_a * CG::speed_of_light);
    }

    double FleckFactorCompton(double const dt, double const beta, double const sigma_a, double const sigma_s, double const Erad, double const Cv)
    {
        return 1.0 / (1 + beta * dt * sigma_a * CG::speed_of_light + dt * 16 * sigma_s * CG::boltzmann_constant * Erad / (CG::electron_mass * CG::speed_of_light * Cv));
    }
}

Diffusion::Diffusion(OpacityCalculator const& D_coefficient_calc, 
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
                                             cells_temp(),
                                             extensives_temp(),
                                             cooling_time_limiter_on_(cooling_time_limiter_on) {}

double Diffusion::GetSingleFleckFactor(ComputationalCell3D const& cell, double const dt)const
{
    ComputationalCell3D cell_cgs(cell);
    cell_cgs.density *= mass_scale_ / (length_scale_ * length_scale_ * length_scale_);
    double const Er = cell.Erad * cell.density *  mass_scale_ / (time_scale_ * time_scale_ * length_scale_);
    double sigma_planck = D_coefficient_calcualtor.CalcPlanckOpacity(cell_cgs);
    sigma_planck = std::min(sigma_planck, 100.0 / (CG::speed_of_light * dt * time_scale_)); // avoid too large opacities
    double const sigma_s = D_coefficient_calcualtor.CalcScatteringOpacity(cell_cgs);
    double const T = cell.temperature;
    double Cv = eos_.dT2cv(cell.density, T, cell.tracers, ComputationalCell3D::tracerNames);
    double const energy_ratio = Cv * cell.temperature / (cell.internal_energy * cell.density);
    Cv *= mass_scale_ / (time_scale_ * time_scale_ * length_scale_);
    double const beta = std::max(1.0, 0.5 * energy_ratio) * 4 * CG::radiation_constant * T * T * T / Cv;
    return compton_on_ ? FleckFactorCompton(dt * time_scale_, beta, sigma_planck, sigma_s, Er, Cv) : FleckFactor(dt * time_scale_, beta, sigma_planck);
}

bool Diffusion::prestep(Tessellation3D const& tess,
                        std::vector<ComputationalCell3D> const& cells) const {
    MEMORY_PROFILE_SCOPE("diffusion prestep");
    auto const N = tess.GetPointNo();
    
    sigma_planck.resize(N, 0.0);
    conditional_shrink(sigma_planck);
    sigma_s.resize(N, 0.0);
    conditional_shrink(sigma_s);
    fleck_factor.resize(N, 0.0);
    conditional_shrink(fleck_factor);
    D.resize(N, 0.0);
    conditional_shrink(D);
    R2.resize(N, 0.0);
    conditional_shrink(R2);
    cell_flux_limiter.resize(N, 0.0);
    conditional_shrink(cell_flux_limiter);

    new_Er.resize(N, 0.0);
    conditional_shrink(new_Er);
    new_Er_full.resize(N, 0.0);
    conditional_shrink(new_Er_full);
    old_Er.resize(N, 0.0);
    conditional_shrink(old_Er);

    cells_temp.resize(N);
    conditional_shrink(cells_temp);
    extensives_temp.resize(N);
    conditional_shrink(extensives_temp);

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
    }

    return true;
}

bool Diffusion::poststep() const {    
    cells_temp.clear();
    extensives_temp.clear();

    return false;
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
    MEMORY_PROFILE_SCOPE("diffusion step");
    
    int rank = 0;
#ifdef RICH_MPI
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif
    
    extensives_temp = extensives;
    cells_temp = cells;

    std::size_t const N = tess.GetPointNo();
    bool good_end = false;
    new_Er = CG::BiCGSTAB(tolerance, total_iters, tess, cells, dt, *this, time, new_Er_full, good_end, cg_workspace_);
    MEMORY_DEBUG_PRINT("diffusion: after BiCGSTAB");
    if(not good_end)
        return false;
    
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
            std::cout << "Negative Er! Rank: " << minErData.rank << ", Index: " << min_index <<" location "<<tess.GetMeshPoint(min_index)<<" "<<cells[min_index]<<std::endl;
        }

        return false;
    }

    try {
        PostCG(tess, extensives, dt, cells, new_Er, new_Er_full);
        MEMORY_DEBUG_PRINT("diffusion: after PostCG");
    } catch(UniversalError const& eo) {
        if(rank == 0){
            std::cout<< "PostCG Exception:" << std::endl;
            std::cout<< eo.getErrorMessage() << std::endl;
        }
        
        extensives = std::move(extensives_temp);
        cells = cells_temp;
        return false;
    }

    return true;
}

void Diffusion::BuildMatrix(Tessellation3D const& tess, mat& A, size_t_mat& A_indeces, std::vector<ComputationalCell3D> const& cells,
    double const dt, std::vector<double>& b, std::vector<double>& x0, double const current_time) const
{
    double const max_v = 0.1 * CG::speed_of_light * length_scale_ / time_scale_;
    int rank = 0;
#ifdef RICH_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif
    size_t const Nlocal = tess.GetPointNo();
    std::vector<ComputationalCell3D> cells_cgs(cells);
    for(size_t i = 0; i < Nlocal; ++i)
    {
        cells_cgs[i].density *= mass_scale_ / (length_scale_ * length_scale_ * length_scale_);
        cells_cgs[i].Erad *= length_scale_ * length_scale_ / (time_scale_ * time_scale_);
        cells_cgs[i].Erad_dt *= length_scale_ * length_scale_ / (time_scale_ * time_scale_ * time_scale_);
        cells_cgs[i].Erad_dt_dt *= length_scale_ * length_scale_ / (time_scale_ * time_scale_ * time_scale_ * time_scale_);
        cells_cgs[i].velocity *= length_scale_ / time_scale_;
     }
#ifdef RICH_MPI
	MPI_exchange_data(tess, cells_cgs, true);	
#endif
    b.resize(Nlocal, 0);
    x0.resize(Nlocal, 0);
    D.resize(Nlocal);
    fleck_factor.resize(Nlocal);
    sigma_planck.resize(Nlocal);
    sigma_s.resize(Nlocal);
    std::vector<size_t> neighbors;
    face_vec faces;
    std::vector<size_t> zero_indeces;
    size_t const Nzero = zero_cells_.size();
    for(size_t i = 0; i < Nzero; ++i)
        zero_indeces.push_back(binary_index_find(ComputationalCell3D::stickerNames, zero_cells_[i]));
    double const zero_value = 1e-10;
    std::vector<double> new_Er(Nlocal, 0), Er_for_limit(Nlocal, 0);
    for(size_t i = 0; i < Nlocal; ++i)
    {
        double const volume = tess.GetVolume(i) * length_scale_ * length_scale_ * length_scale_;
        double const cell_width = std::max(tess.GetWidth(i) * length_scale_, 1e-200);
        bool set_to_zero = false;
        for(size_t j = 0; j < Nzero; ++j)
            if(cells_cgs[i].stickers[zero_indeces[j]])
                set_to_zero = true;
        double const Er = cells_cgs[i].Erad * cells_cgs[i].density * (set_to_zero ? zero_value : 1);
        new_Er[i] = Er;

        D[i] = D_coefficient_calcualtor.CalcDiffusionCoefficient(cells_cgs[i]);
        if(D[i] < 0)
            throw UniversalError("Negative D");
        double const T = cells_cgs[i].temperature;
        sigma_planck[i] = D_coefficient_calcualtor.CalcPlanckOpacity(cells_cgs[i]);
        if(sigma_planck[i] < 0)
            throw UniversalError("Negative sigma_planck");
        sigma_planck[i] = std::min(sigma_planck[i], 10.0 / (CG::speed_of_light * dt * time_scale_)); // avoid too large opacities
        sigma_s[i] = D_coefficient_calcualtor.CalcScatteringOpacity(cells_cgs[i]);

        // Optional cooling limiter for under-resolved post-shock cooling layers.
        if(cooling_time_limiter_on_)
        {
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

                // Include both cooling and heating terms: only limit when net effect is cooling.
                double const net_cooling_power = planck_exchange + compton_exchange;
                if(net_cooling_power > 0)
                {
                    double const thermal_energy = std::max(cells_cgs[i].internal_energy * cells_cgs[i].density , 1e-200);
                    double const cool_time = thermal_energy / net_cooling_power;
                    double const target_cool_time = 2.0 * hydro_time;
                    if(cool_time < target_cool_time)
                    {
                        // Preserve relative Planck/Compton term weighting while reducing net cooling.
                        double const target_cooling_power = thermal_energy / target_cool_time;
                        double const opacity_scale = std::max(target_cooling_power / std::max(net_cooling_power, 1e-200), 1e-8);
                        sigma_planck[i] *= opacity_scale;
                        if(compton_on_)
                            sigma_s[i] *= opacity_scale;
                    }
                }
            }
        }

        double Cv = eos_.dT2cv(cells[i].density, T, cells[i].tracers, ComputationalCell3D::tracerNames);
        double const energy_ratio = Cv * cells[i].temperature / (cells[i].internal_energy * cells[i].density);
        Cv *= mass_scale_ / (time_scale_ * time_scale_ * length_scale_);
        double const beta = std::max(1.0, 0.5 * energy_ratio) * 4 * CG::radiation_constant * T * T * T / Cv;
        fleck_factor[i] = compton_on_ ? FleckFactorCompton(dt * time_scale_, beta, sigma_planck[i], sigma_s[i], Er, Cv) : FleckFactor(dt * time_scale_, beta, sigma_planck[i]);
        if(fleck_factor[i] < 0)
            throw UniversalError("Negative fleck_factor");
        b[i] = volume * Er;
        double const Um = CG::radiation_constant * T * T * T * T;
        if(fleck_factor[i] < 0.8 && Um > Er)
        {
            double const prefactor = fleck_factor[i] * dt * CG::speed_of_light * sigma_planck[i];
            x0[i] = (Er + prefactor * Um) / (1 + prefactor);
        }
        else
            x0[i] = std::min(2 * Er, std::max(0.5 * Er, Er + cells_cgs[i].Erad_dt * cells_cgs[i].density * dt * time_scale_  + 0.5 * cells_cgs[i].Erad_dt_dt * cells_cgs[i].density * dt * dt * time_scale_ * time_scale_));//std::max(Er + 0.5 * std::min(fleck_factor[i] * dt * sigma_planck[i] * CG::speed_of_light * time_scale_, 1.0) * (CG::radiation_constant * T * T * T * T - Er), 0.25 * Er);
        b[i] += volume * fleck_factor[i] * dt * CG::speed_of_light * sigma_planck[i] * T * T * T * T * CG::radiation_constant * time_scale_;
        Er_for_limit[i] = std::min(Er, std::max(1e-5 * Er, Er + dt * time_scale_ * fleck_factor[i] * sigma_planck[i] * CG::speed_of_light * (CG::radiation_constant * T * T * T * T - Er)));
    }
#ifdef RICH_MPI
    MPI_exchange_data(tess, D, true);
#endif
    size_t max_neigh = 0;
    // Find maximum number of neighbors and allocate data
    for(size_t i = 0; i < Nlocal; ++i)
        max_neigh = std::max(max_neigh, tess.GetCellFaces(i).size());
    ++max_neigh;
    A.clear();
    A.resize(Nlocal);
    A_indeces.clear();
    A_indeces.resize(Nlocal);
    R2.clear();
    R2.resize(Nlocal, 0);
    conditional_shrink(R2);
    cell_flux_limiter.clear();
    cell_flux_limiter.resize(Nlocal, 0);
    conditional_shrink(cell_flux_limiter);

    // Build the matrix
    for(size_t i = 0; i < Nlocal; ++i)
    {
        A_indeces[i].push_back(i);
        double const volume = tess.GetVolume(i) * length_scale_ * length_scale_ * length_scale_;
        double const T = cells_cgs[i].temperature;
        A[i].push_back(volume * (1 + fleck_factor[i] * dt * CG::speed_of_light * sigma_planck[i] * time_scale_));
        if(compton_on_ && cells[i].tracers[1] > 0.5)
        {
            double const Tr = std::pow(new_Er[i] / CG::radiation_constant, 0.25);
	        double const pre_factor = fleck_factor[i] * dt * time_scale_ * 4 * sigma_s[i] * CG::boltzmann_constant / (CG::electron_mass * CG::speed_of_light);
            double const compton_term = pre_factor * (Tr - T);
            double const theta = (fleck_factor[i] < 0.5 || std::abs(compton_term) > 1e-3) ? 1 : 0.1;
            A[i][0] += pre_factor * (Tr - (1 - theta) * T) * volume;
            b[i] += pre_factor * volume * theta * T * new_Er[i];
        }
        if(A[i][0] < 0)
	        std::cout<<"Negative A in matrix build, density "<<cells_cgs[i].density<<" T "<<cells_cgs[i].temperature<<" fleck "<<fleck_factor[i]<<
	        " sig_P "<<sigma_planck[i]<<" sig_s "<<sigma_s[i]<<" dt "<<dt * time_scale_<<" Erad "<<cells_cgs[i].Erad * cells_cgs[i].density<<" compton term "<<fleck_factor[i] * dt * time_scale_ * 4 * sigma_s[i] * CG::boltzmann_constant / (CG::electron_mass * CG::speed_of_light)<<std::endl;
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
        double const volume = tess.GetVolume(i) * length_scale_ * length_scale_ * length_scale_;
        faces = tess.GetCellFaces(i);
        tess.GetNeighbors(i, neighbors);
        size_t const Nneigh = neighbors.size();
        Vector3D const CM = tess.GetCellCM(i);
        Vector3D const point = tess.GetMeshPoint(i);
        gradE[i] = Vector3D(0, 0, 0) ;
        double const Dcell = D[i];
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
                    
                    double const T1 = cells_cgs[i].temperature;
                    double const T2 = cells_cgs[neighbor_j].temperature;
                    double const maxT = std::pow(0.5 * (pow<4>(T1) + pow<4>(T2)), 0.25);
                    cells_cgs[i].temperature = maxT;
                    double const D1 =  D_coefficient_calcualtor.CalcDiffusionCoefficient(cells_cgs[i]);
                    cells_cgs[i].temperature = T1;
                    cells_cgs[neighbor_j].temperature = maxT;
                    double const D2 =  D_coefficient_calcualtor.CalcDiffusionCoefficient(cells_cgs[neighbor_j]);
                    cells_cgs[neighbor_j].temperature = T2;
                    double mid_D = 2 * D1 * D2 / (D1 + D2);

                    // double mid_D = 0.5 * (D[neighbor_j] + Dcell);
                    double const grad_magnitude = std::max(std::numeric_limits<double>::min() * 1e40, std::abs(fastabs(grad_E) * (Er - Er_j)));
                    double grad_factor = 1;
                    max_neighbor_R[i] = std::max(max_neighbor_R[i], max_R[neighbor_j]);
                    if(grad_magnitude < 0.15 * (max_R[i] + max_R[neighbor_j]))
                        grad_factor = 0.15 * (max_R[i] + max_R[neighbor_j]) / grad_magnitude;
                    double const flux_limiter = flux_limiter_ ? CalcSingleFluxLimiter(grad_E * ((Er - Er_j) * grad_factor), mid_D, 0.5 * (Er + Er_j)) : 1;
                    mid_D *= flux_limiter;
                    double const flux = ((self_zero || set_to_zero) ? tess.GetArea(faces[j]) * dt * CG::speed_of_light * 0.5 : ScalarProd(grad_E, r_ij) * tess.GetArea(faces[j]) * dt * mid_D) * length_scale_ * length_scale_ * time_scale_; 
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
                    boundary_calc_.SetBoundaryValues(tess, i, neighbor_j, dt * time_scale_, cells_cgs, tess.GetArea(faces[j]) * length_scale_ * length_scale_, A[i][0], b[i], faces[j]);
                boundary_calc_.GetOutSideValues(tess, cells_cgs, i, neighbor_j, new_Er, Er_j, dummy_v);
            }
            gradE[i] += r_ij * (tess.GetArea(faces[j]) * 0.5 * (Er + Er_j) * length_scale_ * length_scale_);
        }
    }
    for(size_t i = 0; i < Nlocal; ++i)
    {
        double const volume = tess.GetVolume(i) * length_scale_ * length_scale_* length_scale_;
        gradE[i] *= -1.0 / volume;
        faces = tess.GetCellFaces(i);
        tess.GetNeighbors(i, neighbors);
        size_t const Nneigh = neighbors.size();
        Vector3D const point = tess.GetMeshPoint(i);
        double const Dcell = D[i];
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
        double const flux_limiter = flux_limiter_ ? CalcSingleFluxLimiter(grad_for_limiter, Dcell, Er_for_limit[i]) : 1;
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
                // double mid_D = 0.5 * (D[neighbor_j] + Dcell);
                // double const Er_j = cells_cgs[neighbor_j].Erad * cells_cgs[neighbor_j].density;
                // double const flux_limiter_face = flux_limiter_ ? CalcSingleFluxLimiter(grad_E * (Er - Er_j), mid_D, 0.5 * (Er + Er_j)) : 1;

                // double const T1 = cells_cgs[i].temperature;
                // double const T2 = cells_cgs[neighbor_j].temperature;
                // double const maxT = std::max(T1, T2);
                // cells_cgs[i].temperature = maxT;
                // double const D1 =  D_coefficient_calcualtor.CalcDiffusionCoefficient(cells_cgs[i]);
                // cells_cgs[i].temperature = T1;
                // cells_cgs[neighbor_j].temperature = maxT;
                // double const D2 =  D_coefficient_calcualtor.CalcDiffusionCoefficient(cells_cgs[neighbor_j]);
                // cells_cgs[neighbor_j].temperature = T2;
                // double mid_D = 2 * D1 * D2 / (D1 + D2);
                double const mid_D = Dcell;
                double const momentum_relativity_term = -0.5 * dt * flux_limiter * tess.GetArea(faces[j]) * (v_ratio * fleck_factor[i] * 2 * 3 * sigma_planck[i] * mid_D / CG::speed_of_light - 1) * length_scale_ * length_scale_ * time_scale_
                    * ScalarProd(cells_cgs[i].velocity, r_ij) / 3;
                A[i][0] += momentum_relativity_term;
                auto it = std::find(A_indeces[i].begin(), A_indeces[i].end(), neighbor_j);
                if(it == A_indeces[i].end())
                    throw UniversalError("Key not equal in diffusion");
                size_t const neigh_counter = static_cast<size_t>(it - A_indeces[i].begin());
                if(A_indeces[i][neigh_counter] != neighbor_j)
                    throw UniversalError("Key not equal value in diffusion");
                A[i][neigh_counter] += momentum_relativity_term;
            }
            else
                boundary_calc_.SetMomentumTermBoundary(tess, i, neighbor_j, dt * time_scale_, cells_cgs[i],
                    tess.GetArea(faces[j]) * length_scale_ * length_scale_, A[i][0], b[i], faces[j], fleck_factor[i],
                    flux_limiter, Dcell, sigma_planck[i]);
        }
        R2[i] = flux_limiter_ ? flux_limiter / 3 + boost::math::pow<2>(flux_limiter * abs(gradE[i]) * Dcell / (CG::speed_of_light * Er)) : 1.0 / 3.0;
        A[i][0] -= volume * fleck_factor[i] * dt * 0.5 * (3 - R2[i]) * sigma_planck[i] * std::min(0.01 * CG::speed_of_light * CG::speed_of_light, ScalarProd(cells_cgs[i].velocity, cells_cgs[i].velocity)) * time_scale_ / CG::speed_of_light;
    }
    for(size_t i = 0; i < Nlocal; ++i)
    {
        A[i].resize(max_neigh, 0);
        A_indeces[i].resize(max_neigh, max_size_t);
	if(A[i][0] < 0)
	  std::cout<<"Negative A in matrix build, density "<<cells_cgs[i].density<<" T "<<cells_cgs[i].temperature<<" fleck "<<fleck_factor[i]<<
	    " sig_P "<<sigma_planck[i]<<" dt "<<dt * time_scale_<<" Erad "<<cells_cgs[i].Erad * cells_cgs[i].density<<std::endl;
    }
}

void Diffusion::PostCG(Tessellation3D const& tess, std::vector<Conserved3D>& extensives, double const dt, std::vector<ComputationalCell3D>& cells,
        std::vector<double>const& full_CG_result, std::vector<double> const& CG_result) const
{
    double const max_v = 0.1 * CG::speed_of_light * time_scale_ / length_scale_;
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

    double Einit = 0;
    for(size_t i = 0; i < N; ++i)
        Einit += extensives[i].Erad + extensives[i].energy;
#ifdef RICH_MPI
    MPI_Allreduce(MPI_IN_PLACE, &Einit, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
#endif

    int good_end = 1;
    old_T.resize(N, 0);
    for(size_t i = 0; i < N; ++i)
    {
        double const old_e_therm = extensives[i].internal_energy;
        double const volume = tess.GetVolume(i) * length_scale_ * length_scale_* length_scale_;
        extensives[i].Erad = CG_result[i] * volume * time_scale_ * time_scale_ / (length_scale_ * length_scale_ * mass_scale_);
        double const T = cells[i].temperature;
        old_T[i] = T;
        double dE = fleck_factor[i] * CG::speed_of_light * dt * sigma_planck[i] * (full_CG_result[i] - T * T * T * T * CG::radiation_constant
            -0.5 * (3 - R2[i]) * std::min(max_v * max_v, ScalarProd(cells[i].velocity, cells[i].velocity)) * full_CG_result[i] * length_scale_ * length_scale_ / (CG::speed_of_light * CG::speed_of_light * time_scale_ * time_scale_)) * volume * time_scale_;
	    double old_Tr = 0;
        double compton_term = 0;
        double e_absorb = fleck_factor[i] * CG::speed_of_light * dt * sigma_planck[i] * full_CG_result[i] * volume * time_scale_;
        double e_emitt = -fleck_factor[i] * CG::speed_of_light * dt * sigma_planck[i] * T * T * T * T * CG::radiation_constant * volume * time_scale_;
        double e_v2 =  fleck_factor[i] * CG::speed_of_light * dt * sigma_planck[i] * (-0.5 * (3 - R2[i]) * std::min(max_v * max_v, ScalarProd(cells[i].velocity, cells[i].velocity)) * full_CG_result[i] * length_scale_ * length_scale_ / (CG::speed_of_light * CG::speed_of_light * time_scale_ * time_scale_)) * volume * time_scale_;
        if(compton_on_)
        {
            double const old_Er = cells[i].Erad * cells[i].density * mass_scale_ / (time_scale_ * time_scale_ * length_scale_);
            old_Tr = std::pow(old_Er / CG::radiation_constant, 0.25);
	        double const pre_factor = fleck_factor[i] * dt * time_scale_ * 4 * sigma_s[i] * CG::boltzmann_constant / (CG::electron_mass * CG::speed_of_light);
            compton_term = pre_factor * (old_Tr - T);
            double const theta = (fleck_factor[i] < 0.5 || std::abs(compton_term) > 1e-3) ? 1 : 0.1;
            compton_term = pre_factor * volume * (full_CG_result[i] * (old_Tr - T * (1 - theta)) - T * theta * old_Er);
            dE += pre_factor * volume * (full_CG_result[i] * (old_Tr - T * (1 - theta)) - T * theta * old_Er);
        }
        if(compton_on_)
            compton_term *= time_scale_ * time_scale_ / (length_scale_ * length_scale_ * mass_scale_);
        dE *= time_scale_ * time_scale_ / (length_scale_ * length_scale_ * mass_scale_);
        e_absorb *= time_scale_ * time_scale_ / (length_scale_ * length_scale_ * mass_scale_);
        e_emitt *= time_scale_ * time_scale_ / (length_scale_ * length_scale_ * mass_scale_);
        e_v2 *= time_scale_ * time_scale_ / (length_scale_ * length_scale_ * mass_scale_);
        extensives[i].energy += dE;
        extensives[i].internal_energy += dE;
        if(extensives[i].internal_energy < 0 || !std::isfinite(extensives[i].internal_energy) || extensives[i].Erad < 0)
        {
            good_end = 0;
            std::cout<<"Negative internal energy in postcg1, "<<extensives[i].internal_energy<<" ID "<<cells[i].ID<<
                " T "<<T<<std::endl;
                std::cout<<" CG_result "<<CG_result[i]<<" old Er "<<cells[i].Erad * cells[i].density * mass_scale_ / (time_scale_ * time_scale_ * length_scale_)
                <<" v "<<fastabs(cells[i].velocity)<<" mass "<<
                extensives[i].mass<<" dE "<<dE<<" R2 "<<R2[i]<<" old_e_therm "<<old_e_therm<<std::endl;
            std::cout<<cells[i]<<std::endl;
            std::cout<<extensives[i]<<std::endl;
             std::cout<<"max emitt "<<fleck_factor[i] * CG::speed_of_light * dt * sigma_planck[i] * T * T * T * T * CG::radiation_constant
             * volume * time_scale_ * time_scale_ * time_scale_ / (length_scale_ * length_scale_ * mass_scale_)<<" full_CG_result "<<full_CG_result[i]<<std::endl;
             std::cout<<"relativity "<<fleck_factor[i] * CG::speed_of_light * dt * sigma_planck[i]
             *0.5 * (3 - R2[i]) * ScalarProd(cells[i].velocity, cells[i].velocity) * full_CG_result[i] * length_scale_ * length_scale_ / (CG::speed_of_light * CG::speed_of_light * time_scale_ * time_scale_) * volume * time_scale_* time_scale_ * time_scale_ / (length_scale_ * length_scale_ * mass_scale_)<<std::endl;
             std::cout<<" fleck "<<fleck_factor[i]<<" sigma_planck "<<sigma_planck[i]<<
                " other dE "<<fleck_factor[i] * CG::speed_of_light * dt * sigma_planck[i] * (full_CG_result[i] - T * T * T * T * CG::radiation_constant
            -0.5 * (3 - R2[i]) * ScalarProd(cells[i].velocity, cells[i].velocity) * full_CG_result[i] * length_scale_ * length_scale_ / (CG::speed_of_light * CG::speed_of_light * time_scale_ * time_scale_)) * volume * time_scale_
             * time_scale_ * time_scale_ / (length_scale_ * length_scale_ * mass_scale_)<<" density "<<cells[i].density<<
             " compton_term "<<compton_term<<" old_Tr "<<old_Tr<<std::endl;
            break;
            // UniversalError eo("Bad internal energy in Diffusion::PostCG");
            // eo.addEntry("cell index", i);
            // eo.addEntry("energy", extensives[i].internal_energy);
	        // eo.addEntry("mass",extensives[i].mass);
            // eo.addEntry("CG_result", CG_result[i]);
            // eo.addEntry("T", T);
            // eo.addEntry("Density", cells[i].density);
            // eo.addEntry("ID", cells[i].ID);
            // eo.addEntry("Fleck",fleck_factor[i]);
            // eo.addEntry("old_Tr",old_Tr);
            // eo.addEntry("dE",dE);
            // throw eo;
        }

        tess.GetNeighbors(i, neighbors);
        size_t const Nneigh = neighbors.size();
        faces = tess.GetCellFaces(i);
        Vector3D const point = tess.GetMeshPoint(i);
        Vector3D gradE(0, 0, 0);
        double const Dcell = D[i];
        Vector3D r_ij;
        Vector3D const CM = tess.GetCellCM(i);
        double const Erad_factor = mass_scale_ / (time_scale_ * time_scale_ * length_scale_);
        double const cell_old_Er = Erad_factor * cells[i].Erad * cells[i].density;

        double total_relativity = 0;
        double etherm_mid = extensives[i].internal_energy;
        double const v_ratio = std::min(1.0, 0.05 * CG::speed_of_light / (fastabs(cells[i].velocity) * length_scale_ / time_scale_ + 1e-2));
        for(size_t j = 0; j < Nneigh; ++j)
        {
            size_t const neighbor_j = neighbors[j];
            r_ij = point - tess.GetMeshPoint(neighbor_j);
            double const r_ij_size = abs(r_ij);
            r_ij *= 1.0 / r_ij_size;
            double Er_j = 0;
            if(tess.IsPointOutsideBox(neighbor_j))
                boundary_calc_.GetOutSideValues(tess, cells, i, neighbor_j, full_CG_result, Er_j, dummy_v);
            else
                Er_j = full_CG_result[neighbor_j];
  

            gradE += (0.5 * tess.GetArea(faces[j]) * (Er_j + full_CG_result[i])) * r_ij * length_scale_ * length_scale_;
            double const momentum_term = (0.5 * dt * cell_flux_limiter[i] * tess.GetArea(faces[j]) * ScalarProd(cells[i].velocity, r_ij) * (Er_j + full_CG_result[i]) / 3) * (time_scale_ * time_scale_ * length_scale_ / mass_scale_);
            double const relativity_term = -v_ratio * momentum_term * 2 * 3 * sigma_planck[i] * Dcell / CG::speed_of_light;
            extensives[i].energy += /*momentum_term + */fleck_factor[i] * relativity_term;
            extensives[i].internal_energy += fleck_factor[i] * relativity_term;
            total_relativity += fleck_factor[i] * relativity_term;
        }
        Vector3D dP;
        double Erad_dE = 0;
        if(hydro_on_)
        {
            double const old_Ek = 0.5 * ScalarProd(extensives[i].momentum, extensives[i].momentum) / extensives[i].mass;
            dP = (cell_flux_limiter[i] * dt * time_scale_ / 3) * gradE * (time_scale_ / (length_scale_ * mass_scale_));
            Erad_dE = ScalarProd(dP, extensives[i].momentum) / extensives[i].mass;
            extensives[i].momentum += dP;
            double const new_Ek = 0.5 * ScalarProd(extensives[i].momentum, extensives[i].momentum) / extensives[i].mass;
            double const dE = -new_Ek + old_Ek + Erad_dE;
            extensives[i].Erad += dE;
            if(extensives[i].Erad < 0 && dE < 0 && std::abs(dE) < extensives[i].energy * 0.01)
                 extensives[i].Erad -= dE;
            extensives[i].energy = extensives[i].internal_energy +  ScalarProd(extensives[i].momentum, extensives[i].momentum) / (2 * extensives[i].mass);
        }
        if(extensives[i].Erad < 0 || extensives[i].internal_energy < 0 || !std::isfinite(extensives[i].internal_energy) || cells[i].Erad < 0)
        {
            std::cout<<"Negative internal energy is postcg2, "<<extensives[i].internal_energy<<" ID "<<cells[i].ID<<
                " T "<<T<<" CG_result "<<CG_result[i]<<" full_CG_result "<<full_CG_result[i]<<" v "<<fastabs(cells[i].velocity)<<" sigma_planck "<<
                sigma_planck[i]<<" sigma_r "<<CG::speed_of_light / (3 * Dcell)<<" E_init "<<cells[i].Erad*cells[i].density* mass_scale_ / (time_scale_ * time_scale_ * length_scale_)
                <<" volume "<<tess.GetVolume(i)<<" Erad "<<extensives[i].Erad<<" mass "<<extensives[i].mass<<" dP "<<dP.x<<","<<dP.y<<","<<dP.z<<" momentum "<<extensives[i].momentum.x<<","<<extensives[i].momentum.y<<","<<extensives[i].momentum.z<<std::endl;
            std::cout<<"Erad_dE "<<Erad_dE<<" cell_flux_limiter "<<cell_flux_limiter[i]<<" e_absorb "<<e_absorb<<" e_emitt "<<e_emitt<<" e_v2 "<<e_v2<<std::endl;
            std::cout<<"total relativity "<<total_relativity<<" etherm_mid "<<etherm_mid<<" fleck_factor "<<fleck_factor[i]<<" gradE "<<gradE * (1.0 / tess.GetVolume(i))<<std::endl;
            for(size_t j = 0; j < Nneigh; ++j)
            {
                size_t const neighbor_j = neighbors[j];
                r_ij = point - tess.GetMeshPoint(neighbor_j);
                double const r_ij_size = abs(r_ij);
                r_ij *= 1.0 / r_ij_size;
                double Er_j = 0;
                if(tess.IsPointOutsideBox(neighbor_j))
                    boundary_calc_.GetOutSideValues(tess, cells, i, neighbor_j, CG_result, Er_j, dummy_v);
                else
                    Er_j = full_CG_result[neighbor_j];

                Vector3D const cm_ij = CM - tess.GetCellCM(neighbor_j);
                Vector3D const grad_E = r_ij * ScalarProd(r_ij, cm_ij) * (1.0 / (length_scale_ * ScalarProd(cm_ij, cm_ij)));   
                double mid_D = 0.5 * (D[neighbor_j] + Dcell);
                double const flux_limiter_face = flux_limiter_ ? CalcSingleFluxLimiter(grad_E * (CG_result[i] - Er_j), mid_D, 0.5 * (CG_result[i] + Er_j)) : 1;
                double const max_local_v = std::min(max_v, std::max(-max_v, ScalarProd(cells[i].velocity, r_ij)));
                double const v_ratio = max_local_v / ScalarProd(cells[i].velocity, r_ij);
                double const momentum_term = (0.5 * dt * cell_flux_limiter[i] * tess.GetArea(faces[j]) * ScalarProd(cells[i].velocity, r_ij) * (Er_j + full_CG_result[i]) / 3) * (time_scale_ * time_scale_ * length_scale_ / mass_scale_);
                double const relativity_term = -v_ratio * fleck_factor[i] * momentum_term * 2 * 3 * sigma_planck[i] * Dcell / CG::speed_of_light;
                std::cout<<"relativity_term "<<relativity_term<<" flux_limiter_face "<<flux_limiter_face<<" Er_j "<<Er_j<<" Er_j_init "<<cells[neighbor_j].density * 
                    cells[neighbor_j].Erad* mass_scale_ / (time_scale_ * time_scale_ * length_scale_)<<" ID "<<cells[neighbor_j].ID<<" v_ratio "<<v_ratio<<" Area "<<tess.GetArea(faces[j])<<std::endl;
            }

            good_end = 0;
            std::cout<<"Negative internal energy is postcg2, "<<extensives[i].internal_energy<<" ID "<<cells[i].ID<<
                " T "<<T<<" CG_result "<<CG_result[i]<<" v "<<fastabs(cells[i].velocity)<<" mass "<<
                extensives[i].mass<<std::endl;
            break;
            UniversalError eo("Bad internal energy in Diffusion::PostCG second part");
            eo.addEntry("cell index", i);
            eo.addEntry("energy", extensives[i].internal_energy);
            eo.addEntry("CG_result", CG_result[i]);
            eo.addEntry("T", T);
            eo.addEntry("Density", cells[i].density);
            eo.addEntry("ID", cells[i].ID);
            eo.addEntry("cell_flux_limiter", cell_flux_limiter[i]);
            eo.addEntry("sigma_planck", sigma_planck[i]);
            eo.addEntry("sigma_rossland", CG::speed_of_light / (3 * Dcell));
            eo.addEntry("Vx", cells[i].velocity.x);
            eo.addEntry("Vy", cells[i].velocity.y);
            eo.addEntry("Vz", cells[i].velocity.z);
            eo.addEntry("newVx", extensives[i].momentum.x / extensives[i].mass);
            eo.addEntry("newVy", extensives[i].momentum.y / extensives[i].mass);
            eo.addEntry("newVz", extensives[i].momentum.z / extensives[i].mass);
            throw eo;
        }

        double const old_Edot = cells[i].Erad_dt;
        cells[i].Erad_dt = (extensives[i].Erad / extensives[i].mass - cells[i].Erad) / dt;
        cells[i].Erad_dt_dt = (cells[i].Erad_dt - old_Edot) / dt;
        if(!std::isfinite(cells[i].Erad_dt) || !std::isfinite(cells[i].Erad_dt))
            std::cout<<"Bad Edot "<<cells[i].Erad_dt<<" edot_dt_dt "<<cells[i].Erad_dt_dt<<" i "<<i<<" ID "<<cells[i].ID<<" dt "<<dt<<" Erad "<<extensives[i].Erad<<" m "<<extensives[i].mass<<" old_Edot "<<old_Edot<<std::endl;
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
        std::cout<<"Zero good_end rank "<<rank<<std::endl;
        was_bad = true;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &good_end, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    if(was_bad)
        std::cout<<"rank "<<rank<<" good_end "<<good_end<<std::endl;
#endif
    if(good_end == 0)
    {
        // std::cout<<"throwing error"<<std::endl;
        throw UniversalError("Negative energy in POSTCG");
    }

    double Efinal = 0;
    for(size_t i = 0; i < N; ++i)
        Efinal += extensives[i].Erad + extensives[i].energy;
#ifdef RICH_MPI
    MPI_Allreduce(MPI_IN_PLACE, &Efinal, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
    if(rank == 0)
        std::cout<<std::setprecision(14)<<"Einit "<<Einit<<" Efinal "<<Efinal<<std::endl;
}

void DiffusionSideBoundary::SetBoundaryValues(Tessellation3D const& tess, size_t const index, size_t const outside_point, double const dt, 
        std::vector<ComputationalCell3D> const& /*cells*/, double const Area, double& A, double &b, size_t const /*face_index*/)const
{
    double const R = tess.GetWidth(index);
    if(tess.GetMeshPoint(index).x > (tess.GetMeshPoint(outside_point).x + R * 1e-4))
    {
        A += 0.5 * CG::speed_of_light * dt * Area;
        b += 2 * Area * dt * CG::stefan_boltzman * T_ * T_ * T_ * T_;
    }
}

void DiffusionSideBoundary::SetMomentumTermBoundary(Tessellation3D const& tess, size_t const index, size_t const outside_point, double const dt,
        ComputationalCell3D const& cell, double const Area, double& A, double &b, size_t const face_index, double const fleck_factor,
        double const flux_limiter, double const D, double const sigma_planck)const
{
    double const R = tess.GetWidth(index);
    Vector3D r_ij = tess.GetMeshPoint(index) - tess.GetMeshPoint(outside_point);
    double const r_ij_size = abs(r_ij);
    r_ij *= 1.0 / r_ij_size;
    double const momentum_relativity_term = -0.5 * fleck_factor * dt * flux_limiter * Area * 
        (2 * 3 * sigma_planck * D / CG::speed_of_light - 1) * ScalarProd(cell.velocity, r_ij) / 3;
    if(tess.GetMeshPoint(index).x > (tess.GetMeshPoint(outside_point).x + R * 1e-4))
    {
        A += momentum_relativity_term;
        b -= momentum_relativity_term * CG::radiation_constant * T_ * T_ * T_ * T_;
    }
    else
        A += 2 * momentum_relativity_term;
}

void DiffusionSideBoundary::GetOutSideValues(Tessellation3D const& tess, std::vector<ComputationalCell3D> const& cells, size_t const index, size_t const outside_point,
    std::vector<double> const& new_E, double& E_outside, Vector3D& v_outside)const
{
    double const R = tess.GetWidth(index);
    if(tess.GetMeshPoint(index).x > (tess.GetMeshPoint(outside_point).x + R * 1e-4))
        E_outside = CG::radiation_constant * T_ * T_ * T_ * T_;
    else
        E_outside = new_E[index];
    v_outside = cells[index].velocity;
}

void DiffusionClosedBox::SetMomentumTermBoundary(Tessellation3D const& tess, size_t const index, size_t const outside_point, double const dt,
        ComputationalCell3D const& cell, double const Area, double& A, 
        double& /*b*/, size_t const /*face_index*/, double const fleck_factor, double const flux_limiter, 
        double const D, double const sigma_planck)const
{
    Vector3D r_ij = tess.GetMeshPoint(index) - tess.GetMeshPoint(outside_point);
    double const r_ij_size = abs(r_ij);
    r_ij *= 1.0 / r_ij_size;
    double const momentum_relativity_term = -0.5 * fleck_factor * dt * flux_limiter * Area * 
        (2 * 3 * sigma_planck * D / CG::speed_of_light - 1) * ScalarProd(cell.velocity, r_ij) / 3;
    A += 2 * momentum_relativity_term;
}

void DiffusionClosedBox::SetBoundaryValues(Tessellation3D const& /*tess*/, size_t const /*index*/, size_t const /*outside_point*/, double const /*dt*/, 
        std::vector<ComputationalCell3D> const& /*cells*/, double const /*Area*/, double& /*A*/, double& /*b*/, size_t const /*face_index*/)const
{}

void DiffusionClosedBox::GetOutSideValues(Tessellation3D const& tess, std::vector<ComputationalCell3D> const& cells, size_t const index, size_t const outside_point,
    std::vector<double> const& new_E, double& E_outside, Vector3D& v_outside)const
{
    E_outside = new_E[index];
    Vector3D normal = normalize(tess.GetMeshPoint(outside_point) - tess.GetMeshPoint(index));
    v_outside = cells[index].velocity;
    v_outside -= 2 * normal * ScalarProd(normal, v_outside);
}

double PowerLawOpacity::CalcDiffusionCoefficient(ComputationalCell3D const& cell) const
{
    return D0_ * std::pow(cell.density, alpha_) * std::pow(cell.temperature, beta_);
}

double PowerLawOpacity::CalcPlanckOpacity(ComputationalCell3D const& cell) const
{
    return planck0_ * std::pow(cell.density, alpha_planck_) * std::pow(cell.temperature, beta_planck_);
}

void DiffusionXInflowBoundary::SetBoundaryValues(Tessellation3D const& tess, size_t const index, size_t const outside_point, double const dt, 
        std::vector<ComputationalCell3D> const& cells, double const Area, double& A, double &b, size_t const face_index)const
{
    double const R = tess.GetWidth(index);
    if(tess.GetMeshPoint(index).x > (tess.GetMeshPoint(outside_point).x + R * 1e-4))
    {
        double const Er_j = left_state_.Erad * left_state_.density;
        double const Er = cells[index].Erad * cells[index].density;
        double const dx = tess.GetCellCM(index).x - tess.GetBoxCoordinates().first.x;
        Vector3D const grad_E = Vector3D(1, 0, 0) * (1.0 / (2 * dx));
        Vector3D const r_ij = tess.GetMeshPoint(index) - tess.GetMeshPoint(outside_point);
        double mid_D = 0.5 * (D_calc_.CalcDiffusionCoefficient(cells[index]) + D_calc_.CalcDiffusionCoefficient(left_state_));
        double const flux_limiter = CalcSingleFluxLimiter(grad_E * (Er - Er_j), mid_D, 0.5 * (Er + Er_j));
        mid_D *= flux_limiter;
        double const flux = ScalarProd(grad_E, r_ij * (tess.GetArea(face_index) / abs(r_ij))) * dt * mid_D; 
        A += flux;
        b += flux * Er_j;
    }
    else
    {
        if(tess.GetMeshPoint(index).x < (tess.GetMeshPoint(outside_point).x - R * 1e-4))
        {
            double const Er_j = right_state_.Erad * right_state_.density;
            double const Er = cells[index].Erad * cells[index].density;
            double const dx = tess.GetBoxCoordinates().second.x - tess.GetCellCM(index).x;
            Vector3D const grad_E = Vector3D(-1, 0, 0) * (1.0 / (2 * dx));
            Vector3D const r_ij = tess.GetMeshPoint(index) - tess.GetMeshPoint(outside_point);
            double mid_D = 0.5 * (D_calc_.CalcDiffusionCoefficient(cells[index]) + D_calc_.CalcDiffusionCoefficient(right_state_));
            double const flux_limiter = CalcSingleFluxLimiter(grad_E * (Er - Er_j), mid_D, 0.5 * (Er + Er_j));
            mid_D *= flux_limiter;
            double const flux = ScalarProd(grad_E, r_ij * (tess.GetArea(face_index) / abs(r_ij))) * dt * mid_D; 
            A += flux;
            b += flux * Er_j;
        }
    }
}

void DiffusionXInflowBoundary::SetMomentumTermBoundary(Tessellation3D const& tess, size_t const index, size_t const outside_point, double const dt,
        ComputationalCell3D const& cell, double const Area, double& A, double &b, size_t const face_index, double const fleck_factor,
        double const flux_limiter, double const D, double const sigma_planck)const
{
    double const R = tess.GetWidth(index);
    Vector3D r_ij = tess.GetMeshPoint(index) - tess.GetMeshPoint(outside_point);
    double const r_ij_size = abs(r_ij);
    r_ij *= 1.0 / r_ij_size;
    double const momentum_relativity_term = -0.5 * fleck_factor * dt * flux_limiter * Area * 
        (2 * 3 * sigma_planck * D / CG::speed_of_light - 1) * ScalarProd(cell.velocity, r_ij) / 3;
    if(tess.GetMeshPoint(index).x > (tess.GetMeshPoint(outside_point).x + R * 1e-4))
    {
        A += momentum_relativity_term;
        b -= momentum_relativity_term * left_state_.Erad * left_state_.density;
    }
    else
    {
        if(tess.GetMeshPoint(index).x < (tess.GetMeshPoint(outside_point).x - R * 1e-4))
        {
            A += momentum_relativity_term;
            b -= momentum_relativity_term * right_state_.Erad * right_state_.density;
        }
        else
            A += 2 * momentum_relativity_term;
    }
}

void DiffusionXInflowBoundary::GetOutSideValues(Tessellation3D const& tess, std::vector<ComputationalCell3D> const& cells, size_t const index, size_t const outside_point,
    std::vector<double> const& new_E, double& E_outside, Vector3D& v_outside)const
{
    double const R = tess.GetWidth(index);
    if(tess.GetMeshPoint(index).x > (tess.GetMeshPoint(outside_point).x + R * 1e-4))
    {
        E_outside = left_state_.Erad * left_state_.density;
        v_outside = left_state_.velocity;
    }
    else
    {
        if(tess.GetMeshPoint(index).x < (tess.GetMeshPoint(outside_point).x - R * 1e-4))
        {
            E_outside = right_state_.Erad * right_state_.density;
            v_outside = right_state_.velocity;
        }
        else
        {
            E_outside = new_E[index];
            Vector3D normal = normalize(tess.GetMeshPoint(outside_point) - tess.GetMeshPoint(index));
            v_outside = cells[index].velocity;
            v_outside -= 2 * normal * ScalarProd(normal, v_outside);
        }
    }
}

void DiffusionOpenBoundary::SetBoundaryValues(Tessellation3D const& /*tess*/, size_t const /*index*/, size_t const /*outside_point*/, double const dt,
    std::vector<ComputationalCell3D> const& /*cells*/, double const Area, double& A, double& /*b*/, size_t const /*face_index*/)const
{
    A += Area * dt * 0.5 * CG::speed_of_light;
}

void DiffusionOpenBoundary::GetOutSideValues(Tessellation3D const& tess, std::vector<ComputationalCell3D> const& cells, size_t const index, size_t const outside_point,
    std::vector<double> const& new_E, double& E_outside, Vector3D& v_outside)const
{
    E_outside = new_E[index] * 1e-20;
    v_outside = cells[index].velocity;
}

void DiffusionOpenBoundary::SetMomentumTermBoundary(Tessellation3D const& tess, size_t const index, size_t const outside_point, double const dt,
    ComputationalCell3D const& cell, double const Area, double& A, double &b, size_t const face_index, 
    double const fleck_factor, double const flux_limiter, double const D, double const sigma_planck)const
{
    Vector3D r_ij = tess.GetMeshPoint(index) - tess.GetMeshPoint(outside_point);
    double const r_ij_size = abs(r_ij);
    r_ij *= 1.0 / r_ij_size;
    double const momentum_relativity_term = -0.5 * fleck_factor * dt * flux_limiter * Area * (2 * 3 * 
        sigma_planck * D / CG::speed_of_light - 1) * ScalarProd(cell.velocity, r_ij) / 3;
    A += momentum_relativity_term;
}
