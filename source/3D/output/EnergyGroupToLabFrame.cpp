#include "EnergyGroupToLabFrame.hpp"

EnergyGroupToLabFrame::EnergyGroupToLabFrame(MultigroupDiffusion const& multigroup, std::size_t const group) : multigroup_(multigroup), group(group) {}

std::string EnergyGroupToLabFrame::getName(void) const {
    return "EnergyGroupToLabFrame[" + std::to_string(group) + "]";
}

std::vector<double> EnergyGroupToLabFrame::operator()(HDSim3D const& sim) const {
    int rank = 0;
#ifdef RICH_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif
    if(rank == 0){
        std::cout<<"Starting EnergyGroupToLabFrame for group " << group <<std::endl;
    }

    auto const mass_scale   = multigroup_.mass_scale_;
    auto const length_scale = multigroup_.length_scale_;
    auto const time_scale   = multigroup_.time_scale_;

    auto const energy_per_volume_scale = mass_scale / (length_scale * pow<2>(time_scale));

    Tessellation3D const& tess = sim.getTesselation();
    std::vector<ComputationalCell3D> const& cells = sim.getCells();

    std::size_t const N = tess.GetPointNo();
    std::vector<double> Eg_lab(N, 0.0);
    std::vector<double> Eg_fluid(cells.size(), 0.0);

    std::vector<std::size_t> neighbors;
    face_vec faces;
    Vector3D dummy_v;

    for(std::size_t i = 0; i < N; ++i) {
        Eg_fluid[i] = cells[i].Eg[group] * cells[i].density * energy_per_volume_scale;

        // first term
        Eg_lab[i] = Eg_fluid[i];
    }

    for(size_t i = N + 3; i < cells.size(); ++i)
        Eg_fluid[i] = cells[i].Eg[group] * cells[i].density * energy_per_volume_scale;

    std::vector<double> lambda_g(N, 0.0);
    // second term
    Vector3D grad_Eg(0.0, 0.0, 0.0);
    ComputationalCell3D cell_temp; 
    for(std::size_t i=0; i < N; ++i){
        double const volume = tess.GetVolume(i) * pow<3>(length_scale);

        tess.GetNeighbors(i, neighbors);
        std::size_t const Nneighbors = neighbors.size();

        faces = tess.GetCellFaces(i);

        Vector3D const r_i = tess.GetMeshPoint(i);
        double const Eg_fluid_i = Eg_fluid[i];

        grad_Eg.Set(0.0, 0.0, 0.0);

        for(std::size_t j=0; j < Nneighbors; ++j){
            std::size_t const neighbor_j = neighbors[j];

            double Eg_fluid_j;

            if(!tess.IsPointOutsideBox(neighbor_j)){
                Eg_fluid_j = Eg_fluid[neighbor_j];
            } else {
                multigroup_.boundary_calculator.getOutsideValuesGroup(group, tess, i, neighbor_j, cells, Eg_fluid_i, Eg_fluid_j, dummy_v);
            }

            auto const r_ij = normalize(r_i - tess.GetMeshPoint(neighbor_j));
            double const A_ij = tess.GetArea(faces[j]) * pow<2>(length_scale);

            grad_Eg += r_ij * (0.5 * A_ij * (Eg_fluid_i + Eg_fluid_j));
        }
        grad_Eg *= (1.0 / volume);

        cell_temp = cells[i];
        cell_temp.density *= mass_scale / pow<3>(length_scale);

        double const Dg_i = multigroup_.coefficient_calculator.CalcDiffusionCoefficientGroup(cell_temp, group);

        lambda_g[i] = multigroup_.flux_limiter_ ? CG::CalcSingleFluxLimiter(grad_Eg, Dg_i, Eg_fluid_i) : 1.0;
        
        cell_temp.velocity *= length_scale / time_scale;

        Eg_lab[i] += 2.0 / (CG::speed_of_light * CG::speed_of_light) * lambda_g[i] * Dg_i * ScalarProd(grad_Eg, cell_temp.velocity);
    }


    // third term
    Vector3D grad_Egp(0.0, 0.0, 0.0);
    std::size_t const gp = (group + 1) < ENERGY_GROUPS_NUM ? (group+1) : group;
    double const dnu_p = multigroup_.energy_groups_width[gp];
    
    Vector3D grad_Egm(0.0, 0.0, 0.0);   
    std::size_t const gm = group > 0 ? group-1 : group;
    double const dnu_m = multigroup_.energy_groups_width[gm];

    double const dnu = multigroup_.energy_groups_width[group];

    double const nu_p = multigroup_.energy_groups_boundary[group+1];
    double const nu_m = multigroup_.energy_groups_boundary[group];

    for(std::size_t i=0; i < N; ++i){
        double const volume = tess.GetVolume(i) * pow<3>(length_scale);

        tess.GetNeighbors(i, neighbors);
        std::size_t const Nneighbors = neighbors.size();

        faces = tess.GetCellFaces(i);

        Vector3D const r_i = tess.GetMeshPoint(i);
        Vector3D const velocity_i = cells[i].velocity * length_scale / time_scale;

        double const Eg_fluid_i = Eg_fluid[i];
        
        double const Egp_fluid_i = cells[i].Eg[gp] * cells[i].density * energy_per_volume_scale;
        double const Egm_fluid_i = cells[i].Eg[gm] * cells[i].density * energy_per_volume_scale;

        grad_Egp.Set(0.0, 0.0, 0.0);
        grad_Egm.Set(0.0, 0.0, 0.0);
        grad_Eg.Set(0.0, 0.0, 0.0);
        for(std::size_t j=0; j < Nneighbors; ++j){
            std::size_t const neighbor_j = neighbors[j];

            double Eg_fluid_j, Egp_fluid_j, Egm_fluid_j;
            if(!tess.IsPointOutsideBox(neighbor_j)){
                Eg_fluid_j = Eg_fluid[neighbor_j];
                
                Egp_fluid_j = cells[neighbor_j].Eg[gp] * cells[neighbor_j].density * energy_per_volume_scale;
                Egm_fluid_j = cells[neighbor_j].Eg[gm] * cells[neighbor_j].density * energy_per_volume_scale;
            } else {
                multigroup_.boundary_calculator.getOutsideValuesGroup(gp, tess, i, neighbor_j, cells, Eg_fluid_i, Eg_fluid_j, dummy_v);
                multigroup_.boundary_calculator.getOutsideValuesGroup(gp, tess, i, neighbor_j, cells, Egp_fluid_i, Egp_fluid_j, dummy_v);
                multigroup_.boundary_calculator.getOutsideValuesGroup(gm, tess, i, neighbor_j, cells, Egm_fluid_i, Egm_fluid_j, dummy_v);
            }

            auto const r_ij = normalize(r_i - tess.GetMeshPoint(neighbor_j));
            double const A_ij = tess.GetArea(faces[j]) * pow<2>(length_scale);

            grad_Egp += r_ij * (0.5 * A_ij * (Egp_fluid_i + Egp_fluid_j));

            grad_Egm += r_ij * (0.5 * A_ij * (Egm_fluid_i + Egm_fluid_j));

            grad_Eg += r_ij * (0.5 * A_ij * (Eg_fluid_i + Eg_fluid_j));
        }

        grad_Egp *= -1.0 / volume;
        grad_Egm *= -1.0 / volume;
        grad_Eg  *= -1.0 / volume;
        
        cell_temp = cells[i];
        cell_temp.density *= mass_scale / pow<3>(length_scale);

        double const Dg_i = multigroup_.coefficient_calculator.CalcDiffusionCoefficientGroup(cell_temp, group);

        double const Dgp_i = multigroup_.coefficient_calculator.CalcDiffusionCoefficientGroup(cell_temp, gp);

        double const Dgm_i = multigroup_.coefficient_calculator.CalcDiffusionCoefficientGroup(cell_temp, gm);

        double const lambda_gp = multigroup_.flux_limiter_ ? CG::CalcSingleFluxLimiter(grad_Egp, Dgp_i, Egp_fluid_i) : 1.0;

        double const lambda_gm = multigroup_.flux_limiter_ ? CG::CalcSingleFluxLimiter(grad_Egm, Dgm_i, Egm_fluid_i) : 1.0;
        
        double const lambda_nu_p = (dnu_p * lambda_gp + dnu * lambda_g[i]) / (dnu_p + dnu);

        double const lambda_nu_m = (dnu * lambda_g[i] + dnu_m * lambda_gm) / (dnu + dnu_m);

        double const Dg_nu_p = (dnu_p * Dgp_i + dnu * Dg_i) / (dnu_p + dnu);

        double const Dg_nu_m = (dnu * Dg_i + dnu_m * Dgm_i) / (dnu + dnu_m);


        double const coeff_nu_p = nu_p * Dg_nu_p * lambda_nu_p;
        double const coeff_nu_m = nu_m * Dg_nu_m * lambda_nu_m;

        Vector3D const grad_E_nu_diff = coeff_nu_p * (dnu_p*grad_Egp + dnu*grad_Eg) / (dnu_p + dnu) - coeff_nu_m*(dnu*grad_Eg + dnu_m*grad_Egm) / (dnu + dnu_m); 


        Eg_lab[i] -= 1.0 / (CG::speed_of_light * CG::speed_of_light) * ScalarProd(velocity_i, grad_E_nu_diff);

        Eg_lab[i] /= energy_per_volume_scale;
        Eg_lab[i] /= cells[i].density;
        Eg_lab[i] = std::max(Eg_lab[i], 0.0);
    }

    return Eg_lab;
}