#include "TimeAdvance2.hpp"

TimeAdvance2::TimeAdvance2(Tessellation3D& tess, std::vector<ComputationalCell3D> &cells, vector<Conserved3D> &extensive,
                            const EquationOfState& eos, const FluxCalculator3D& fc, const CellUpdater3D& cu,
                            const ExtensiveUpdater3D& eu, const SourceTerm3D& source):
                            HydroTimeAdvance(tess, cells, extensive, eos, fc, cu, eu, source)
{}

void TimeAdvance2::beforeAdvance(dt_t currentTime, dt_t dt, std::vector<Vector3D> &point_vel, std::vector<Vector3D> &face_vel)
{    
    this->fluxes.clear();
    this->mid_extensives.clear();
	std::vector<std::pair<ComputationalCell3D, ComputationalCell3D>> face_values = this->fc(this->fluxes, this->tess, face_vel, this->cells, this->extensive, this->eos, currentTime, dt);
	this->mid_extensives = this->extensive;
	this->eu(this->fluxes, this->tess, dt, this->cells, this->mid_extensives, currentTime, face_vel, face_values);
	this->source(this->tess, this->cells, this->fluxes, point_vel, currentTime, dt, this->mid_extensives);

	// if(cycle % 10 == 0)
	// {
	// 	vector<Vector3D>& mesh = this->tess.accessMeshPoints();
	// 	mesh.resize(this->tess.GetPointNo());
	// 	vector<size_t> order = HilbertOrder3D(mesh);
	// 	mesh = VectorValues(mesh, order);
	// 	mid_extensives = VectorValues(mid_extensives, order);
	// 	this->extensive = VectorValues(this->extensive, order);
	// 	this->cells = VectorValues(this->cells, order);
	// 	point_vel = VectorValues(point_vel, order);
	// }
	MovePoints(this->tess, point_vel, dt);
	auto t1 = get_time();
	UpdateTessellation(this->tess, point_vel, dt);
	auto t2 = get_time();
	DisplayTime(t1, t2, "Voronoi build time");
    #ifdef RICH_MPI
        // Keep relevant points
        Conserved3D edummy;
        ComputationalCell3D cdummy;
        Vector3D vdummy;
        MPI_exchange_data(this->tess, this->mid_extensives, false, &edummy);
        MPI_exchange_data(this->tess, this->extensive, false, &edummy);
        MPI_exchange_data(this->tess, this->cells, false, &cdummy);
        MPI_exchange_data(this->tess, point_vel, false, &vdummy);
        MPI_exchange_data(this->tess, point_vel, true, &vdummy);
    #endif

    this->cu(this->cells, this->eos, this->tess, this->mid_extensives);
    #ifdef RICH_MPI
    MPI_exchange_data(this->tess, this->cells, true, &cdummy);
    #endif
}

void TimeAdvance2::afterAdvance(dt_t currentTime, dt_t dt, std::vector<Vector3D> &point_vel, std::vector<Vector3D> &face_vel)
{
    CalcFaceVelocities(this->tess, point_vel, face_vel);
    std::vector<std::pair<ComputationalCell3D, ComputationalCell3D>> face_values = this->fc(this->fluxes, this->tess, face_vel, this->cells, this->mid_extensives, this->eos, currentTime, dt);
    this->source(this->tess, this->cells, this->fluxes, point_vel, currentTime, dt, this->mid_extensives);
    this->eu(this->fluxes, this->tess, dt, this->cells, this->mid_extensives, currentTime, face_vel, face_values);
    ExtensiveAvg(this->extensive, this->mid_extensives);
    this->cu(this->cells, this->eos, this->tess, this->extensive);
    #ifdef RICH_MPI
        ComputationalCell3D cdummy;
        MPI_exchange_data(this->tess, this->cells, true, &cdummy);
    #endif
}