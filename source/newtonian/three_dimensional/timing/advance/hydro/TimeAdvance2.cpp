// #include "TimeAdvance2.hpp"

// #include "3D/tessellation/mpi_exchange.hpp"

// void TimeAdvance2::UpdateVelocities(const std::vector<ComputationalCell3D> &activeCells, const std::vector<size_t> &participatingIndices, dt_t currentTime, dt_t dt)
// {
//     // todo: check correctness of this method
//     if(activeCells.size() != this->tess.GetPointNo())
//     {
//         UniversalError eo("UpdateVelocities: activeCells size is not equal to tess.GetPointNo()");
//         eo.addEntry("activeCells.size()", activeCells.size());
//         eo.addEntry("tess.GetPointNo()", this->tess.GetPointNo());
//         throw eo;
//     }

// 	this->pm(this->tess, activeCells, currentTime, this->activePointVel);
    
//     // todo: this->point_vel value???????
//     tess.SyncPartialBuildData(this->activePointVel, this->point_vel);

//     // TODO: exchange needed? Maybe happens in `SyncPartialBuildData`?
//     #ifdef RICH_MPI
//         MPI_Tessellation_ghost_exchange_data(this->tess, this->point_vel);
//     #endif // RICH_MPI

//     this->pm.ApplyFix(this->tess, activeCells, currentTime, dt, this->activePointVel);

//     tess.SyncPartialBuildData(this->activePointVel, this->point_vel);

//     #ifdef RICH_MPI
//         MPI_Tessellation_ghost_exchange_data(this->tess, this->point_vel);
//     #endif // RICH_MPI

//     CalcFaceVelocities(this->tess, this->activePointVel, this->activeFaceVelocities); // todo: calculation is OK (because of pointVel)?

//     // TODO: think about what happens here
// }

// TimeAdvance2::TimeAdvance2(Tessellation3D& tess, std::vector<ComputationalCell3D> &cells, vector<Conserved3D> &extensive,
//                             const EquationOfState& eos, const PointMotion3D& pm, const FluxCalculator3D& fc, const CellUpdater3D& cu,
//                             const ExtensiveUpdater3D& eu, const SourceTerm3D& source):
//                             HydroTimeAdvance(tess, cells, extensive, eos, pm, fc, cu, eu, source)
// {}

// void TimeAdvance2::beforeAdvance(const std::vector<size_t> &participatingIndices, dt_t currentTime, dt_t dt)
// {    
//     this->fluxes.clear();
//     // this->mid_extensives.clear();
//     this->activeExtensive.clear();
//     this->activeCells.clear();
//     this->activeMidExtensive.clear();
//     this->point_vel.clear();
//     this->activeFaceVelocities.clear();


//     const Tessellation3D::AllPointsMap &indicesInBigArray = tess.GetIndicesInAllPoints();
//     for(const std::pair<size_t, size_t> &indices : indicesInBigArray)
//     {
//         size_t pointIdxInAll = indices.second;
//         this->activeCells.push_back(this->cells[pointIdxInAll]);
//         this->activeMidExtensive.push_back(this->mid_extensives[pointIdxInAll]);
//         this->activeExtensive.push_back(this->extensive[pointIdxInAll]);
//     }
//     this->cu(this->activeCells, this->eos, this->tess, this->activeMidExtensive); // calculates intensive variables

//     // at the end of this function, `this->point_vel` and `this->activeFaceVelocities` contain the correct velocities of *all* (even non active) points.
//     UpdateVelocities(this->activeCells, participatingIndices, currentTime, dt);

// 	auto face_values = this->fc(this->fluxes, this->tess, this->activeFaceVelocities, this->activeCells, this->activeMidExtensive, this->eos, currentTime, dt);
// 	this->source(this->tess, this->activeCells, this->fluxes, this->activePointVel, currentTime, dt, this->activeMidExtensive);
//     // TODO: currently, the CM of neighbors we used might be incorrect if neighbors were not built in this timestep. Maybe should be changed?
// 	this->eu(this->fluxes, this->tess, dt, this->activeCells, this->activeMidExtensive, currentTime, this->activeFaceVelocities, face_values); // calculates extensive variables

//     // size_t N = this->tess.GetPointNo();
//     // for(size_t i = 0; i < N; i++)
//     // {
//     //     size_t pointIdxAmongAll = indicesInBigArray.at(i);
//     //     this->activeMidExtensive[i] += this->mid_extensives[pointIdxAmongAll] - this->extensive[pointIdxAmongAll];
//     // }

// 	// if(cycle % 10 == 0)
// 	// {
// 	// 	vector<Vector3D>& mesh = this->tess.accessMeshPoints();
// 	// 	mesh.resize(this->tess.GetPointNo());
// 	// 	vector<size_t> order = HilbertOrder3D(mesh);
// 	// 	mesh = VectorValues(mesh, order);
// 	// 	mid_extensives = VectorValues(mid_extensives, order);
// 	// 	this->extensive = VectorValues(this->extensive, order);
// 	// 	this->cells = VectorValues(this->cells, order);
// 	// 	point_vel = VectorValues(point_vel, order);
// 	// }
// 	MovePoints(this->tess, this->point_vel, dt);
// 	auto t1 = get_time();
// 	UpdateTessellation(this->tess, participatingIndices, this->point_vel, dt);
// 	auto t2 = get_time();
// 	DisplayTime(t1, t2, "Voronoi build time");
    
//     #ifdef RICH_MPI
//         MPI_Tessellation_movement_exchange_data(this->tess, this->cells);
//         MPI_Tessellation_movement_exchange_data(this->tess, this->mid_extensives);
//         MPI_Tessellation_movement_exchange_data(this->tess, this->extensive);
//         MPI_Tessellation_movement_exchange_data(this->tess, this->point_vel);
//     #endif // RICH_MPI
    
//     this->tess.SyncPartialBuildData(this->activeCells, this->cells);
//     this->tess.SyncPartialBuildData(this->activeMidExtensive, this->mid_extensives);
//     this->tess.SyncPartialBuildData(this->activeExtensive, this->extensive);
//     this->tess.SyncPartialBuildData(this->activePointVel, this->point_vel);
//     // // todo: I added:
//     // // set the cells and extensives back:
//     // size_t index = 0;
//     // for(const std::pair<size_t, size_t> &indices : indicesInBigArray)
//     // {
//     //     size_t pointIdxInAll = indices.second;
//     //     this->cells[pointIdxInAll] = this->activeCells[index];
//     //     this->extensive[pointIdxInAll] = this->activeExtensive[index];
//     //     this->mid_extensives[pointIdxInAll] = this->activeMidExtensive[index];
//     //     index++;
//     // }
//     #ifdef RICH_MPI
//         MPI_Tessellation_ghost_exchange_data(this->tess, this->point_vel);
//     #endif // RICH_MPI
//     ExtensiveAvg(this->activeExtensive, this->activeMidExtensive);
//     this->tess.SyncPartialBuildData(this->activeExtensive, this->extensive);

//     // todo: active from here?
//     this->cu(this->activeCells, this->eos, this->tess, this->activeExtensive); // calculates intensive variables
//     this->tess.SyncPartialBuildData(this->activeExtensive, this->extensive);
//     #ifdef RICH_MPI
//         MPI_Tessellation_ghost_exchange_data(this->tess, this->activeCells);
//     #endif // RICH_MPI
// }

// void TimeAdvance2::afterAdvance(const std::vector<size_t> &participatingIndices, dt_t currentTime, dt_t dt)
// {
//     this->activeMidExtensive.clear();
//     const Tessellation3D::AllPointsMap &indicesInBigArray = tess.GetIndicesInAllPoints();
//     for(const std::pair<size_t, size_t> &indices : indicesInBigArray)
//     {
//         size_t pointIdxInAll = indices.second;
//         // this->activeCells.push_back(this->cells[pointIdxInAll]);
//         this->activeMidExtensive.push_back(this->extensive[pointIdxInAll]);
//     }
//     // todo: change this method to active
//     CalcFaceVelocities(this->tess, this->activePointVel, this->activeFaceVelocities); // todo: calculation is correct?
//     auto face_values = this->fc(this->fluxes, this->tess, this->activeFaceVelocities, this->activeCells, this->activeMidExtensive, this->eos, currentTime, dt);
//     this->source(this->tess, this->cells, this->fluxes, this->point_vel, currentTime, dt, this->activeMidExtensive);
//     this->eu(this->fluxes, this->tess, dt, this->cells, this->activeMidExtensive, currentTime, this->activeFaceVelocities, face_values);

//     this->cu(this->activeCells, this->eos, this->tess, this->activeMidExtensive);
//     this->tess.SyncPartialBuildData(this->activeMidExtensive, this->mid_extensives);
//     this->tess.SyncPartialBuildData(this->activeCells, this->cells);
//     #ifdef RICH_MPI
//         MPI_Tessellation_ghost_exchange_data(this->tess, this->activeCells);
//     #endif // RICH_MPI

//     // todo: update `cells` and `extensive` back
// }