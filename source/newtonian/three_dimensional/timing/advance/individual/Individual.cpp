#if 0

#include "Individual.hpp"

#include "mpi/mpi_exchange_commands.hpp"
#include "3D/tessellation/mpi_exchange.hpp"

IndividualTimeStep::IndividualTimeStep(HydroTimeAdvance &hydroAdvance, const PointMotion3D &pm, dt_t currentTime):
    TimeStep(hydroAdvance, pm, currentTime)
{
    const std::pair<Vector3D, Vector3D> &boundaries = this->tess.GetBoxCoordinates();
    this->ll = boundaries.first;
    this->ur = boundaries.second;
}

double IndividualTimeStep::determineBigTimestep(const vector<Vector3D> &face_velocities) const
{
    #ifdef RICH_MPI
        DistributedTimestepCalculator tc(this->ll, this->ur, this->tess, this->cells, face_velocities);
    #else // RICH_MPI
        SerialTimestepCalculator tc(this->ll, this->ur, this->tess, this->cells, face_velocities);
    #endif // RICH_MPI
    std::vector<dt_t> times = tc.calculateTimes();
    double big_time = MIN_TIME;
    for(const dt_t &time : times)
    {
        big_time = std::max(big_time, time);
    }
    
    #ifdef RICH_MPI
        MPI_Allreduce(MPI_IN_PLACE, &big_time, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD); // calculate maximum
    #endif // RICH_MPI

    return big_time;
}

dt_t IndividualTimeStep::apply()
{
    std::vector<Vector3D> points = this->tess.getMeshPoints();
    points.resize(this->tess.GetPointNo());

    if(points.empty())
    {
        throw UniversalError("Before using individual advance, a tessellation from all the points must be built");
    }

    auto [initial_point_velocities, initial_face_velocities] = this->CalculateInitialPointFaceVelocities();
    double bigTimestep = this->determineBigTimestep(initial_face_velocities);
    this->FixPointFaceVelocities(initial_point_velocities, initial_face_velocities, bigTimestep);
    
    // first, determine the timesteps required by all cells
    bigTimestep = this->determineBigTimestep(initial_face_velocities); // determine timestep again, after the fix of velocities
    this->timestepManager = TimestepManager(bigTimestep);
    for(const ComputationalCell3D &cell : this->cells)
    {
        this->timestepManager.addCell(cell);
    }

    double wholeTimeStep = (1 - EPSILON) * this->timestepManager.getWholeTimestep();

    while(this->timestepManager.getElapsedTime() < wholeTimeStep)
    {
        dt_t dt = this->timestepManager.getSmallTimestep();
        
        // step 6: update points array
        points = this->tess.getAllPoints();
        size_t originalNumPoint = points.size();

        // step 1: determine what are the participating cells indices
        std::vector<size_t> participatingIndices = this->timestepManager.getCellsOfCurrentTime(); // TODO: returns ID, not index
        
        // step 7: advance in time
        this->currentTime += dt;
        this->timestepManager.advance();

        // step 3: hydrodynamical step part 1 // todo: there's a build inside
        this->hydroAdvance.beforeAdvance(participatingIndices, this->currentTime, dt);

        #ifdef RICH_MPI
        // todo: this part is wrong, should iterate over points, and points that do not appear in `GetSelfIndex()` should be removed, also should be before the exchange communication above (probably)
        // consider points movement between processors - remove sent points
        size_t pointsStayedAtMeNum = tess.GetSelfIndex().size();
        this->timestepManager.removeCells(this->cells.begin() + pointsStayedAtMeNum, this->cells.begin() + originalNumPoint); // todo: correct?
        #endif // RICH_MPI

        // step 4: move and update participating cells (or their neighbors)
        // TODO: needed? there's a thing in hydro advance
        
        // step 5: determine the new timestep for each cell
        std::vector<std::pair<size_t, dt_t>> newCellTimes;
        for(const size_t &cellIdx : participatingIndices) // todo: a neighbor can also change dt?
        {
            newCellTimes.push_back({cellIdx, this->cells[cellIdx].dt});
        }
        this->timestepManager.updateCellTimes(newCellTimes);

        #ifdef RICH_MPI
        // consider points movement between processors - now add received points
        this->timestepManager.addCells(this->cells.begin() + pointsStayedAtMeNum, this->cells.begin() + pointsStayedAtMeNum + tess.GetAllPointsNo());
        #endif // RICH_MPI

        // step 8: hydrodynamical step part 2
        this->hydroAdvance.afterAdvance(participatingIndices, this->currentTime, dt);
    }
    // this->hydroAdvance.afterBigAdvance(this->currentTime, bigTimestep, point_velocities, face_velocities);
    return bigTimestep;
}

#endif