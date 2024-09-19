#include "Individual.hpp"

IndividualTimeStep::TimestepManager::TimestepManager(double bigTimestep): TimestepManager()
{
    this->initializeTimestep(bigTimestep);
}

void IndividualTimeStep::TimestepManager::initializeTimestep(double bigTimestep)
{
    this->wholeTimeStep = bigTimestep;
    this->powerOfStep = 0;
    this->timesCells.clear();
    this->cellsToTimes.clear();
}

double IndividualTimeStep::TimestepManager::getCellNewTimestep(const ComputationalCell3D &cell) const
{
    return std::pow(2, this->cellsToTimes.at(cell.ID)) * this->getSmallTimestep();
}

void IndividualTimeStep::TimestepManager::internalUpdate()
{
    this->powerOfStep = 0;
    this->timesCells.clear();
    for(const std::pair<size_t, size_t> &cellTimePair : this->cellsToTimes)
    { 
        size_t cellID = cellTimePair.first;
        size_t cellPower = cellTimePair.second;
        if(cellPower > this->powerOfStep)
        {
            this->powerOfStep = cellPower;
        }
        if(not this->timesCells.contains(cellPower))
        {
            this->timesCells.insert({cellPower, std::vector<size_t>()});
        }
        this->timesCells[cellPower].push_back(cellID);
    }

    // update the nextTimesForPowers array
    size_t lastSize = this->nextTimesForPowers.size();
    this->nextTimesForPowers.resize(this->powerOfStep + 1);
    for(size_t pow = lastSize; pow <= this->powerOfStep; pow++)
    {
        // round to the closest multiple of a `pow` step, relatively to the time elapsed
        double mySteps = this->wholeTimeStep / std::pow(2, pow);
        this->nextTimesForPowers[pow] = mySteps * std::ceil(this->currentElapsedTime / mySteps);
    }
}

void IndividualTimeStep::TimestepManager::updateCellTimes(const std::vector<std::pair<size_t, dt_t>> &cellTimePair)
{
    for(const std::pair<size_t, dt_t> &cellTime : cellTimePair)
    {
        int powOfCell = std::ceil(std::log2(this->wholeTimeStep / cellTime.second));
        size_t cellID = cellTime.first;
        if(this->cellsToTimes.contains(cellID))
        {
            this->cellsToTimes[cellID] = powOfCell;
        }
        else
        {
            this->cellsToTimes.insert({cellID, powOfCell});
        }
    }
    this->internalUpdate();
}

void IndividualTimeStep::TimestepManager::addCell(const ComputationalCell3D &cell)
{
    this->updateCellTimes({{cell.ID, cell.dt}});
}

void IndividualTimeStep::TimestepManager::addCells(const std::vector<ComputationalCell3D>::const_iterator &begin, const std::vector<ComputationalCell3D>::const_iterator &end)
{
    std::vector<std::pair<size_t, dt_t>> cellTimePairs;
    for(auto it = begin; it != end; it++)
    {
        cellTimePairs.emplace_back(std::pair<size_t, dt_t>(it->ID, it->dt));
    }
    this->updateCellTimes(cellTimePairs);
}

void IndividualTimeStep::TimestepManager::removeCells(const std::vector<ComputationalCell3D>::const_iterator &begin, const std::vector<ComputationalCell3D>::const_iterator &end)
{
    for(auto it = begin; it != end; it++)
    {
        size_t ID = it->ID;
        this->cellsToTimes.erase(ID);
    }
    this->internalUpdate();
}

std::vector<size_t> IndividualTimeStep::TimestepManager::getCellsOfCurrentTime()
{
    std::vector<size_t> cellsOfCurrentTime;
    for(size_t pow = 0; pow <= this->powerOfStep; pow++)
    {
        if(this->currentElapsedTime >= (1 - EPSILON) * this->nextTimesForPowers[pow])
        {
            // std::cout << "for pow = " << pow << ", we have " << this->timesCells[pow].size() << " cells" << std::endl;
            if(this->timesCells.contains(pow))
            {
                cellsOfCurrentTime.insert(cellsOfCurrentTime.end(), this->timesCells[pow].cbegin(), this->timesCells[pow].cend());
            }
            double mySteps = this->wholeTimeStep / std::pow(2, pow);
            this->nextTimesForPowers[pow] += mySteps;
            // std::cout << "increasing pow = " << pow << " by " << mySteps << " steps, now its " << this->nextTimesForPowers[pow] << std::endl;
        } 
    }
    return cellsOfCurrentTime;
}