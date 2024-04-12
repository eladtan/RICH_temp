#include <iostream>
#include <algorithm>
#include <vector>
#include <boost/container/flat_map.hpp>

#include "../TimeStep.hpp"
#ifdef RICH_MPI
    #include "../../tree/DistributedTimestepCalculator.hpp"
#else // RICH_MPI
    #include "../../tree/SerialTimestepCalculator.hpp"
#endif // RICH_MPI

class IndividualTimeStep: public TimeStep
{
private:
    class TimestepManager
    {
        using CellsToTimesMap = boost::container::flat_map<size_t, size_t>;
        using TimesMap = boost::container::flat_map<size_t, std::vector<size_t>>;

    public:
        explicit TimestepManager() = default;

        explicit TimestepManager(double bigTimestep);

        void initializeTimestep(double bigTimestep);

        void addCell(const ComputationalCell3D &cell);

        void addCells(const std::vector<ComputationalCell3D>::const_iterator &begin, const std::vector<ComputationalCell3D>::const_iterator &end);
        
        void removeCells(const std::vector<ComputationalCell3D>::const_iterator &begin, const std::vector<ComputationalCell3D>::const_iterator &end);

        void updateCellTimes(const std::vector<std::pair<size_t, dt_t>> &cellTimePair);

        std::vector<size_t> getCellsOfCurrentTime();
        
        inline void advance()
        {
            // for(int i = 0; i < this->nextTimesForPowers.size(); i++)
            //     std::cout << this->nextTimesForPowers[i] << " ";
            // std::cout << std::endl;
            this->currentElapsedTime = *std::min_element(this->nextTimesForPowers.begin(), this->nextTimesForPowers.end());
        };

        inline double getElapsedTime() const{return this->currentElapsedTime;};
        
        inline double getSmallTimestep() const{return this->wholeTimeStep / std::pow(2, this->powerOfStep);};
        
        inline double getWholeTimestep() const{return this->wholeTimeStep;};
        
        double getCellNewTimestep(const ComputationalCell3D &cell) const;

    private:
        CellsToTimesMap cellsToTimes;
        TimesMap timesCells;
        double wholeTimeStep;
        double currentElapsedTime;
        size_t powerOfStep;
        std::vector<dt_t> nextTimesForPowers;

        void internalUpdate();
    };

    double determineBigTimestep(const vector<Vector3D> &face_velocities) const;

public:
    IndividualTimeStep(HydroTimeAdvance &hydroAdvance, const PointMotion3D &pm, dt_t currentTime);
    
    dt_t apply() override;

    Vector3D ll, ur;
    TimestepManager timestepManager;
    /*
    todo:
    double cfl_constant;
    double source_cfl_constant;
    std::vector<size_t> no_calc_indeces;
    ?
    */
};