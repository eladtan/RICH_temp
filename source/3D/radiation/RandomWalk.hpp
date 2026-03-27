#ifndef RANDOM_WALK_HPP
#define RANDOM_WALK_HPP

#include <cstddef>
#include <vector>

struct PGRWCellData
{
    std::size_t groupCutoff = 0;
    double sigmaA_bar = 0.0;
    double sigmaT_bar = 0.0;
    double D = 0.0;
    double gamma = 1.0;
};

class RandomWalk
{
public:
    RandomWalk();

    double sampleLeakTime(double xi) const;
    double sampleRadius(double tau, double xi) const;

    double computeSurvival(double tau) const;
    double computeRadialCDF(double x, double tau) const;

private:
    double survivalEigen(double tau) const;
    double radialCdfEigen(double x, double tau) const;

    std::vector<double> tauTable;
    std::vector<double> survivalTable;
    std::vector<double> radiusTable;

    static constexpr double tauMin = 1e-8;
    static constexpr double tauMax = 64.0;
    static constexpr std::size_t tableSize = 1024;
    static constexpr std::size_t radiusTableSize = 256;
};

#endif // RANDOM_WALK_HPP
