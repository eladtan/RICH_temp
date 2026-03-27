#include "RandomWalk.hpp"
#include <cmath>
#include <algorithm>
#include <cassert>
#include <stdexcept>

static inline double clamp01(double v) { return std::clamp(v, 0.0, 1.0); }

static constexpr int eigenTerms = 500;
static constexpr double PI = 3.14159265358979323846;
static constexpr double tauFreeSpace = 0.01;

double RandomWalk::survivalEigen(double tau) const
{
    double sum = 0.0;
    for(int n = 1; n <= eigenTerms; ++n)
    {
        double arg = -static_cast<double>(n * n) * PI * PI * tau;
        if(arg < -700.0)
            break;
        double term = std::exp(arg);
        sum += (n % 2 == 1) ? term : -term;
    }
    return clamp01(2.0 * sum);
}

double RandomWalk::computeSurvival(double tau) const
{
    if(tau <= 0.0)
        return 1.0;
    if(tau < 1e-4)
        return 1.0;
    return survivalEigen(tau);
}

/*
 * Radial CDF G(x,tau) = P(r/R <= x | still inside at tau).
 *
 *   Eigenfunction series (derived from integrating the 3D radial eigenfunction
 *   expansion; integral_0^x r*sin(n*pi*r) dr = sin(n*pi*x)/(n*pi) - x*cos(n*pi*x)):
 *     G(x,tau) = (2/S) * sum_{n>=1} exp(-n^2*pi^2*tau) *
 *                [sin(n*pi*x)/(n*pi) - x*cos(n*pi*x)]
 *
 *   Free-space 3D Gaussian (small tau, boundary negligible):
 *     G(x,tau) = erf(a) - 2*a/sqrt(pi) * exp(-a^2),  a = x/(2*sqrt(tau))
 */

static double radialCdfFreeSpace(double x, double tau)
{
    if(x <= 0.0)
        return 0.0;
    if(x >= 1.0)
        return 1.0;
    double a = x / (2.0 * std::sqrt(tau));
    return clamp01(std::erf(a) - 2.0 * a * std::exp(-a * a) / std::sqrt(PI));
}

double RandomWalk::radialCdfEigen(double x, double tau) const
{
    if(x <= 0.0)
        return 0.0;
    if(x >= 1.0)
        return 1.0;
    double S = computeSurvival(tau);
    if(S < 1e-30)
        return 1.0;
    double sum = 0.0;
    for(int n = 1; n <= eigenTerms; ++n)
    {
        double arg = -static_cast<double>(n * n) * PI * PI * tau;
        if(arg < -700.0)
            break;
        double npi = static_cast<double>(n) * PI;
        double npx = npi * x;
        double term = std::exp(arg) * (std::sin(npx) / npi - x * std::cos(npx));
        sum += term;
    }
    return clamp01(2.0 * sum / S);
}

double RandomWalk::computeRadialCDF(double x, double tau) const
{
    if(tau <= 0.0)
        return (x > 0.0) ? 1.0 : 0.0;
    if(x <= 0.0)
        return 0.0;
    if(x >= 1.0)
        return 1.0;
    return (tau < tauFreeSpace) ? radialCdfFreeSpace(x, tau) : radialCdfEigen(x, tau);
}

RandomWalk::RandomWalk()
{
    tauTable.resize(tableSize);
    survivalTable.resize(tableSize);

    double logMin = std::log(tauMin);
    double logMax = std::log(tauMax);
    for(std::size_t i = 0; i < tableSize; ++i)
    {
        double frac = static_cast<double>(i) / static_cast<double>(tableSize - 1);
        double tau = std::exp(logMin + frac * (logMax - logMin));
        tauTable[i] = tau;
        survivalTable[i] = computeSurvival(tau);
    }

    static constexpr std::size_t xGridSize = 512;
    radiusTable.resize(tableSize * radiusTableSize);

    for(std::size_t i = 0; i < tableSize; ++i)
    {
        double tau = tauTable[i];
        std::vector<double> gGrid(xGridSize);
        for(std::size_t k = 0; k < xGridSize; ++k)
        {
            double x = static_cast<double>(k) / static_cast<double>(xGridSize - 1);
            gGrid[k] = computeRadialCDF(x, tau);
        }

        for(std::size_t j = 0; j < radiusTableSize; ++j)
        {
            double xi = static_cast<double>(j) / static_cast<double>(radiusTableSize - 1);
            if(xi <= gGrid[0])
            {
                radiusTable[i * radiusTableSize + j] = 0.0;
            }
            else if(xi >= gGrid[xGridSize - 1])
            {
                radiusTable[i * radiusTableSize + j] = 1.0;
            }
            else
            {
                auto it = std::lower_bound(gGrid.begin(), gGrid.end(), xi);
                std::size_t idx = static_cast<std::size_t>(it - gGrid.begin());
                if(idx == 0) idx = 1;
                double g0 = gGrid[idx - 1];
                double g1 = gGrid[idx];
                double x0 = static_cast<double>(idx - 1) / static_cast<double>(xGridSize - 1);
                double x1 = static_cast<double>(idx) / static_cast<double>(xGridSize - 1);
                double f = (g1 > g0) ? (xi - g0) / (g1 - g0) : 0.5;
                radiusTable[i * radiusTableSize + j] = x0 + f * (x1 - x0);
            }
        }
    }
}

double RandomWalk::sampleLeakTime(double xi) const
{
    double target = 1.0 - xi;
    if(target >= survivalTable.front())
        return tauTable.front();
    if(target <= survivalTable.back())
        return tauTable.back();

    auto it = std::lower_bound(survivalTable.begin(), survivalTable.end(), target,
                               [](double a, double b) { return a > b; });
    if(it == survivalTable.begin())
        return tauTable.front();
    if(it == survivalTable.end())
        return tauTable.back();

    std::size_t idx = static_cast<std::size_t>(it - survivalTable.begin());
    double S0 = survivalTable[idx - 1];
    double S1 = survivalTable[idx];
    double t0 = tauTable[idx - 1];
    double t1 = tauTable[idx];
    double frac = (S0 - target) / (S0 - S1);
    double logT = std::log(t0) + frac * (std::log(t1) - std::log(t0));
    return std::exp(logT);
}

double RandomWalk::sampleRadius(double tau, double xi) const
{
    double logTau = std::log(std::clamp(tau, tauMin, tauMax));
    double logMin = std::log(tauMin);
    double logMax = std::log(tauMax);
    double tauPos = (logTau - logMin) / (logMax - logMin) * static_cast<double>(tableSize - 1);
    std::size_t tauIdx = std::min(static_cast<std::size_t>(tauPos), tableSize - 2);
    double tauFrac = tauPos - static_cast<double>(tauIdx);

    double xiClamped = std::clamp(xi, 0.0, 1.0);
    double xiPos = xiClamped * static_cast<double>(radiusTableSize - 1);
    std::size_t xiIdx = std::min(static_cast<std::size_t>(xiPos), radiusTableSize - 2);
    double xiFrac = xiPos - static_cast<double>(xiIdx);

    double r00 = radiusTable[tauIdx * radiusTableSize + xiIdx];
    double r01 = radiusTable[tauIdx * radiusTableSize + xiIdx + 1];
    double r10 = radiusTable[(tauIdx + 1) * radiusTableSize + xiIdx];
    double r11 = radiusTable[(tauIdx + 1) * radiusTableSize + xiIdx + 1];

    double r0 = r00 + xiFrac * (r01 - r00);
    double r1 = r10 + xiFrac * (r11 - r10);

    return std::clamp(r0 + tauFrac * (r1 - r0), 0.0, 1.0);
}
