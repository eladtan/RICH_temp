#include "RadiationIMC.hpp"
#include "SphericalObserver.hpp"
#include "PostProcessIMCHelpers.hpp"
#include "IMCPolarization.hpp"
#include "mpi/mpi_commands_3d.hpp"
#include "Radiation/conj_grad_solve.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>

// #define MONTECARLO_EPS 1e-7

namespace {
    const char *ComptonInducedModeName(ComptonInducedMode mode)
    {
        switch(mode)
        {
            case ComptonInducedMode::RadiationField:
                return "radiation-field";
            case ComptonInducedMode::AdaptivePlanckFallback:
                return "adaptive-planck-fallback";
        }
        return "unknown";
    }

    inline void ClampFrequencyToBounds(double &frequency)
    {
        frequency = std::clamp(frequency,
            ComputationalCell3D::energyBoundaries[0],
            ComputationalCell3D::energyBoundaries[ENERGY_GROUPS_NUM]);
    }

    inline void SetInitialWeightFromWeight(RadiationIMC::Particle &particle)
    {
        particle.initialWeight = std::abs(particle.weight);
    }

#if defined(__GNUC__) || defined(__clang__)
    __attribute__((noinline))
#endif
    double SafeGroupWeightCorrection(double physicalPdf, double samplingPdf)
    {
        if(!(samplingPdf > 0.0) || !std::isfinite(samplingPdf))
            return std::numeric_limits<double>::quiet_NaN();
        volatile double denominator = samplingPdf;
        return physicalPdf / denominator;
    }

    std::vector<double> BuildComptonTemperatures()
    {
        std::vector<double> temperatures;
        temperatures.reserve(131);
        temperatures.push_back(0.0001 * units::kev_kelvin);
        temperatures.push_back(0.001 * units::kev_kelvin);
        temperatures.push_back(0.005 * units::kev_kelvin);
        for(size_t i = 0; i < 128; i++)
        {
            double const x = -2.0 + 6.0 * static_cast<double>(i) / 127.0;
            temperatures.push_back(std::pow(10.0, x) * units::kev_kelvin);
        }
        return temperatures;
    }

    std::vector<double> BuildComptonCentersVector()
    {
        std::vector<double> centers(ENERGY_GROUPS_NUM, 0.0);
        for(size_t g = 0; g < ENERGY_GROUPS_NUM; g++)
        {
            centers[g] = 0.5 * (ComputationalCell3D::energyBoundaries[g] +
                                ComputationalCell3D::energyBoundaries[g + 1]);
        }
        return centers;
    }

    std::vector<double> BuildComptonBoundariesVector()
    {
        std::vector<double> boundaries(ENERGY_GROUPS_NUM + 1, 0.0);
        for(size_t g = 0; g <= ENERGY_GROUPS_NUM; g++)
        {
            boundaries[g] = ComputationalCell3D::energyBoundaries[g];
        }
        return boundaries;
    }

    template<class MatrixLike>
    void ZeroGroupMatrix(MatrixLike &matrix)
    {
        for(auto &row : matrix)
        {
            row.fill(0.0);
        }
    }

    struct SolverDiagnostics
    {
        double minPivot = std::numeric_limits<double>::infinity();
        double maxCoeff = 0.0;
    };

    bool SolveComptonGroupSystem(RadiationIMC::GroupMatrix matrix,
                                 RadiationIMC::GroupArray rhs,
                                 RadiationIMC::GroupArray &solution,
                                 SolverDiagnostics &diag)
    {
        diag.minPivot = std::numeric_limits<double>::infinity();
        diag.maxCoeff = 0.0;
        for(size_t r = 0; r < ENERGY_GROUPS_NUM; r++)
            for(size_t c = 0; c < ENERGY_GROUPS_NUM; c++)
                diag.maxCoeff = std::max(diag.maxCoeff, std::abs(matrix[r][c]));

        for(size_t col = 0; col < ENERGY_GROUPS_NUM; col++)
        {
            size_t pivot = col;
            double pivotAbs = std::abs(matrix[col][col]);
            for(size_t row = col + 1; row < ENERGY_GROUPS_NUM; row++)
            {
                double const candidateAbs = std::abs(matrix[row][col]);
                if(candidateAbs > pivotAbs)
                {
                    pivot = row;
                    pivotAbs = candidateAbs;
                }
            }
            if(pivotAbs <= 1e-200)
                return false;
            diag.minPivot = std::min(diag.minPivot, pivotAbs);
            if(pivot != col)
            {
                std::swap(matrix[pivot], matrix[col]);
                std::swap(rhs[pivot], rhs[col]);
            }

            double const pivotValue = matrix[col][col];
            for(size_t row = col + 1; row < ENERGY_GROUPS_NUM; row++)
            {
                double const factor = matrix[row][col] / pivotValue;
                if(factor == 0.0)
                    continue;
                matrix[row][col] = 0.0;
                for(size_t j = col + 1; j < ENERGY_GROUPS_NUM; j++)
                    matrix[row][j] -= factor * matrix[col][j];
                rhs[row] -= factor * rhs[col];
            }
        }

        solution.fill(0.0);
        for(size_t rev = 0; rev < ENERGY_GROUPS_NUM; rev++)
        {
            size_t const row = ENERGY_GROUPS_NUM - 1 - rev;
            double value = rhs[row];
            for(size_t col = row + 1; col < ENERGY_GROUPS_NUM; col++)
                value -= matrix[row][col] * solution[col];
            solution[row] = value / matrix[row][row];
            if(!std::isfinite(solution[row]))
                return false;
        }
        return true;
    }
    struct BoundedSolverDiagnostics
    {
        bool used = false;
        bool directSolveFailed = false;
        bool materialCapActive = false;
        size_t iterations = 0;
        size_t activeVariables = 0;
        double minUnconstrained = 0.0;
        double relativeResidual = 0.0;
        double absoluteResidual = 0.0;
        double unweightedRelativeResidual = 0.0;
        double maxGroupResidualFraction = 0.0;
        double maxBoundViolation = 0.0;
        double sumEnergy = 0.0;
        double materialCap = 0.0;
        double capLambda = 0.0;
        double capEqualityResidual = 0.0;
        double capLambdaSpread = 0.0;
        double minPassivePivot = std::numeric_limits<double>::infinity();
        double maxDirectDeviationFraction = 0.0;
        size_t maxDirectDeviationGroup = 0;
    };

    using GroupMask = std::array<bool, ENERGY_GROUPS_NUM>;

    static double Dot(RadiationIMC::GroupArray const &a, RadiationIMC::GroupArray const &b)
    {
        double s = 0.0;
        for(size_t g = 0; g < ENERGY_GROUPS_NUM; g++)
            s += a[g] * b[g];
        return s;
    }

    static double Norm2(RadiationIMC::GroupArray const &a)
    {
        return std::sqrt(Dot(a, a));
    }

    static void MatVec(RadiationIMC::GroupMatrix const &A,
                       RadiationIMC::GroupArray const &x,
                       RadiationIMC::GroupArray &y)
    {
        for(size_t g = 0; g < ENERGY_GROUPS_NUM; g++)
        {
            y[g] = 0.0;
            for(size_t h = 0; h < ENERGY_GROUPS_NUM; h++)
                y[g] += A[g][h] * x[h];
        }
    }

    static double SumGroups(RadiationIMC::GroupArray const &x)
    {
        double s = 0.0;
        for(size_t g = 0; g < ENERGY_GROUPS_NUM; g++)
            s += x[g];
        return s;
    }

    static double MinGroups(RadiationIMC::GroupArray const &x)
    {
        double m = x[0];
        for(size_t g = 1; g < ENERGY_GROUPS_NUM; g++)
            m = std::min(m, x[g]);
        return m;
    }

    static double MaxAbsGroups(RadiationIMC::GroupArray const &x)
    {
        double m = 0.0;
        for(size_t g = 0; g < ENERGY_GROUPS_NUM; g++)
            m = std::max(m, std::abs(x[g]));
        return m;
    }

    static bool SolvePassiveSubsystem(
        RadiationIMC::GroupMatrix const &WA,
        RadiationIMC::GroupArray const &Wd,
        size_t const passiveIndices[],
        size_t nPassive,
        double z[],
        double &outMinPivot)
    {
        constexpr size_t N = ENERGY_GROUPS_NUM;
        std::array<std::array<double, N>, N> H{};
        std::array<double, N> q{};
        outMinPivot = std::numeric_limits<double>::infinity();

        for(size_t a = 0; a < nPassive; a++)
        {
            size_t const ia = passiveIndices[a];
            for(size_t b = 0; b <= a; b++)
            {
                double hval = 0.0;
                size_t const ib = passiveIndices[b];
                for(size_t r = 0; r < N; r++)
                    hval += WA[r][ia] * WA[r][ib];
                H[a][b] = hval;
                H[b][a] = hval;
            }
            double qval = 0.0;
            for(size_t r = 0; r < N; r++)
                qval += WA[r][ia] * Wd[r];
            q[a] = qval;
        }

        for(size_t col = 0; col < nPassive; col++)
        {
            size_t pivot = col;
            double pivotAbs = std::abs(H[col][col]);
            for(size_t row = col + 1; row < nPassive; row++)
            {
                double const cand = std::abs(H[row][col]);
                if(cand > pivotAbs) { pivot = row; pivotAbs = cand; }
            }
            if(pivotAbs <= 1e-200)
                return false;
            outMinPivot = std::min(outMinPivot, pivotAbs);
            if(pivot != col)
            {
                std::swap(H[pivot], H[col]);
                std::swap(q[pivot], q[col]);
            }
            double const pv = H[col][col];
            for(size_t row = col + 1; row < nPassive; row++)
            {
                double const factor = H[row][col] / pv;
                H[row][col] = 0.0;
                for(size_t j = col + 1; j < nPassive; j++)
                    H[row][j] -= factor * H[col][j];
                q[row] -= factor * q[col];
            }
        }

        for(size_t rev = 0; rev < nPassive; rev++)
        {
            size_t const row = nPassive - 1 - rev;
            double val = q[row];
            for(size_t col = row + 1; col < nPassive; col++)
                val -= H[row][col] * z[col];
            z[row] = val / H[row][row];
            if(!std::isfinite(z[row]))
                return false;
        }
        return true;
    }

    static bool SolvePassiveWithCap(
        RadiationIMC::GroupMatrix const &WA,
        RadiationIMC::GroupArray const &Wd,
        size_t const passiveIndices[],
        size_t nPassive,
        double capValue,
        double z[],
        double &lambda,
        double &outMinPivot)
    {
        constexpr size_t N = ENERGY_GROUPS_NUM;
        size_t const n = nPassive + 1;
        std::array<std::array<double, N + 1>, N + 1> M{};
        std::array<double, N + 1> b{};
        outMinPivot = std::numeric_limits<double>::infinity();

        for(size_t a = 0; a < nPassive; a++)
        {
            size_t const ia = passiveIndices[a];
            for(size_t bb = 0; bb <= a; bb++)
            {
                double hval = 0.0;
                size_t const ib = passiveIndices[bb];
                for(size_t r = 0; r < N; r++)
                    hval += WA[r][ia] * WA[r][ib];
                M[a][bb] = hval;
                M[bb][a] = hval;
            }
            double qval = 0.0;
            for(size_t r = 0; r < N; r++)
                qval += WA[r][ia] * Wd[r];
            b[a] = qval;
            M[a][nPassive] = 1.0;
            M[nPassive][a] = 1.0;
        }
        M[nPassive][nPassive] = 0.0;
        b[nPassive] = capValue;

        for(size_t col = 0; col < n; col++)
        {
            size_t pivot = col;
            double pivotAbs = std::abs(M[col][col]);
            for(size_t row = col + 1; row < n; row++)
            {
                double const cand = std::abs(M[row][col]);
                if(cand > pivotAbs) { pivot = row; pivotAbs = cand; }
            }
            if(pivotAbs <= 1e-200)
                return false;
            outMinPivot = std::min(outMinPivot, pivotAbs);
            if(pivot != col)
            {
                std::swap(M[pivot], M[col]);
                std::swap(b[pivot], b[col]);
            }
            double const pv = M[col][col];
            for(size_t row = col + 1; row < n; row++)
            {
                double const factor = M[row][col] / pv;
                M[row][col] = 0.0;
                for(size_t j = col + 1; j < n; j++)
                    M[row][j] -= factor * M[col][j];
                b[row] -= factor * b[col];
            }
        }

        std::array<double, N + 1> sol{};
        for(size_t rev = 0; rev < n; rev++)
        {
            size_t const row = n - 1 - rev;
            double val = b[row];
            for(size_t col = row + 1; col < n; col++)
                val -= M[row][col] * sol[col];
            sol[row] = val / M[row][row];
            if(!std::isfinite(sol[row]))
                return false;
        }

        for(size_t a = 0; a < nPassive; a++)
            z[a] = sol[a];
        lambda = sol[nPassive];
        return true;
    }

    static bool SolveNNLS(
        RadiationIMC::GroupMatrix const &WA,
        RadiationIMC::GroupArray const &Wd,
        RadiationIMC::GroupArray &solution,
        size_t &totalIterations,
        double &outMinPivot)
    {
        constexpr size_t N = ENERGY_GROUPS_NUM;
        constexpr size_t maxIter = 3 * N * N + N;
        outMinPivot = std::numeric_limits<double>::infinity();

        GroupMask passive{};
        RadiationIMC::GroupArray x{};
        x.fill(0.0);
        passive.fill(false);

        RadiationIMC::GroupArray w0{};
        for(size_t g = 0; g < N; g++)
            for(size_t r = 0; r < N; r++)
                w0[g] += WA[r][g] * Wd[r];
        double maxInitialGrad = 0.0;
        for(size_t g = 0; g < N; g++)
            maxInitialGrad = std::max(maxInitialGrad, std::abs(w0[g]));
        double const gradTol = 1e-12 * std::max(1.0, maxInitialGrad);

        totalIterations = 0;
        while(totalIterations < maxIter)
        {
            RadiationIMC::GroupArray Ax{};
            MatVec(WA, x, Ax);
            RadiationIMC::GroupArray w{};
            for(size_t g = 0; g < N; g++)
                for(size_t r = 0; r < N; r++)
                    w[g] += WA[r][g] * (Wd[r] - Ax[r]);

            double maxGradZ = -std::numeric_limits<double>::infinity();
            size_t maxGradIdx = 0;
            bool anyZero = false;
            for(size_t g = 0; g < N; g++)
            {
                if(!passive[g])
                {
                    anyZero = true;
                    if(w[g] > maxGradZ) { maxGradZ = w[g]; maxGradIdx = g; }
                }
            }
            if(!anyZero || maxGradZ <= gradTol)
            {
                solution = x;
                return true;
            }

            passive[maxGradIdx] = true;

            for(;;)
            {
                totalIterations++;
                if(totalIterations >= maxIter) { solution = x; return false; }

                size_t passiveIndices[N];
                size_t nP = 0;
                for(size_t g = 0; g < N; g++)
                    if(passive[g]) passiveIndices[nP++] = g;

                if(nP == 0) { x.fill(0.0); break; }

                double z[N]{};
                double subPivot = 0.0;
                if(!SolvePassiveSubsystem(WA, Wd, passiveIndices, nP, z, subPivot))
                {
                    solution = x;
                    return false;
                }
                outMinPivot = std::min(outMinPivot, subPivot);

                bool allPositive = true;
                for(size_t a = 0; a < nP; a++)
                    if(z[a] <= 0.0) { allPositive = false; break; }

                if(allPositive)
                {
                    for(size_t g = 0; g < N; g++) x[g] = 0.0;
                    for(size_t a = 0; a < nP; a++) x[passiveIndices[a]] = z[a];
                    break;
                }

                double alpha = std::numeric_limits<double>::infinity();
                for(size_t a = 0; a < nP; a++)
                {
                    if(z[a] <= 0.0)
                    {
                        double const xg = x[passiveIndices[a]];
                        double const denom = xg - z[a];
                        if(denom > 0.0)
                            alpha = std::min(alpha, xg / denom);
                    }
                }
                if(!std::isfinite(alpha) || alpha <= 0.0) alpha = 0.0;

                for(size_t a = 0; a < nP; a++)
                {
                    size_t const g = passiveIndices[a];
                    x[g] += alpha * (z[a] - x[g]);
                    if(x[g] <= 0.0) { x[g] = 0.0; passive[g] = false; }
                }
            }
        }
        solution = x;
        return false;
    }

    static bool SolveCapConstrainedNNLS(
        RadiationIMC::GroupMatrix const &WA,
        RadiationIMC::GroupArray const &Wd,
        RadiationIMC::GroupArray const &nnlsSolution,
        double capValue,
        RadiationIMC::GroupArray &solution,
        size_t &totalIterations,
        double &outMinPivot)
    {
        constexpr size_t N = ENERGY_GROUPS_NUM;
        constexpr size_t maxIter = 3 * N * N + N;
        outMinPivot = std::numeric_limits<double>::infinity();

        if(capValue <= 0.0) { solution.fill(0.0); totalIterations = 0; return true; }

        RadiationIMC::GroupArray w0c{};
        for(size_t g = 0; g < N; g++)
            for(size_t r = 0; r < N; r++)
                w0c[g] += WA[r][g] * Wd[r];
        double maxInitGradCap = 0.0;
        for(size_t g = 0; g < N; g++)
            maxInitGradCap = std::max(maxInitGradCap, std::abs(w0c[g]));
        double const gradTol = 1e-12 * std::max(1.0, maxInitGradCap);

        GroupMask passive{};
        size_t nPassiveInit = 0;
        for(size_t g = 0; g < N; g++)
        {
            passive[g] = (nnlsSolution[g] > 0.0);
            if(passive[g]) nPassiveInit++;
        }
        if(nPassiveInit == 0)
        {
            RadiationIMC::GroupArray w{};
            for(size_t g = 0; g < N; g++)
                for(size_t r = 0; r < N; r++)
                    w[g] += WA[r][g] * Wd[r];
            size_t maxG = 0;
            for(size_t g = 1; g < N; g++)
                if(w[g] > w[maxG]) maxG = g;
            passive[maxG] = true;
        }

        totalIterations = 0;
        RadiationIMC::GroupArray x{};

        while(totalIterations < maxIter)
        {
            totalIterations++;
            size_t passiveIndices[N];
            size_t nP = 0;
            for(size_t g = 0; g < N; g++)
                if(passive[g]) passiveIndices[nP++] = g;
            if(nP == 0) { solution.fill(0.0); return false; }

            double z[N]{};
            double lambda = 0.0;
            double capSubPivot = 0.0;
            if(!SolvePassiveWithCap(WA, Wd, passiveIndices, nP, capValue, z, lambda, capSubPivot))
            {
                solution = x;
                return false;
            }
            outMinPivot = std::min(outMinPivot, capSubPivot);

            bool needRemove = true;
            while(needRemove && totalIterations < maxIter)
            {
                needRemove = false;
                double worstZ = 0.0;
                size_t worstA = 0;
                for(size_t a = 0; a < nP; a++)
                    if(z[a] < worstZ) { worstZ = z[a]; worstA = a; }
                if(worstZ < 0.0)
                {
                    totalIterations++;
                    passive[passiveIndices[worstA]] = false;
                    nP = 0;
                    for(size_t g = 0; g < N; g++)
                        if(passive[g]) passiveIndices[nP++] = g;
                    if(nP == 0) { solution.fill(0.0); return false; }
                    if(!SolvePassiveWithCap(WA, Wd, passiveIndices, nP, capValue, z, lambda, capSubPivot))
                    {
                        solution = x;
                        return false;
                    }
                    outMinPivot = std::min(outMinPivot, capSubPivot);
                    needRemove = true;
                }
            }

            x.fill(0.0);
            for(size_t a = 0; a < nP; a++)
                x[passiveIndices[a]] = z[a];

            RadiationIMC::GroupArray Ax{};
            MatVec(WA, x, Ax);
            RadiationIMC::GroupArray grad{};
            for(size_t g = 0; g < N; g++)
                for(size_t r = 0; r < N; r++)
                    grad[g] += WA[r][g] * (Ax[r] - Wd[r]);

            double worstViolation = 0.0;
            size_t worstViolIdx = 0;
            for(size_t g = 0; g < N; g++)
            {
                if(!passive[g])
                {
                    double const kktVal = grad[g] + lambda;
                    if(kktVal < -gradTol && kktVal < worstViolation)
                    {
                        worstViolation = kktVal;
                        worstViolIdx = g;
                    }
                }
            }
            if(worstViolation >= -gradTol)
            {
                solution = x;
                return true;
            }
            passive[worstViolIdx] = true;
        }
        solution = x;
        return false;
    }

    [[maybe_unused]] static bool SolveBoundedComptonCorrection(
        RadiationIMC::GroupMatrix const &A,
        RadiationIMC::GroupArray const &rhs,
        RadiationIMC::GroupArray const &rawGroupEnergy,
        RadiationIMC::GroupArray const &solveInputGroupEnergy,
        RadiationIMC::GroupArray const &supportFloorEnergy,
        double materialCap,
        RadiationIMC::GroupArray const &initialGuess,
        RadiationIMC::GroupArray &solution,
        BoundedSolverDiagnostics &bdiag)
    {
        constexpr size_t N = ENERGY_GROUPS_NUM;
        bdiag.used = true;
        bdiag.materialCap = materialCap;

        if(std::isfinite(materialCap) && materialCap < 0.0)
            return false;

        double const cellEnergyScale = std::max({
            1.0,
            SumGroups(rawGroupEnergy),
            SumGroups(solveInputGroupEnergy),
            Norm2(rhs),
            MaxAbsGroups(rhs),
            MaxAbsGroups(rawGroupEnergy),
            MaxAbsGroups(solveInputGroupEnergy)
        });
        constexpr double boundedResidualFloorFrac = 1e-6;
        double const residualAbsFloor = std::max(1.0, boundedResidualFloorFrac * cellEnergyScale);

        RadiationIMC::GroupArray W{};
        for(size_t g = 0; g < N; g++)
        {
            double rowL1 = 0.0;
            for(size_t h = 0; h < N; h++)
                rowL1 += std::abs(A[g][h]);
            double const denom = std::max({std::abs(rhs[g]),
                                           rawGroupEnergy[g],
                                           solveInputGroupEnergy[g],
                                           supportFloorEnergy[g],
                                           rowL1 * cellEnergyScale * 1e-12,
                                           residualAbsFloor});
            W[g] = 1.0 / denom;
        }

        RadiationIMC::GroupMatrix WA{};
        RadiationIMC::GroupArray Wd{};
        for(size_t g = 0; g < N; g++)
        {
            for(size_t h = 0; h < N; h++)
                WA[g][h] = W[g] * A[g][h];
            Wd[g] = W[g] * rhs[g];
        }

        size_t nnlsIter = 0;
        double nnlsMinPivot = 0.0;
        RadiationIMC::GroupArray nnlsSolution{};
        if(!SolveNNLS(WA, Wd, nnlsSolution, nnlsIter, nnlsMinPivot))
        {
            solution = nnlsSolution;
            bdiag.iterations = nnlsIter;
            bdiag.minPassivePivot = nnlsMinPivot;
            return false;
        }
        bdiag.iterations = nnlsIter;
        bdiag.minPassivePivot = nnlsMinPivot;

        double const capTol = 1e-10 * std::max(1.0, std::abs(materialCap));
        if(!std::isfinite(materialCap) || SumGroups(nnlsSolution) <= materialCap + capTol)
        {
            solution = nnlsSolution;
            bdiag.materialCapActive = false;
        }
        else
        {
            size_t capIter = 0;
            double capMinPivot = 0.0;
            if(!SolveCapConstrainedNNLS(WA, Wd, nnlsSolution, materialCap, solution, capIter, capMinPivot))
            {
                bdiag.iterations += capIter;
                bdiag.minPassivePivot = std::min(bdiag.minPassivePivot, capMinPivot);
                return false;
            }
            bdiag.iterations += capIter;
            bdiag.minPassivePivot = std::min(bdiag.minPassivePivot, capMinPivot);
            bdiag.materialCapActive = true;

            bdiag.capEqualityResidual = std::abs(SumGroups(solution) - materialCap);
            RadiationIMC::GroupArray capAx{};
            MatVec(WA, solution, capAx);
            double lambdaMin = std::numeric_limits<double>::infinity();
            double lambdaMax = -std::numeric_limits<double>::infinity();
            double const activeTol = 1e-14 * cellEnergyScale;
            for(size_t g = 0; g < N; g++)
            {
                if(solution[g] > activeTol)
                {
                    double grad_g = 0.0;
                    for(size_t r = 0; r < N; r++)
                        grad_g += WA[r][g] * (capAx[r] - Wd[r]);
                    double const impliedLambda = -grad_g;
                    lambdaMin = std::min(lambdaMin, impliedLambda);
                    lambdaMax = std::max(lambdaMax, impliedLambda);
                }
            }
            if(std::isfinite(lambdaMin))
            {
                bdiag.capLambda = 0.5 * (lambdaMin + lambdaMax);
                bdiag.capLambdaSpread = lambdaMax - lambdaMin;
            }

            double const capEqTol = 1e-10 * std::max(1.0, std::abs(materialCap));
            if(bdiag.capEqualityResidual > capEqTol)
                return false;
            double const capGradScale = 1e-12 * std::max(1.0, MaxAbsGroups(Wd));
            if(bdiag.capLambda < -capGradScale)
                return false;
            double const capSpreadTol = 1e-8 * std::max({1.0,
                std::abs(bdiag.capLambda), MaxAbsGroups(Wd)});
            if(bdiag.capLambdaSpread > capSpreadTol)
                return false;
        }

        double const boundTol = 1e-12 * cellEnergyScale;
        bdiag.maxBoundViolation = 0.0;
        bdiag.activeVariables = 0;
        for(size_t g = 0; g < N; g++)
        {
            if(solution[g] < 0.0)
            {
                bdiag.maxBoundViolation = std::max(bdiag.maxBoundViolation, -solution[g]);
                if(solution[g] < -boundTol)
                    return false;
                solution[g] = 0.0;
            }
            if(solution[g] > 0.0)
                bdiag.activeVariables++;
            if(!std::isfinite(solution[g]))
                return false;
        }

        bdiag.sumEnergy = SumGroups(solution);
        if(std::isfinite(materialCap) && bdiag.sumEnergy > materialCap + capTol)
            return false;

        bdiag.minUnconstrained = MinGroups(initialGuess);
        if(!std::isfinite(bdiag.minPassivePivot))
            bdiag.minPassivePivot = 0.0;

        RadiationIMC::GroupArray Ax{};
        MatVec(A, solution, Ax);
        RadiationIMC::GroupArray resid{};
        RadiationIMC::GroupArray wResid{};
        RadiationIMC::GroupArray wRhs{};
        bdiag.maxGroupResidualFraction = 0.0;
        for(size_t g = 0; g < N; g++)
        {
            resid[g] = Ax[g] - rhs[g];
            wResid[g] = W[g] * resid[g];
            wRhs[g] = W[g] * rhs[g];
            double const groupDenom = std::max({std::abs(rhs[g]),
                rawGroupEnergy[g], solveInputGroupEnergy[g], supportFloorEnergy[g],
                residualAbsFloor});
            bdiag.maxGroupResidualFraction = std::max(bdiag.maxGroupResidualFraction,
                                                       std::abs(resid[g]) / groupDenom);
        }

        bdiag.absoluteResidual = Norm2(wResid);
        bdiag.relativeResidual = bdiag.absoluteResidual / std::max(Norm2(wRhs), 1.0);
        double const unweightedResidNorm = Norm2(resid);
        bdiag.unweightedRelativeResidual = unweightedResidNorm / std::max(Norm2(rhs), cellEnergyScale);

        return true;
    }

    struct ComptonSubcycleDiagnostics
    {
        bool used = false;
        bool success = false;
        bool earlyStopped = false;
        bool usedPartialCorrection = false;
        bool returnedRawNoCorrection = false;

        size_t subcycles = 0;
        size_t rejectedSteps = 0;
        size_t directSolveFailures = 0;

        double consumedFraction = 0.0;
        double minAcceptedFraction = std::numeric_limits<double>::infinity();
        double maxAcceptedFraction = 0.0;
        double finalMinGroupEnergy = 0.0;
        double finalRadiationDelta = 0.0;
        double finalMaterialDeposit = 0.0;
        double maxNegativeTrialMass = 0.0;
        double worstTrialMin = std::numeric_limits<double>::infinity();
    };

    struct SubcycleTrialResult
    {
        bool finite = false;
        bool positive = false;
        bool capOk = false;
        bool acceptable = false;

        RadiationIMC::GroupArray Etrial{};
        double minValue = 0.0;
        double negativeMass = 0.0;
        double sumEnergy = 0.0;
    };

    static SubcycleTrialResult EvaluateSubcycleTrial(
        RadiationIMC::GroupArray const &Etrial,
        double materialCap,
        double cellEnergyScale)
    {
        SubcycleTrialResult result;
        result.Etrial = Etrial;
        result.finite = true;
        result.minValue = MinGroups(Etrial);
        result.sumEnergy = SumGroups(Etrial);
        result.negativeMass = 0.0;

        constexpr double tinyNegFrac = 1e-13;
        constexpr double tinyNegMassFrac = 1e-12;
        double const tinyNegAbs = tinyNegFrac * std::max(1.0, cellEnergyScale);
        double const tinyNegMass = tinyNegMassFrac * std::max(1.0, cellEnergyScale);

        for(size_t g = 0; g < ENERGY_GROUPS_NUM; ++g)
        {
            if(!std::isfinite(Etrial[g]))
                result.finite = false;
            if(Etrial[g] < 0.0)
                result.negativeMass += -Etrial[g];
        }

        result.positive =
            result.finite &&
            result.minValue >= -tinyNegAbs &&
            result.negativeMass <= tinyNegMass;

        double const materialCapTol = std::isfinite(materialCap)
            ? 1e-12 * std::max(1.0, std::abs(materialCap))
            : std::numeric_limits<double>::infinity();

        result.capOk =
            !std::isfinite(materialCap) ||
            result.sumEnergy <= materialCap + materialCapTol;

        result.acceptable = result.positive && result.capOk;
        return result;
    }

    static void ClampTinyNegativeGroups(
        RadiationIMC::GroupArray &E,
        double cellEnergyScale)
    {
        constexpr double tinyNegFrac = 1e-13;
        double const tinyNegAbs = tinyNegFrac * std::max(1.0, cellEnergyScale);

        for(double &v : E)
        {
            if(v < 0.0 && v >= -tinyNegAbs)
                v = 0.0;
        }
    }

    static double ComputeSubcycleShrinkFraction(
        RadiationIMC::GroupArray const &Ecurrent,
        RadiationIMC::GroupArray const &Etrial,
        double currentFraction,
        double materialCap)
    {
        double theta = 1.0;

        for(size_t g = 0; g < ENERGY_GROUPS_NUM; ++g)
        {
            if(Etrial[g] < 0.0)
            {
                double const denom = Ecurrent[g] - Etrial[g];
                if(denom > 0.0)
                    theta = std::min(theta, Ecurrent[g] / denom);
                else
                    theta = 0.0;
            }
        }

        if(std::isfinite(materialCap))
        {
            double const sumCurrent = SumGroups(Ecurrent);
            double const sumTrial = SumGroups(Etrial);
            if(sumTrial > materialCap)
            {
                double const dsum = sumTrial - sumCurrent;
                if(dsum > 0.0)
                    theta = std::min(theta, (materialCap - sumCurrent) / dsum);
                else
                    theta = 0.0;
            }
        }

        constexpr double safety = 0.8;
        constexpr double minShrink = 0.1;
        constexpr double maxShrink = 0.7;

        double const rawNew = safety * theta * currentFraction;
        return std::clamp(rawNew, minShrink * currentFraction, maxShrink * currentFraction);
    }

    static bool SolveComptonCorrectionByAdaptiveSubcycling(
        RadiationIMC::GroupArray const &rawGroupEnergy,
        RadiationIMC::GroupArray const &solveInputGroupEnergy,
        RadiationIMC::GroupArray const &Btotal,
        RadiationIMC::GroupMatrix const &residualKernel,
        double fullDt,
        double materialCap,
        double cellEnergyScale,
        RadiationIMC::GroupArray &solution,
        ComptonSubcycleDiagnostics &sdiag)
    {
        sdiag.used = true;

        constexpr size_t maxSubcycles = 256;
        constexpr double minFraction = 1e-12;
        constexpr double growFactor = 1.5;

        RadiationIMC::GroupArray E = solveInputGroupEnergy;
        ClampTinyNegativeGroups(E, cellEnergyScale);

        if(MinGroups(E) < 0.0)
            return false;

        double const materialCapTol = std::isfinite(materialCap)
            ? 1e-12 * std::max(1.0, std::abs(materialCap))
            : std::numeric_limits<double>::infinity();

        if(std::isfinite(materialCap) &&
           SumGroups(E) > materialCap + materialCapTol)
        {
            solution = rawGroupEnergy;
            sdiag.earlyStopped = true;
            sdiag.usedPartialCorrection = true;
            sdiag.returnedRawNoCorrection = true;
            sdiag.success = true;
            sdiag.consumedFraction = 0.0;
            sdiag.finalMinGroupEnergy = MinGroups(solution);
            sdiag.finalRadiationDelta = 0.0;
            sdiag.finalMaterialDeposit = 0.0;
            return true;
        }

        double tau = 0.0;
        double fraction = 1.0;

        while(tau < 1.0 - 1e-14 && sdiag.subcycles < maxSubcycles)
        {
            fraction = std::min(fraction, 1.0 - tau);

            RadiationIMC::GroupMatrix A{};
            RadiationIMC::GroupArray rhs_sub{};

            for(size_t g = 0; g < ENERGY_GROUPS_NUM; ++g)
            {
                rhs_sub[g] = E[g] + fraction * Btotal[g];

                for(size_t h = 0; h < ENERGY_GROUPS_NUM; ++h)
                {
                    A[g][h] = ((g == h) ? 1.0 : 0.0)
                        - fraction * fullDt * units::clight * residualKernel[h][g];
                }
            }

            SolverDiagnostics localDiag;
            RadiationIMC::GroupArray Etrial{};
            bool const solveOk = SolveComptonGroupSystem(A, rhs_sub, Etrial, localDiag);

            if(!solveOk)
            {
                sdiag.directSolveFailures++;
                sdiag.rejectedSteps++;
                fraction *= 0.5;
                if(fraction < minFraction)
                    break;
                continue;
            }

            SubcycleTrialResult trial =
                EvaluateSubcycleTrial(Etrial, materialCap, cellEnergyScale);

            sdiag.maxNegativeTrialMass =
                std::max(sdiag.maxNegativeTrialMass, trial.negativeMass);
            sdiag.worstTrialMin =
                std::min(sdiag.worstTrialMin, trial.minValue);

            if(trial.acceptable)
            {
                ClampTinyNegativeGroups(Etrial, cellEnergyScale);
                E = Etrial;
                tau += fraction;

                sdiag.subcycles++;
                sdiag.consumedFraction = tau;
                sdiag.minAcceptedFraction =
                    std::min(sdiag.minAcceptedFraction, fraction);
                sdiag.maxAcceptedFraction =
                    std::max(sdiag.maxAcceptedFraction, fraction);

                fraction = std::min(growFactor * fraction, 1.0 - tau);
                continue;
            }

            sdiag.rejectedSteps++;
            double const newFraction =
                ComputeSubcycleShrinkFraction(E, Etrial, fraction, materialCap);

            if(newFraction < minFraction || newFraction >= 0.99 * fraction)
                break;

            fraction = newFraction;
        }

        // If only a microscopic correction fraction was accepted, return the
        // physical endpoint spectrum rather than the support-floor-conditioned
        // solve input.  This prevents numerical support floors from reshaping
        // the spectrum when the Compton correction could not make meaningful
        // progress.
        constexpr double minUsefulConsumedFraction = 1e-6;

        if(tau >= 1.0 - 1e-12)
        {
            solution = E;
            sdiag.success = true;
        }
        else if(sdiag.subcycles > 0 && tau >= minUsefulConsumedFraction)
        {
            solution = E;
            sdiag.earlyStopped = true;
            sdiag.usedPartialCorrection = true;
            sdiag.success = true;
        }
        else
        {
            solution = rawGroupEnergy;
            sdiag.earlyStopped = true;
            sdiag.usedPartialCorrection = true;
            sdiag.returnedRawNoCorrection = true;
            sdiag.success = true;
        }

        sdiag.finalMinGroupEnergy = MinGroups(solution);
        sdiag.finalRadiationDelta = SumGroups(solution) - SumGroups(rawGroupEnergy);
        sdiag.finalMaterialDeposit = -sdiag.finalRadiationDelta;
        return true;
    }
}

    RadiationIMC::RadiationIMC(Tessellation3D &grid, const std::shared_ptr<BoundaryCond> &boundary, std::vector<ComputationalCell3D> &cells, std::vector<Conserved3D> &conserved, std::shared_ptr<EquationOfState> eos, std::shared_ptr<OpacityCalculator> opacity, RadiationIMCParameters parameters)
    : MonteCarloRadiationPhysics3D(grid, boundary, cells, conserved, eos, opacity), withHydro(parameters.withHydro), diffusionPressureGradient(parameters.diffusionPressureGradient), MMC(parameters.MMC), newPhotonsPerCell(parameters.newPhotonsPerCell), withRandomWalk(parameters.withRandomWalk), rwMinCellOpticalDepth(parameters.rwMinCellOpticalDepth), rwMinParticleOpticalDepth(parameters.rwMinParticleOpticalDepth), withDDMC(parameters.withDDMC), ddmcMinCellOpticalDepth(parameters.ddmcMinCellOpticalDepth), ddmcUseMovingInterfaceCorrection(parameters.ddmcUseMovingInterfaceCorrection), ddmcMaxInterfaceVelocityOverC(parameters.ddmcMaxInterfaceVelocityOverC), ddmcInterfaceTargetWeightRatio(parameters.ddmcInterfaceTargetWeightRatio), ddmcMaxInterfaceSplits(parameters.ddmcMaxInterfaceSplits), ddmcUseMultigroupPGRW(parameters.ddmcUseMultigroupPGRW), ddmcMaxGroupCutoff(parameters.ddmcMaxGroupCutoff), ddmcInterfaceDiagnostics(parameters.ddmcInterfaceDiagnostics), noHydroFeedback(parameters.noHydroFeedback), withEgTimeAvg(parameters.withEgTimeAvg), capAbsorptionOpacity(parameters.capAbsorptionOpacity), withCompton(parameters.withCompton), postProcess_(parameters.postProcess), useTransportVelocities_((parameters.withHydro && !parameters.MMC) || (parameters.postProcess.enabled && parameters.postProcess.useCellVelocities)), comptonUseInduced(parameters.comptonUseInduced), comptonInducedMode(parameters.comptonInducedMode), comptonAllowNZeroFallback(parameters.comptonAllowNZeroFallback), comptonAngleDependent(parameters.comptonAngleDependent), comptonDebugParityCheck(parameters.comptonDebugParityCheck), comptonCheckSignedTallies(parameters.comptonCheckSignedTallies), comptonDiagnostics(parameters.comptonDiagnostics), comptonSignedTallyTolerance(parameters.comptonSignedTallyTolerance), comptonMatrixSamples(parameters.comptonMatrixSamples)
{
    if(postProcess_.enabled || postProcess_.polarization.enabled)
    {
        RadiationIMCPostProcessConfig validationConfig;
        validationConfig.enabled = postProcess_.enabled;
        validationConfig.sourceDt = postProcess_.sourceDt;
        validationConfig.transportTime = postProcess_.transportTime;
        validationConfig.forceGreyFleckOne = postProcess_.forceGreyFleckOne;
        validationConfig.useCellVelocities = postProcess_.useCellVelocities;
        validationConfig.polarization.enabled = postProcess_.polarization.enabled;
        validationConfig.polarization.manualScatteringsAfterAcceleration =
            postProcess_.polarization.manualScatteringsAfterAcceleration;
        validationConfig.polarization.depolarizationScatterings =
            postProcess_.polarization.depolarizationScatterings;
        validationConfig.polarization.acceleratedClosure =
            postProcess_.polarization.acceleratedClosure;

        PostProcessIMC::NormalizeAndValidateConfig(
            validationConfig, withCompton, parameters.withMultigroupOpacity,
            withRandomWalk, withDDMC);

        postProcess_.enabled = validationConfig.enabled;
        postProcess_.sourceDt = validationConfig.sourceDt;
        postProcess_.transportTime = validationConfig.transportTime;
        postProcess_.forceGreyFleckOne = validationConfig.forceGreyFleckOne;
        postProcess_.useCellVelocities = validationConfig.useCellVelocities;
        postProcess_.polarization.enabled = validationConfig.polarization.enabled;
        postProcess_.polarization.manualScatteringsAfterAcceleration =
            validationConfig.polarization.manualScatteringsAfterAcceleration;
        postProcess_.polarization.depolarizationScatterings =
            validationConfig.polarization.depolarizationScatterings;
        postProcess_.polarization.acceleratedClosure =
            validationConfig.polarization.acceleratedClosure;
    }
    if(postProcess_.enabled)
    {
        withHydro = false;
        diffusionPressureGradient = false;
        MMC = false;
        noHydroFeedback = true;
    }
    if(this->withDDMC)
    {
#ifndef RICH_IMC_DDMC_ENABLED
        throw UniversalError("RadiationIMC requested DDMC, but this executable was built without RadiationIMC_DDMC.cpp");
#endif
    }
    if(this->withDDMC)
    {
        if(!(this->ddmcMaxInterfaceVelocityOverC > 0.0) ||
           !(this->ddmcMaxInterfaceVelocityOverC < 1.0))
            throw UniversalError("RadiationIMC: DDMC interface velocity limit must lie in (0,1)");
        if(!(this->ddmcInterfaceTargetWeightRatio > 0.0) ||
           !std::isfinite(this->ddmcInterfaceTargetWeightRatio))
            throw UniversalError("RadiationIMC: DDMC interface target-weight ratio must be positive and finite");
        if(this->ddmcMaxInterfaceSplits == 0)
            throw UniversalError("RadiationIMC: DDMC maximum interface split count must be nonzero");
        if(this->ddmcMaxGroupCutoff == 0 ||
           this->ddmcMaxGroupCutoff > ENERGY_GROUPS_NUM)
            throw UniversalError(
                "RadiationIMC: DDMC maximum group cutoff must lie in [1, ENERGY_GROUPS_NUM]");
    }
    if(this->withCompton && this->withRandomWalk)
    {
        throw UniversalError("RadiationIMC Compton precompute is not compatible with random walk yet");
    }
    if(this->withCompton && this->withDDMC)
    {
        throw UniversalError(
            "RadiationIMC: DDMC currently supports absorption, IMC effective "
            "scattering, and elastic physical scattering only. Compton "
            "redistribution with DDMC is intentionally deferred.");
    }
    if(this->withCompton && !parameters.withMultigroupOpacity)
    {
        throw UniversalError("RadiationIMC Compton requires multigroup opacity");
    }
    if(parameters.withMultigroupOpacity)
    {
        this->multigroupOpacity = std::make_shared<MultigroupOpacity>(opacity);
    }
    else
    {
        this->multigroupOpacity = nullptr;
    }
    if(this->withRandomWalk)
    {
        this->randomWalk = std::make_unique<RandomWalk>();
    }
    int rank = 0;
    #ifdef RICH_MPI
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    #endif
    if(rank == 0)
    {
        std::cout << parameters << std::endl;
    }
}

void RadiationIMC::setAdaptiveSourceCellScores(
    std::unordered_map<size_t, double> scores,
    double strength, double maxFactor,
    double learnedReserveFrac, double learnedMinFactor,
    double observerBudgetMultiplier)
{
    adaptiveSourceScores_ = std::move(scores);
    adaptiveSourceStrength_ = std::clamp(strength, 0.0, 1.0);
    adaptiveSourceMaxFactor_ = std::max(1.0, maxFactor);
    adaptiveSourceLearnedReserveFrac_ = std::clamp(learnedReserveFrac, 0.0, 1.0);
    adaptiveSourceLearnedMinFactor_ = std::max(1.0, learnedMinFactor);
    adaptiveSourceObserverBudgetMultiplier_ = std::max(1.0, observerBudgetMultiplier);
    adaptiveSourceScoresEnabled_ = !adaptiveSourceScores_.empty();
}

void RadiationIMC::clearAdaptiveSourceCellScores()
{
    adaptiveSourceScores_.clear();
    adaptiveSourceScoresEnabled_ = false;
    adaptiveSourceStrength_ = 0.0;
    adaptiveSourceMaxFactor_ = 1.0;
    adaptiveSourceLearnedReserveFrac_ = 0.0;
    adaptiveSourceLearnedMinFactor_ = 1.0;
    adaptiveSourceObserverBudgetMultiplier_ = 1.0;
}

void RadiationIMC::setAdaptiveSourceCellGroupScores(
    std::unordered_map<size_t, GroupArray> scores,
    double strength,
    double pdfFloor,
    double maxBias,
    double maxWeightCorrection)
{
    if(!this->multigroupOpacity)
    {
        this->clearAdaptiveSourceCellGroupScores();
        return;
    }
    adaptiveSourceCellGroupScores_ = std::move(scores);
    adaptiveGroupStrength_ = std::clamp(strength, 0.0, 1.0);
    adaptiveGroupPdfFloor_ = std::clamp(pdfFloor, 0.0, 1.0);
    adaptiveGroupMaxBias_ = std::max(1.0, maxBias);
    adaptiveGroupMaxWeightCorrection_ = std::max(1.0, maxWeightCorrection);
    adaptiveSourceCellGroupScoresEnabled_ = !adaptiveSourceCellGroupScores_.empty() && adaptiveGroupStrength_ > 0.0;
}

void RadiationIMC::clearAdaptiveSourceCellGroupScores()
{
    adaptiveSourceCellGroupScores_.clear();
    adaptiveSourceCellGroupScoresEnabled_ = false;
    adaptiveGroupStrength_ = 0.0;
    adaptiveGroupPdfFloor_ = 0.0;
    adaptiveGroupMaxBias_ = 1.0;
    adaptiveGroupMaxWeightCorrection_ = 1.0;
}

void RadiationIMC::setSourceEmissionControl(
    bool useLearnedScores, bool includeUniformBase, size_t baseMultiplier,
    size_t learnedBoostFactor, size_t learnedExtraBudget)
{
    sourceEmissionControlEnabled_ = true;
    sourceEmissionUseLearnedScores_ = useLearnedScores;
    sourceEmissionIncludeUniformBase_ = includeUniformBase;
    sourceEmissionBaseMultiplier_ = std::max<size_t>(1, baseMultiplier);
    sourceEmissionLearnedBoostFactor_ = std::max<size_t>(1, learnedBoostFactor);
    sourceEmissionLearnedExtraBudget_ = learnedExtraBudget;
}

void RadiationIMC::clearSourceEmissionControl()
{
    sourceEmissionControlEnabled_ = false;
    sourceEmissionUseLearnedScores_ = false;
    sourceEmissionIncludeUniformBase_ = true;
    sourceEmissionBaseMultiplier_ = 1;
    sourceEmissionLearnedBoostFactor_ = 20;
    sourceEmissionLearnedExtraBudget_ = 0;
}

typename RadiationIMC::Functionality RadiationIMC::step(Particle &particle, std::vector<Particle> &particlesToAdd)
{
    Functionality functionality;

    // A zero-weight packet carries no energy, but all geometric and opacity
    // event rates remain finite.  If initialWeight is also zero, the old
    // relative cutoff `abs(weight) < initialWeight * eps` is false (0 < 0),
    // allowing an energy-free packet to scatter forever in an optically thick
    // cell.  Remove it before DDMC admission/fallback or IMC event sampling.
    double const absoluteWeight = std::abs(particle.weight);
    double const absoluteInitialWeight = std::abs(particle.initialWeight);
    if(!std::isfinite(absoluteWeight) ||
       !std::isfinite(absoluteInitialWeight))
    {
        UniversalError eo("Non-finite Monte Carlo particle weight");
        eo.addEntry("Particle", particle);
        eo.addEntry("Weight", particle.weight);
        eo.addEntry("Initial weight", particle.initialWeight);
        throw eo;
    }
    if(absoluteWeight == 0.0)
    {
        functionality.change = MonteCarloParticleStatus::REMOVE;
        return functionality;
    }
    if(!(absoluteInitialWeight > 0.0))
        particle.initialWeight = absoluteWeight;

    size_t constexpr noBypassCell = std::numeric_limits<size_t>::max();
    if(particle.ddmcBypassCellID != noBypassCell &&
       particle.ddmcBypassCellID != particle.cellID)
    {
        particle.ddmcBypassCellID = noBypassCell;
    }
    bool const bypassDDMCInCurrentCell =
        particle.ddmcBypassCellID != noBypassCell;

    bool debug = false;

    if(this->withDDMC && particle.ddmcMode)
    {
#ifdef RICH_IMC_DDMC_ENABLED
        if(this->tryDDMCStep(particle, functionality, 1.0))
            return functionality;
#else
        throw UniversalError("RadiationIMC DDMC step requested, but DDMC support was not compiled");
#endif
    }

    size_t cellIndex = particle.cellIndex;
    ComputationalCell3D &cell = this->cells[cellIndex];

    auto [faceIntersect, timeIntersect, nextCellIndex] = this->getIntersectionDetails(particle);
    assert(timeIntersect >= 0);

    // todo: change opacity with doppler shift in cast of frequency dependance
    bool const useVelocityTransport = this->useTransportVelocities_ && !this->MMC;
    double dopplerShift = useVelocityTransport ? DopplerShift(particle, cell.velocity) : 1.0;

    if(this->withDDMC && !particle.ddmcMode &&
       !bypassDDMCInCurrentCell)
    {
#ifdef RICH_IMC_DDMC_ENABLED
        if(this->tryDDMCStep(particle, functionality, dopplerShift))
            return functionality;
#else
        throw UniversalError("RadiationIMC DDMC step requested, but DDMC support was not compiled");
#endif
    }

    if(this->randomWalk && this->rwCellEligible[cellIndex])
    {
        if(this->tryRandomWalkStep(particle, functionality, dopplerShift))
        {
            ++this->rwStepCount;
            return functionality;
        }
    }

    double absorptionOpacity;
    size_t group = std::numeric_limits<size_t>::max();
    if(this->multigroupOpacity)
    {
        double shiftedFrequency = particle.frequency * dopplerShift;
        ClampFrequencyToBounds(shiftedFrequency);
        group = this->opacity->findGroup(shiftedFrequency);
        if(this->withCompton)
        {
            absorptionOpacity = this->comptonData[cellIndex].absorptionOpacity[group];
        }
        else
        {
            absorptionOpacity = this->opacity->CalcAbsorptionOpacity(cell, shiftedFrequency);
        }
    }
    else
    {
        absorptionOpacity = this->planckOpacities[cellIndex];
    }
    bool const comptonMcGroup = this->withCompton && group < ENERGY_GROUPS_NUM;
    double elasticScatteringOpacity = this->withCompton ? 0.0 : this->opacity->CalcScatteringOpacity(cell);
    double effectiveAbsorptionOpacity;
    if(comptonMcGroup)
        effectiveAbsorptionOpacity = (1.0 - this->comptonData[cellIndex].fleck) * absorptionOpacity;
    else if(this->withCompton)
        effectiveAbsorptionOpacity = this->comptonData[cellIndex].baseEffectiveOpacity[group];
    else
        effectiveAbsorptionOpacity = (1 - this->factorFleck[cellIndex]) * absorptionOpacity;
    double implicitComptonOpacity =
        comptonMcGroup ? this->comptonData[cellIndex].comptonOutRate[group] : 0.0;
    double eventOpacity = elasticScatteringOpacity + effectiveAbsorptionOpacity + implicitComptonOpacity;
    double scatteringLength = (eventOpacity > 0.0) ? 1.0 / eventOpacity : std::numeric_limits<double>::infinity();
    double _log1p = -std::log1p(this->dist(this->re) - 1); 
    distance_t scatteringDistance = scatteringLength * _log1p / dopplerShift; 
    if(scatteringDistance < 0)
    {
        UniversalError eo("Negative scattering distance in RadiationIMC::step");
        eo.addEntry("Cell scattering distance", opacity->CalcScatteringOpacity(cell));
        eo.addEntry("Factor fleck", this->factorFleck[cellIndex]);
        eo.addEntry("Planck opacity", this->planckOpacities[cellIndex]);
        eo.addEntry("log(1-randm)", _log1p);
        eo.addEntry("doppler shift", dopplerShift);
        eo.addEntry("particle", particle);
        throw eo;
    }
    dt_t timeScattering = std::isfinite(scatteringDistance) ? scatteringDistance / abs(particle.velocity) : std::numeric_limits<dt_t>::infinity();

    dt_t timeLeft = particle.timeLeft;

    dt_t timeObserver = std::numeric_limits<dt_t>::infinity();
    SphericalObserver::Crossing observerCrossing;
    if (postProcess_.enabled && observer_)
    {
        observerCrossing = observer_->nextOutwardCrossing(
            particle.location, particle.velocity, particle.timeLeft);
        if (observerCrossing.hit)
            timeObserver = observerCrossing.time;
    }

    std::array<std::pair<size_t, dt_t>, 4> times;
    enum Events
    { 
        INTERSECTION = 0, 
        SCATTERING = 1, 
        TIMELEFT = 2,
        OBSERVER = 3
    };
    times[Events::INTERSECTION] = {INTERSECTION, timeIntersect};
    times[Events::SCATTERING] = {SCATTERING, timeScattering};
    times[Events::TIMELEFT] = {TIMELEFT, timeLeft};
    times[Events::OBSERVER] = {OBSERVER, timeObserver};

    std::pair<size_t, double> min = *std::min_element(times.begin(), times.end(), [](const std::pair<size_t, dt_t> &a, const std::pair<size_t, dt_t> &b) { return a.second < b.second; });
    if(debug)
    {
        std::cout << "min: " << min.first << " with time " << min.second << " for particle " << particle << std::endl;
    }
    dt_t dt = min.second;
    if(dt < 0)
    {
        UniversalError eo("Negative time step in RadiationIMC::step");
        eo.addEntry("time Intersect", timeIntersect);
        eo.addEntry("time Scattering", timeScattering);
        eo.addEntry("time Left", timeLeft);
        eo.addEntry("Particle", particle);
        throw eo;
    }
    particle.timeLeft -= dt;
    double const fleckForDecay = (this->withCompton && group < ENERGY_GROUPS_NUM)
        ? this->comptonData[cellIndex].fleck
        : this->factorFleck[cellIndex];
    double weightEvolutionOpacity = absorptionOpacity * fleckForDecay;
    double tmp2 = weightEvolutionOpacity * units::clight;
    double tmp = -dt * tmp2;
    double expFactor1 = std::expm1(tmp * dopplerShift);
    double expFactor2 = std::expm1(tmp);
    double integratedForTally = particle.weight * dt;
    if(std::abs(tmp2) >= 1e-30 && std::abs(tmp2 * dt) >= 1e-12)
    {
        integratedForTally = particle.weight * expFactor2 * (-1.0 / tmp2);
    }
    particle.location += particle.velocity * dt;
    if(!this->noHydroFeedback)
    {
        double const materialDeposit = -expFactor2 * particle.weight;
        this->conserved[cellIndex].internal_energy += materialDeposit;
        if(this->withCompton)
        {
            this->conserved[cellIndex].energy += materialDeposit;
            this->comptonContinuousMaterialExchange += materialDeposit;
        }
        if(this->withHydro)
        {
            if(not this->diffusionPressureGradient)
            {
                this->conserved[cellIndex].momentum += -expFactor1 * particle.weight * particle.velocity * units::inv_clight2;
            }
        }
    }
    this->Erad_time_avg[cellIndex] += integratedForTally;
    if((this->withEgTimeAvg || this->withCompton) && this->multigroupOpacity)
    {
        size_t g = this->opacity->findGroup(particle.frequency);
        this->Eg_time_avg[cellIndex][g] += integratedForTally;
    }
    double const oldWeightForDiagnostics = particle.weight;
    particle.weight *= 1 + expFactor1;

    if (postProcess_.enabled && observer_)
    {
        double absorbed = oldWeightForDiagnostics - particle.weight;
        if (absorbed > 0.0)
            observer_->addAbsorbedEnergy(absorbed);
    }

    double low_energy_threshold = postProcess_.enabled ? 1e-8 : 1e-3;
    double const referenceWeight = std::abs(particle.initialWeight);
    bool const lowWeight = std::abs(particle.weight) == 0.0 ||
        (referenceWeight > 0.0 &&
         std::abs(particle.weight) <= referenceWeight * low_energy_threshold);

    if (postProcess_.enabled && observer_ && min.first == Events::OBSERVER)
    {
        ObserverCrossingRecord rec;
        rec.crossingPoint = particle.location;
        rec.direction = particle.velocity;
        rec.weight = particle.weight;
        rec.frequency = particle.frequency;
        rec.sourceCellID = particle.sourceCellID;
#ifdef MONTECARLO_POLARIZATION
        if(postProcess_.polarization.enabled)
        {
            IMCPolarization::InitializeIfNeeded(particle);
            rec.stokesQ = particle.stokesQ;
            rec.stokesU = particle.stokesU;
            rec.polBasis = particle.polarizationBasis;
            rec.polarizationInitialized = particle.polarizationInitialized;
        }
#endif
        observer_->recordCrossing(rec);
        Vector3D candidate = PostProcessIMC::ObserverNudgeCandidate(*observer_, particle.location);
        if (this->grid.IsPointInCell(candidate, cellIndex))
            particle.location = candidate;
        if (lowWeight)
        {
            observer_->addCutoffEnergy(particle.weight);
            functionality.change = MonteCarloParticleStatus::REMOVE;
        }
        else
        {
            functionality.change = MonteCarloParticleStatus::NO_CELL_MOVE;
        }
        return functionality;
    }

    if(lowWeight)
    {
        if (postProcess_.enabled && observer_)
            observer_->addCutoffEnergy(particle.weight);
        functionality.change = MonteCarloParticleStatus::REMOVE;
        if(!this->noHydroFeedback)
        {
            this->conserved[cellIndex].internal_energy += particle.weight;
            if(this->withCompton)
            {
                this->conserved[cellIndex].energy += particle.weight;
                this->comptonRemovalMaterialExchange += particle.weight;
            }
        }
        return functionality;
    }

    if(min.first == Events::INTERSECTION)
    {
#ifdef RICH_IMC_DDMC_ENABLED
        if(this->withDDMC && !particle.ddmcMode &&
           this->tryIMCToDDMCInterface(particle, functionality,
                                       particlesToAdd, cellIndex,
                                       nextCellIndex, faceIntersect))
        {
            return functionality;
        }
#endif
        // Keep a DDMC-bypass marker through a physical-boundary reflection:
        // the manager may return the packet to this same cell.  Clear it only
        // when this intersection really enters another mesh cell.
        if(!this->grid.IsPointOutsideBox(nextCellIndex))
        {
            particle.ddmcBypassCellID = noBypassCell;
            if(nextCellIndex < this->cells.size())
                particle.cellID = this->cells[nextCellIndex].ID;
        }
        functionality.change = MonteCarloParticleStatus::CELL_MOVE;
        functionality.nextCellIndex = nextCellIndex;
    }
    else if(min.first == Events::SCATTERING)
    {
        Vector3D oldVelocity = particle.velocity;
#ifdef MONTECARLO_POLARIZATION
        Particle polarizationMaterialParticle = particle;
        Vector3D oldVelocityForPolarization = oldVelocity;
        if(postProcess_.enabled && postProcess_.polarization.enabled && this->useTransportVelocities_)
        {
            LorentzTransformation(polarizationMaterialParticle, cell.velocity);
            oldVelocityForPolarization = polarizationMaterialParticle.velocity;
            if(polarizationMaterialParticle.polarizationInitialized)
                polarizationMaterialParticle.polarizationBasis =
                    IMCPolarization::ProjectBasisToDirection(
                        polarizationMaterialParticle.polarizationBasis,
                        polarizationMaterialParticle.velocity);
        }
#endif
        double oldWeight = particle.weight;
        double D_lab_to_co = dopplerShift;
        double eventRandom = this->dist(this->re) * eventOpacity;
        bool didImplicitCompton = false;
        bool isEffectiveScatter = false;
        if(eventRandom < elasticScatteringOpacity)
        {
#ifdef MONTECARLO_POLARIZATION
            if(postProcess_.enabled && postProcess_.polarization.enabled)
            {
                polarizationMaterialParticle.velocity = oldVelocityForPolarization;
                auto u01 = [this]() -> double {
                    return std::clamp(this->dist(this->re),
                                      std::numeric_limits<double>::min(),
                                      1.0 - std::numeric_limits<double>::epsilon());
                };
                particle.velocity =
                    IMCPolarization::SamplePolarizedThomsonDirection(
                        polarizationMaterialParticle,
                        oldVelocityForPolarization,
                        u01);
                IMCPolarization::ApplyThomsonScatter(polarizationMaterialParticle,
                                                     oldVelocityForPolarization,
                                                     particle.velocity);
                particle.stokesQ = polarizationMaterialParticle.stokesQ;
                particle.stokesU = polarizationMaterialParticle.stokesU;
                particle.polarizationBasis = polarizationMaterialParticle.polarizationBasis;
                particle.polarizationInitialized = polarizationMaterialParticle.polarizationInitialized;
            }
            else
#endif
            {
                particle.velocity = opacity->getNewScatterVelocity(cell, particle);
            }
        }
        else if((eventRandom -= elasticScatteringOpacity) < effectiveAbsorptionOpacity)
        {
            particle.velocity = opacity->getNewScatterVelocity(cell, particle);
            isEffectiveScatter = true;
        }
        else
        {
            this->applyComptonScatterEvent(cellIndex, cell, group, oldVelocity, oldWeight, dopplerShift, particle);
            didImplicitCompton = true;
        }
        if(this->multigroupOpacity)
        {
            if(!didImplicitCompton)
            {
                particle.frequency *= dopplerShift; // lab → comoving
                ClampFrequencyToBounds(particle.frequency);
            }
            if(isEffectiveScatter)
            {
                if(this->withCompton)
                {
                    size_t targetGroup = this->sampleComptonCdf(this->comptonData[cellIndex].baseSourceCdf, this->dist(this->re));
                    particle.frequency = this->frequencyForComptonGroup(targetGroup);
                }
                else
                {
                    double reemitRandom = this->dist(this->re);
                    particle.frequency = this->multigroupOpacity->GetThermalEnergy(cell, reemitRandom);
                }
            }
            if(didImplicitCompton)
            {
                ClampFrequencyToBounds(particle.frequency);
            }
        }
        if(useVelocityTransport && !didImplicitCompton)
        {
            double weightBefore = particle.weight;
            particle.weight *= D_lab_to_co;
            ComovingToLabPacket(particle, cell.velocity);
            if(this->multigroupOpacity)
            {
                ClampFrequencyToBounds(particle.frequency);
            }
            if(this->withHydro && !this->diffusionPressureGradient && !this->noHydroFeedback)
            {
                this->conserved[cellIndex].momentum += (weightBefore * oldVelocity - particle.weight * particle.velocity) * units::inv_clight2;
            }
        }
    }
    else if(min.first == Events::TIMELEFT)
    {
        functionality.change = MonteCarloParticleStatus::DONE;
    }
    else
    {
        UniversalError eo("Unknown case in RadiationIMC::step");
        eo.addEntry("Particle", particle);
        throw eo;
    }

    return functionality;
}

void RadiationIMC::postStep(const std::vector<Particle> &particles, double fullDt)
{
    auto printAccelerationStats = [this]()
    {
        unsigned long long counts[6] = {
            static_cast<unsigned long long>(this->rwStepCount),
            static_cast<unsigned long long>(this->ddmcStepCount),
            static_cast<unsigned long long>(this->ddmcLeakCount),
            static_cast<unsigned long long>(this->ddmcCensusCount),
            static_cast<unsigned long long>(this->ddmcUpscatterCount),
            static_cast<unsigned long long>(this->ddmcFallbackCount)
        };
        int rank = 0;
        #ifdef RICH_MPI
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        if(rank == 0)
            MPI_Reduce(MPI_IN_PLACE, counts, 6, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
        else
            MPI_Reduce(counts, nullptr, 6, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
        #endif
        if(rank == 0)
        {
            if(this->withRandomWalk)
                std::cout << "RW steps: " << counts[0] << std::endl;
            if(this->withDDMC)
            {
                std::cout << "DDMC steps: " << counts[1]
                          << " leaks=" << counts[2]
                          << " census=" << counts[3]
                          << " upscatter=" << counts[4]
                          << " fallback=" << counts[5]
                          << std::endl;
            }
        }
    };

    if (postProcess_.enabled)
    {
        double timedOut = 0.0;
        for (auto const& p : particles)
            timedOut += p.weight;
        if (observer_)
            observer_->addTimedOutEnergy(timedOut);
        printAccelerationStats();
        return;
    }

    size_t Ncells = this->grid.GetPointNo();
    for(size_t i = 0; i < Ncells; i++)
    {
        this->Erad_time_avg[i] /= (fullDt * this->grid.GetVolume(i));
        if((this->withEgTimeAvg || this->withCompton) && this->multigroupOpacity)
        {
            double norm = fullDt * this->grid.GetVolume(i);
            for(size_t g = 0; g < this->Eg_time_avg[i].size(); g++)
                this->Eg_time_avg[i][g] /= norm;
        }
    }

    if(!this->noHydroFeedback)
    {
        std::vector<double> Erad_time_avg_grad = std::vector<double>(Ncells, 0);
        if(this->diffusionPressureGradient)
        {
            #ifdef RICH_MPI
            MPI_exchange_data(this->grid, this->Erad_time_avg, true);
            #endif // RICH_MPI

            // todo: fix for 3D, current is 1D!
            
            for(size_t i = 0; i < Ncells; i++)
            {
                const Vector3D &point = this->grid.GetMeshPoint(i);
                // locate neighbor from right and left
                size_t neighbor_right = std::numeric_limits<size_t>::max();
                size_t neighbor_left = std::numeric_limits<size_t>::max();
                for(size_t faceIdx : this->grid.GetCellFaces(i))
                {
                    const std::pair<size_t, size_t> &neighbors = this->grid.GetFaceNeighbors(faceIdx);
                    size_t neighborIdx = (neighbors.first == i)? neighbors.second : neighbors.first;
                    Vector3D diff = normalize(this->grid.GetMeshPoint(neighborIdx) - point);
                    if(diff.x > 0.99)
                    {
                        neighbor_right = neighborIdx;
                    }
                    else if(diff.x < -0.99)
                    {
                        neighbor_left = neighborIdx;
                    }
                }
                if(neighbor_right == std::numeric_limits<size_t>::max())
                {
                    throw UniversalError("No right neighbor found in RadiationIMC::postStep");
                }
                if(neighbor_left == std::numeric_limits<size_t>::max())
                {
                    throw UniversalError("No left neighbor found in RadiationIMC::postStep");
                }
                const Vector3D &neighbor_right_point = this->grid.GetMeshPoint(neighbor_right);
                const Vector3D &neighbor_left_point = this->grid.GetMeshPoint(neighbor_left);
                double grad;
                if(this->grid.IsPointOutsideBox(neighbor_left))
                {
                    grad = (this->Erad_time_avg[neighbor_right] - this->Erad_time_avg[i]) / (neighbor_right_point - point).x;
                }
                else if(this->grid.IsPointOutsideBox(neighbor_right))
                {
                    grad = (this->Erad_time_avg[i] - this->Erad_time_avg[neighbor_left]) / (point - neighbor_left_point).x;
                }
                else
                {
                    grad = (this->Erad_time_avg[neighbor_right] - this->Erad_time_avg[neighbor_left]) / (neighbor_right_point - neighbor_left_point).x;
                }
                Erad_time_avg_grad[i] = grad;
            }
        }
        
        for(size_t i = 0; i < Ncells; i++)
        {
            ComputationalCell3D &cell = this->cells[i];
            cell.internal_energy = this->conserved[i].internal_energy / this->conserved[i].mass;
            if(cell.internal_energy < 0)
            {
                if(!this->withCompton)
                {
                    UniversalError eo("Negative internal energy in RadiationIMC::postStep");
                    eo.addEntry("Cell index", i);
                    eo.addEntry("Internal energy", cell.internal_energy);
                    eo.addEntry("Mass", this->conserved[i].mass);
                    eo.addEntry("Density", cell.density);
                    eo.addEntry("Temperature", cell.temperature);
                    throw eo;
                }
            }
            if(this->withHydro)
            {
                if(this->diffusionPressureGradient)
                {
                    this->conserved[i].momentum.x -= fullDt * this->grid.GetVolume(i) * Erad_time_avg_grad[i] / 3;
                }
                cell.velocity = this->conserved[i].momentum / this->conserved[i].mass;
                this->conserved[i].energy = this->conserved[i].internal_energy + 0.5 * ScalarProd(this->conserved[i].momentum, this->conserved[i].momentum) / this->conserved[i].mass; // TODO: material strength
            }
            if(cell.internal_energy >= 0.0)
            {
                cell.temperature = this->eos->de2T(cell.density, cell.internal_energy, cell.tracers, cell.tracerNames);
                cell.pressure = this->eos->de2p(cell.density, cell.internal_energy, cell.tracers, cell.tracerNames);
            }
        }
    }

    for(size_t i = 0; i < Ncells; i++)
    {
        this->conserved[i].Erad = 0;
        if(this->multigroupOpacity)
        {
            std::fill(this->conserved[i].Eg.begin(), this->conserved[i].Eg.end(), 0.0);
        }
    }
    if(this->withCompton && this->multigroupOpacity)
    {
        this->lastComptonPacketCounts_.assign(Ncells, std::array<size_t, ENERGY_GROUPS_NUM>{});
        this->lastComptonMaxPacketWeight_.assign(Ncells, GroupArray{});
    }
    for(const Particle &particle : particles)
    {
        size_t cellIndex = particle.cellIndex;
        assert(cellIndex < Ncells);
        this->conserved[cellIndex].Erad += particle.weight;
        if(this->multigroupOpacity)
        {
            size_t g = this->opacity->findGroup(particle.frequency);
            this->conserved[cellIndex].Eg[g] += particle.weight;
            if(this->withCompton)
            {
                ++this->lastComptonPacketCounts_[cellIndex][g];
                this->lastComptonMaxPacketWeight_[cellIndex][g] =
                    std::max(this->lastComptonMaxPacketWeight_[cellIndex][g], std::abs(particle.weight));
            }
        }
    }
    if(this->withCompton && this->multigroupOpacity)
    {
        this->applyComptonEndOfStepCorrection(fullDt);
        this->reconcileComptonParticles(const_cast<std::vector<Particle>&>(particles));
        if(!this->noHydroFeedback)
        {
        for(size_t i = 0; i < Ncells; i++)
        {
                ComputationalCell3D &cell = this->cells[i];
                cell.internal_energy = this->conserved[i].internal_energy / this->conserved[i].mass;
                if(cell.internal_energy < 0.0)
            {
                    UniversalError eo("Negative internal energy after Compton end-of-step correction in RadiationIMC::postStep");
                        eo.addEntry("Cell index", i);
                    eo.addEntry("Internal energy", cell.internal_energy);
                    eo.addEntry("Mass", this->conserved[i].mass);
                    eo.addEntry("Density", cell.density);
                    cell.temperature = this->eos->de2T(cell.density, cell.internal_energy, cell.tracers, cell.tracerNames);
                    eo.addEntry("Temperature", cell.temperature);
                    throw eo;
                }
                cell.temperature = this->eos->de2T(cell.density, cell.internal_energy, cell.tracers, cell.tracerNames);
                cell.pressure = this->eos->de2p(cell.density, cell.internal_energy, cell.tracers, cell.tracerNames);
            }
        }
    }
    if(this->withDDMC && !this->diffusionPressureGradient)
    {
#ifdef RICH_IMC_DDMC_ENABLED
        this->reduceDDMCFaceFluxTallies();
        this->applyDDMCMomentumFeedback(fullDt);
#else
        throw UniversalError("RadiationIMC DDMC momentum feedback requested, but DDMC support was not compiled");
#endif
    }
    for(size_t i = 0; i < Ncells; i++)
    {
        ComputationalCell3D &cell = this->cells[i];
        cell.Erad = this->conserved[i].Erad / this->conserved[i].mass;
        if(this->multigroupOpacity)
        {
            for(size_t g = 0; g < cell.Eg.size(); g++)
            {
                cell.Eg[g] = this->conserved[i].Eg[g] / this->conserved[i].mass;
            }
        }
    }

    if(this->withRandomWalk || this->withDDMC)
    {
        int rank = 0;
        #ifdef RICH_MPI
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        #endif
        if(this->withRandomWalk)
        {
            size_t globalRwSteps = this->rwStepCount;
            #ifdef RICH_MPI
            if(rank == 0)
                MPI_Reduce(MPI_IN_PLACE, &globalRwSteps, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
            else
                MPI_Reduce(&globalRwSteps, nullptr, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
            #endif
            if(rank == 0)
                std::cout << "RW steps: " << globalRwSteps << std::endl;
        }
        if(this->withDDMC)
        {
            unsigned long long counts[6] = {
                static_cast<unsigned long long>(this->rwStepCount),
                static_cast<unsigned long long>(this->ddmcStepCount),
                static_cast<unsigned long long>(this->ddmcLeakCount),
                static_cast<unsigned long long>(this->ddmcCensusCount),
                static_cast<unsigned long long>(this->ddmcUpscatterCount),
                static_cast<unsigned long long>(this->ddmcFallbackCount)
            };
            #ifdef RICH_MPI
            if(rank == 0)
                MPI_Reduce(MPI_IN_PLACE, counts, 6, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
            else
                MPI_Reduce(counts, nullptr, 6, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
            #endif
            if(rank == 0)
            {
                std::cout << "DDMC steps: " << counts[1]
                          << " leaks=" << counts[2]
                          << " census=" << counts[3]
                          << " upscatter=" << counts[4]
                          << " fallback=" << counts[5]
                          << std::endl;
            }
        }
    }
    this->printComptonDiagnostics();
}

void RadiationIMC::applyComptonEndOfStepCorrection(double fullDt)
{
    size_t const Ncells = this->grid.GetPointNo();
    if(this->comptonData.size() != Ncells)
        throw UniversalError("Compton data not initialized in applyComptonEndOfStepCorrection");

    // Bounded-NNLS counters (retained for diagnostics even though the primary
    // fallback is now adaptive subcycling).
    size_t boundedCellCount = 0;
    size_t directFailedCount = 0;
    size_t directInadmissibleCount = 0;
    size_t directNegativeFallbackCount = 0;
    size_t materialCapActiveCount = 0;
    size_t residualWarningCount = 0;
    size_t directSupportAwareClampCount = 0;
    size_t directSupportAwareClampGroupCount = 0;
    double maxDirectClampMassFraction = 0.0;
    size_t historyEndpointMismatchCount = 0;
#ifdef RICH_MPI
    bool haveWorstBounded = false;
#endif
    size_t worstCellIndex = 0;
    double worstScore = 0.0;
    BoundedSolverDiagnostics worstBdiag;

    size_t subcycleCellCount = 0;
    size_t subcyclePartialCount = 0;
    size_t subcycleReturnedRawCount = 0;
    size_t subcycleRejectedCount = 0;
    size_t subcycleMaxCount = 0;
    size_t subcycleDirectFailedCount = 0;
    size_t subcycleDirectNegativeCount = 0;
    size_t subcycleDirectCapCount = 0;
    size_t subcycleDirectOtherCount = 0;
    double minSubcycleFraction = std::numeric_limits<double>::infinity();
    double maxSubcycleNegativeTrialMass = 0.0;
    double minSubcycleConsumedFraction = std::numeric_limits<double>::infinity();
    double maxSubcycleConsumedFraction = 0.0;

    for(size_t i = 0; i < Ncells; i++)
    {
        ComptonCellData const &cd = this->comptonData[i];
        GroupArray rawGroupEnergy{};
        GroupArray timeAvgGroupEnergy{};
        GroupArray solveInputGroupEnergy{};
        GroupArray supportFloorEnergy{};
        GroupArray rhs{};
        GroupArray solvedGroupEnergy{};
        GroupMatrix residualMatrix{};
        double totalCorrectionToRadiation = 0.0;

        // Scale Bcorr by ratio of post-transport to pre-step Erad to prevent
        // overcorrection when transport has depleted radiation from this cell.
        double totalPreStepErad = 0.0;
        double totalPostTransportErad = 0.0;
        double totalTimeAvgErad = 0.0;
        for(size_t g = 0; g < ENERGY_GROUPS_NUM; g++)
        {
            rawGroupEnergy[g] = this->conserved[i].Eg[g];
            if(i < this->Eg_time_avg.size())
                timeAvgGroupEnergy[g] = std::max(0.0, this->Eg_time_avg[i][g] * cd.volume);
            totalPostTransportErad += rawGroupEnergy[g];
            totalTimeAvgErad += timeAvgGroupEnergy[g];
            totalPreStepErad += cd.oldRadiationEnergy[g];
        }
        totalPostTransportErad = std::max(0.0, totalPostTransportErad);
        solveInputGroupEnergy = rawGroupEnergy;
        if(totalPostTransportErad > 0.0 && totalTimeAvgErad > 0.0)
        {
            size_t constexpr minPacketsForRawSupport = 2;
            double constexpr timeAvgSupportThreshold = 1e-8;
            double constexpr timeAvgFloorFraction = 0.25;
            double solveInputTotal = 0.0;
            for(size_t g = 0; g < ENERGY_GROUPS_NUM; g++)
            {
                size_t packetCount = minPacketsForRawSupport;
                if(i < this->lastComptonPacketCounts_.size())
                    packetCount = this->lastComptonPacketCounts_[i][g];
                if(packetCount < minPacketsForRawSupport &&
                   timeAvgGroupEnergy[g] > timeAvgSupportThreshold * totalTimeAvgErad)
                {
                    supportFloorEnergy[g] = timeAvgFloorFraction * timeAvgGroupEnergy[g];
                    solveInputGroupEnergy[g] =
                        std::max(solveInputGroupEnergy[g], supportFloorEnergy[g]);
                }
                solveInputTotal += solveInputGroupEnergy[g];
            }
            if(solveInputTotal > 0.0)
            {
                double const renormalization = totalPostTransportErad / solveInputTotal;
                for(size_t g = 0; g < ENERGY_GROUPS_NUM; g++)
                {
                    solveInputGroupEnergy[g] *= renormalization;
                    supportFloorEnergy[g] *= renormalization;
                }
            }
        }
        double const preStepExtensive = totalPreStepErad * cd.volume;
        double const bcorrScale = (preStepExtensive > 0.0)
            ? std::clamp(totalPostTransportErad / preStepExtensive, 0.0, 1.0)
            : 1.0;
        for(size_t g = 0; g < ENERGY_GROUPS_NUM; g++)
            rhs[g] = solveInputGroupEnergy[g] + bcorrScale * cd.Bcorr[g];

        for(size_t g = 0; g < ENERGY_GROUPS_NUM; g++)
        {
            for(size_t h = 0; h < ENERGY_GROUPS_NUM; h++)
            {
                residualMatrix[g][h] = ((g == h) ? 1.0 : 0.0)
                    - fullDt * units::clight * cd.residualKernel[h][g];
            }
        }

        for(size_t g = 0; g < ENERGY_GROUPS_NUM; g++)
        {
            if(!std::isfinite(rhs[g]))
            {
                UniversalError eo("Nonfinite RHS in Compton end-of-step correction");
                eo.addEntry("Cell index", static_cast<double>(i));
                eo.addEntry("Group", static_cast<double>(g));
                eo.addEntry("RHS value", rhs[g]);
                throw eo;
            }
            for(size_t h = 0; h < ENERGY_GROUPS_NUM; h++)
            {
                if(!std::isfinite(residualMatrix[g][h]))
                {
                    UniversalError eo("Nonfinite matrix in Compton end-of-step correction");
                    eo.addEntry("Cell index", static_cast<double>(i));
                    eo.addEntry("Row", static_cast<double>(g));
                    eo.addEntry("Col", static_cast<double>(h));
                    eo.addEntry("Value", residualMatrix[g][h]);
                    throw eo;
                }
            }
        }

        SolverDiagnostics diag;
        GroupArray directSolution{};
        bool const directOk = SolveComptonGroupSystem(residualMatrix, rhs, directSolution, diag);

        double const cellEnergyScale = std::max({
            1.0,
            totalPostTransportErad,
            SumGroups(solveInputGroupEnergy),
            Norm2(rhs),
            MaxAbsGroups(rhs),
            MaxAbsGroups(rawGroupEnergy)
        });
        constexpr double tailSupportFrac = 1e-8;
        constexpr double weakSupportFrac = 1e-6;
        constexpr double moderateSupportFrac = 1e-4;
        constexpr double tailClampFrac = 1e-3;
        constexpr double weakClampFrac = 1e-4;
        constexpr double moderateClampFrac = 1e-4;
        constexpr double strongClampFrac = 1e-4;
        constexpr double totalDirectNegativeClampFrac = 1e-3;
        constexpr double supportAbsFloor = 1.0;

        double const materialCap = this->noHydroFeedback
            ? std::numeric_limits<double>::infinity()
            : SumGroups(rawGroupEnergy) + this->conserved[i].internal_energy;

        double const materialCapTol = std::isfinite(materialCap)
            ? 1e-12 * std::max(1.0, std::abs(materialCap))
            : std::numeric_limits<double>::infinity();

        GroupArray directClamped{};
        double directNegativeMass = 0.0;
        double directMin = 0.0;
        if(directOk)
        {
            directClamped = directSolution;
            directMin = MinGroups(directSolution);
            for(size_t g = 0; g < ENERGY_GROUPS_NUM; g++)
            {
                if(directClamped[g] < 0.0)
                {
                    directNegativeMass += -directClamped[g];
                    directClamped[g] = 0.0;
                }
            }
        }

        double tailNegMass = 0.0, weakNegMass = 0.0;
        double moderateNegMass = 0.0, strongNegMass = 0.0;
        double totalNegMass = 0.0;
        double tailWorstNeg = 0.0, weakWorstNeg = 0.0;
        double moderateWorstNeg = 0.0, strongWorstNeg = 0.0;
        size_t tailWorstGroup = ENERGY_GROUPS_NUM;
        size_t weakWorstGroup = ENERGY_GROUPS_NUM;
        size_t moderateWorstGroup = ENERGY_GROUPS_NUM;
        size_t strongWorstGroup = ENERGY_GROUPS_NUM;
        size_t tailNegGroupCount = 0, weakNegGroupCount = 0;
        size_t moderateNegGroupCount = 0, strongNegGroupCount = 0;
        double tailWorstEndpointFrac = 0.0, weakWorstEndpointFrac = 0.0;
        double moderateWorstEndpointFrac = 0.0, strongWorstEndpointFrac = 0.0;
        double tailWorstHistoryFrac = 0.0, weakWorstHistoryFrac = 0.0;
        double moderateWorstHistoryFrac = 0.0, strongWorstHistoryFrac = 0.0;

        if(directOk)
        {
            for(size_t g = 0; g < ENERGY_GROUPS_NUM; ++g)
            {
                if(directSolution[g] >= 0.0)
                    continue;
                double const neg = -directSolution[g];
                totalNegMass += neg;
                double const endpointSupportScale = std::max({
                    rawGroupEnergy[g],
                    solveInputGroupEnergy[g],
                    supportFloorEnergy[g],
                    std::abs(rhs[g]),
                    supportAbsFloor
                });
                double const historySupportScale = std::max({
                    endpointSupportScale,
                    timeAvgGroupEnergy[g]
                });
                double const epFrac = endpointSupportScale / cellEnergyScale;
                double const hiFrac = historySupportScale / cellEnergyScale;

                if(hiFrac > 1e-4 && epFrac < 1e-6)
                    ++historyEndpointMismatchCount;

                auto record = [&](double &mass, double &worstNeg,
                                  size_t &worstGrp, size_t &count,
                                  double &wEpFrac, double &wHiFrac)
                {
                    mass += neg;
                    count++;
                    if(neg > worstNeg)
                    {
                        worstNeg = neg;
                        worstGrp = g;
                        wEpFrac = epFrac;
                        wHiFrac = hiFrac;
                    }
                };

                if(epFrac < tailSupportFrac)
                {
                    record(tailNegMass, tailWorstNeg, tailWorstGroup,
                           tailNegGroupCount, tailWorstEndpointFrac,
                           tailWorstHistoryFrac);
                }
                else if(epFrac < weakSupportFrac)
                {
                    record(weakNegMass, weakWorstNeg, weakWorstGroup,
                           weakNegGroupCount, weakWorstEndpointFrac,
                           weakWorstHistoryFrac);
                }
                else if(epFrac < moderateSupportFrac)
                {
                    record(moderateNegMass, moderateWorstNeg, moderateWorstGroup,
                           moderateNegGroupCount, moderateWorstEndpointFrac,
                           moderateWorstHistoryFrac);
                }
                else
                {
                    record(strongNegMass, strongWorstNeg, strongWorstGroup,
                           strongNegGroupCount, strongWorstEndpointFrac,
                           strongWorstHistoryFrac);
                }
            }
        }

        bool const tailOk =
            tailNegMass <= tailClampFrac * cellEnergyScale &&
            tailWorstNeg <= tailClampFrac * cellEnergyScale;
        bool const weakOk =
            weakNegMass <= weakClampFrac * cellEnergyScale &&
            weakWorstNeg <= weakClampFrac * cellEnergyScale;
        bool const moderateOk =
            moderateNegMass <= moderateClampFrac * cellEnergyScale &&
            moderateWorstNeg <= moderateClampFrac * cellEnergyScale;
        bool const strongOk =
            strongNegMass <= strongClampFrac * cellEnergyScale &&
            strongWorstNeg <= strongClampFrac * cellEnergyScale;
        bool const totalNegativeOk =
            totalNegMass <= totalDirectNegativeClampFrac * cellEnergyScale;

        bool const directClampAcceptable =
            directOk &&
            tailOk && weakOk && moderateOk && strongOk &&
            totalNegativeOk;

        bool const directCapOk = !std::isfinite(materialCap)
            || SumGroups(directClamped) <= materialCap + materialCapTol;

        bool const directAdmissible =
            directClampAcceptable && directCapOk;

        if(directAdmissible)
        {
            solvedGroupEnergy = directClamped;
            if(totalNegMass > 0.0)
            {
                directSupportAwareClampCount++;
                directSupportAwareClampGroupCount +=
                    tailNegGroupCount + weakNegGroupCount +
                    moderateNegGroupCount + strongNegGroupCount;
                maxDirectClampMassFraction = std::max(
                    maxDirectClampMassFraction,
                    totalNegMass / cellEnergyScale);
            }
        }
        else
        {
            // Disabled as the primary fallback for non-admissible direct Compton corrections.
            // The bounded NNLS active-set path was observed to select ill-conditioned active
            // sets and zero large, well-supported positive groups. Keep the helper code for
            // possible future diagnostics, but use adaptive Compton subcycling as the
            // production positivity-preserving fallback.
            //
            // bool const directRejectedForNegativity =
            //     directOk && !directClampAcceptable;
            //
            // BoundedSolverDiagnostics bdiag;
            // bdiag.directSolveFailed = !directOk;
            // bool const boundedOk = SolveBoundedComptonCorrection(
            //     residualMatrix, rhs, rawGroupEnergy,
            //     solveInputGroupEnergy, supportFloorEnergy,
            //     materialCap, directSolution, solvedGroupEnergy, bdiag);
            //
            // constexpr double significantGroupFrac = 1e-6;
            // constexpr double maxBoundedDirectDeviation = 1e-2;
            // double const significantFloor = significantGroupFrac * cellEnergyScale;
            //
            // bdiag.maxDirectDeviationFraction = 0.0;
            // bdiag.maxDirectDeviationGroup = 0;
            // if(directOk && boundedOk)
            // {
            //     for(size_t g = 0; g < ENERGY_GROUPS_NUM; ++g)
            //     {
            //         double const groupScale = std::max({
            //             std::abs(directClamped[g]),
            //             rawGroupEnergy[g],
            //             solveInputGroupEnergy[g],
            //             std::abs(rhs[g]),
            //             significantFloor
            //         });
            //         double const deviation = std::abs(solvedGroupEnergy[g] - directClamped[g]) / groupScale;
            //         if(deviation > bdiag.maxDirectDeviationFraction)
            //         {
            //             bdiag.maxDirectDeviationFraction = deviation;
            //             bdiag.maxDirectDeviationGroup = g;
            //         }
            //     }
            // }
            //
            // if(!boundedOk)
            // {
            //     UniversalError eo("Failed bounded end-of-step Compton correction");
            //     throw eo;
            // }
            //
            // constexpr double boundedWeightedRelThrow = 1e-3;
            // constexpr double boundedUnweightedRelThrow = 1e-3;
            // constexpr double boundedGroupRelThrow = 1e-2;
            // constexpr double boundedResidualWarn = 1e-5;
            // double const stricterWeightedThrow = directOk ? boundedWeightedRelThrow : 1e-4;
            //
            // bool const boundedDirectDeviationOk = !directOk
            //     || directRejectedForNegativity
            //     || bdiag.maxDirectDeviationFraction <= maxBoundedDirectDeviation;
            //
            // bool const residualAcceptable = ...;
            //
            // if(!residualAcceptable)
            // {
            //     UniversalError eo("Bounded Compton correction residual too large");
            //     throw eo;
            // }
            //
            // boundedCellCount++;
            // ...counter updates...

            GroupArray Btotal{};
            for(size_t g = 0; g < ENERGY_GROUPS_NUM; ++g)
                Btotal[g] = bcorrScale * cd.Bcorr[g];

            ComptonSubcycleDiagnostics sdiag;
            bool const subcycleOk = SolveComptonCorrectionByAdaptiveSubcycling(
                rawGroupEnergy,
                solveInputGroupEnergy,
                Btotal,
                cd.residualKernel,
                fullDt,
                materialCap,
                cellEnergyScale,
                solvedGroupEnergy,
                sdiag);

            if(!subcycleOk)
            {
                UniversalError eo("Adaptive Compton subcycling failed");
                eo.addEntry("Cell index", static_cast<double>(i));
                eo.addEntry("Cell", this->cells[i]);
                eo.addEntry("Cell volume", this->grid.GetVolume(i));
                eo.addEntry("Cell energy scale", cellEnergyScale);
                eo.addEntry("Planck opacity", cd.planckOpacity);
                eo.addEntry("Planck opacity cdt", cd.planckOpacity * units::clight * fullDt);
                eo.addEntry("Full dt", fullDt);
                eo.addEntry("Gamma", cd.Gamma);
                eo.addEntry("Fleck", cd.fleck);
                eo.addEntry("Upsilon", cd.Upsilon);
                eo.addEntry("bcorrScale", bcorrScale);
                eo.addEntry("Direct solver ok", static_cast<double>(directOk));
                eo.addEntry("Direct negative mass", directNegativeMass);
                eo.addEntry("Material cap", materialCap);
                eo.addEntry("Total raw Erad", totalPostTransportErad);
                eo.addEntry("Total solve-input Erad", SumGroups(solveInputGroupEnergy));
                eo.addEntry("Total time-avg Erad", totalTimeAvgErad);
                eo.addEntry("Sum raw radiation", SumGroups(rawGroupEnergy));
                eo.addEntry("Sum solved radiation", SumGroups(solvedGroupEnergy));
                eo.addEntry("Radiation delta vs raw", SumGroups(solvedGroupEnergy) - SumGroups(rawGroupEnergy));
                eo.addEntry("Material deposit", -(SumGroups(solvedGroupEnergy) - SumGroups(rawGroupEnergy)));
                eo.addEntry("Material energy before correction", this->conserved[i].internal_energy);
                eo.addEntry("Predicted material energy after correction",
                    this->conserved[i].internal_energy - (SumGroups(solvedGroupEnergy) - SumGroups(rawGroupEnergy)));
                eo.addEntry("Subcycle consumed fraction", sdiag.consumedFraction);
                eo.addEntry("Subcycle count", static_cast<double>(sdiag.subcycles));
                eo.addEntry("Subcycle rejected steps", static_cast<double>(sdiag.rejectedSteps));
                eo.addEntry("noHydroFeedback", static_cast<double>(this->noHydroFeedback));
                if(directOk)
                {
                    eo.addEntry("Direct min solution", directMin);
                    eo.addEntry("Direct negative mass fraction",
                        directNegativeMass / cellEnergyScale);
                    eo.addEntry("Tail negative mass", tailNegMass);
                    eo.addEntry("Weak negative mass", weakNegMass);
                    eo.addEntry("Moderate negative mass", moderateNegMass);
                    eo.addEntry("Strong negative mass", strongNegMass);
                    eo.addEntry("Total direct negative mass", totalNegMass);
                }
                eo.addEntry("Min pivot", diag.minPivot);
                eo.addEntry("Max coefficient", diag.maxCoeff);

                auto addGroupInfo = [&](size_t g, std::string const &prefix)
                {
                    eo.addEntry(prefix + " group", static_cast<double>(g));
                    eo.addEntry(prefix + " Eg_raw", rawGroupEnergy[g]);
                    eo.addEntry(prefix + " solveInput", solveInputGroupEnergy[g]);
                    eo.addEntry(prefix + " supportFloor", supportFloorEnergy[g]);
                    eo.addEntry(prefix + " RHS", rhs[g]);
                    eo.addEntry(prefix + " directSolution", directSolution[g]);
                    eo.addEntry(prefix + " subcycleSolution", solvedGroupEnergy[g]);
                    eo.addEntry(prefix + " directClamped", directClamped[g]);
                    if(i < this->lastComptonPacketCounts_.size())
                        eo.addEntry(prefix + " packetCount",
                            static_cast<double>(this->lastComptonPacketCounts_[i][g]));
                    if(i < this->lastComptonMaxPacketWeight_.size())
                        eo.addEntry(prefix + " maxPacketWeight",
                            this->lastComptonMaxPacketWeight_[i][g]);
                    if(i < this->Eg_time_avg.size())
                        eo.addEntry(prefix + " timeAvgEnergy", timeAvgGroupEnergy[g]);
                    eo.addEntry(prefix + " riskScore", cd.riskScore[g]);
                    eo.addEntry(prefix + " riskTargetPackets",
                        static_cast<double>(cd.riskTargetPackets[g]));
                };

                size_t worstDirectG = 0;
                if(directOk)
                {
                    for(size_t g = 1; g < ENERGY_GROUPS_NUM; g++)
                        if(directSolution[g] < directSolution[worstDirectG])
                            worstDirectG = g;
                }
                else
                {
                    for(size_t g = 1; g < ENERGY_GROUPS_NUM; g++)
                        if(std::abs(rhs[g]) > std::abs(rhs[worstDirectG]))
                            worstDirectG = g;
                }
                addGroupInfo(worstDirectG, "DirectWorst");

                if(tailWorstGroup < ENERGY_GROUPS_NUM)
                {
                    addGroupInfo(tailWorstGroup, "TailNegWorst");
                    eo.addEntry("TailNegWorst endpointSupportFraction", tailWorstEndpointFrac);
                    eo.addEntry("TailNegWorst historySupportFraction", tailWorstHistoryFrac);
                }
                if(weakWorstGroup < ENERGY_GROUPS_NUM)
                {
                    addGroupInfo(weakWorstGroup, "WeakNegWorst");
                    eo.addEntry("WeakNegWorst endpointSupportFraction", weakWorstEndpointFrac);
                    eo.addEntry("WeakNegWorst historySupportFraction", weakWorstHistoryFrac);
                }
                if(moderateWorstGroup < ENERGY_GROUPS_NUM)
                {
                    addGroupInfo(moderateWorstGroup, "ModerateNegWorst");
                    eo.addEntry("ModerateNegWorst endpointSupportFraction", moderateWorstEndpointFrac);
                    eo.addEntry("ModerateNegWorst historySupportFraction", moderateWorstHistoryFrac);
                }
                if(strongWorstGroup < ENERGY_GROUPS_NUM)
                {
                    addGroupInfo(strongWorstGroup, "StrongNegWorst");
                    eo.addEntry("StrongNegWorst endpointSupportFraction", strongWorstEndpointFrac);
                    eo.addEntry("StrongNegWorst historySupportFraction", strongWorstHistoryFrac);
                }
                throw eo;
            }

            subcycleCellCount++;
            subcycleRejectedCount += sdiag.rejectedSteps;
            subcycleMaxCount = std::max(subcycleMaxCount, sdiag.subcycles);

            if(sdiag.usedPartialCorrection)
                subcyclePartialCount++;

            if(sdiag.returnedRawNoCorrection)
                subcycleReturnedRawCount++;

            if(std::isfinite(sdiag.minAcceptedFraction))
                minSubcycleFraction = std::min(minSubcycleFraction, sdiag.minAcceptedFraction);

            maxSubcycleNegativeTrialMass =
                std::max(maxSubcycleNegativeTrialMass, sdiag.maxNegativeTrialMass);

            minSubcycleConsumedFraction =
                std::min(minSubcycleConsumedFraction, sdiag.consumedFraction);
            maxSubcycleConsumedFraction =
                std::max(maxSubcycleConsumedFraction, sdiag.consumedFraction);

            bool const directRejectedForNegativity =
                directOk && !directClampAcceptable;
            bool const directRejectedForCap =
                directOk && directClampAcceptable && !directCapOk;

            if(!directOk)
                subcycleDirectFailedCount++;
            else if(directRejectedForNegativity)
                subcycleDirectNegativeCount++;
            else if(directRejectedForCap)
                subcycleDirectCapCount++;
            else
                subcycleDirectOtherCount++;
        }

        for(size_t g = 0; g < ENERGY_GROUPS_NUM; g++)
        {
            totalCorrectionToRadiation += solvedGroupEnergy[g] - rawGroupEnergy[g];
            this->conserved[i].Eg[g] = solvedGroupEnergy[g];
        }

        this->conserved[i].Erad += totalCorrectionToRadiation;
        if(!this->noHydroFeedback)
        {
            double const materialDeposit = -totalCorrectionToRadiation;
            this->conserved[i].internal_energy += materialDeposit;
            this->conserved[i].energy += materialDeposit;
            this->comptonImplicitMaterialExchange += materialDeposit;
            if(this->conserved[i].internal_energy < 0.0)
            {
                UniversalError eo("Negative internal energy after Compton end-of-step correction");
                eo.addEntry("Cell index", static_cast<double>(i));
                eo.addEntry("Material deposit", materialDeposit);
                eo.addEntry("Internal energy", this->conserved[i].internal_energy);
                eo.addEntry("Total radiation delta", totalCorrectionToRadiation);
                eo.addEntry("Fleck", cd.fleck);
                eo.addEntry("Gamma", cd.Gamma);
                throw eo;
            }
            ComputationalCell3D &cell = this->cells[i];
            cell.internal_energy = this->conserved[i].internal_energy / this->conserved[i].mass;
            cell.temperature = this->eos->de2T(cell.density, cell.internal_energy, cell.tracers, cell.tracerNames);
            cell.pressure = this->eos->de2p(cell.density, cell.internal_energy, cell.tracers, cell.tracerNames);
        }
    }

    if(this->comptonDiagnostics)
    {
#ifdef RICH_MPI
        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        unsigned long long counts[9] = {
            static_cast<unsigned long long>(boundedCellCount),
            static_cast<unsigned long long>(directFailedCount),
            static_cast<unsigned long long>(directInadmissibleCount),
            static_cast<unsigned long long>(materialCapActiveCount),
            static_cast<unsigned long long>(residualWarningCount),
            static_cast<unsigned long long>(directSupportAwareClampCount),
            static_cast<unsigned long long>(directSupportAwareClampGroupCount),
            static_cast<unsigned long long>(historyEndpointMismatchCount),
            static_cast<unsigned long long>(directNegativeFallbackCount)
        };
        if(rank == 0)
            MPI_Reduce(MPI_IN_PLACE, counts, 9, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
        else
            MPI_Reduce(counts, nullptr, 9, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
        double globalMaxClampFrac = maxDirectClampMassFraction;
        MPI_Reduce(rank == 0 ? MPI_IN_PLACE : &globalMaxClampFrac,
                   rank == 0 ? &globalMaxClampFrac : nullptr,
                   1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

        double const localWorstValue = haveWorstBounded
            ? worstScore
            : -std::numeric_limits<double>::max();
        struct { double val; int rank; } localWorst = {localWorstValue, rank};
        struct { double val; int rank; } globalWorst;
        MPI_Allreduce(&localWorst, &globalWorst, 1, MPI_DOUBLE_INT, MPI_MAXLOC, MPI_COMM_WORLD);

        if(rank == 0 && (counts[0] > 0 || counts[5] > 0))
        {
            if(counts[0] > 0)
                std::cout << "[Compton bounded correction] " << counts[0]
                          << " cells (directFailed=" << counts[1]
                          << " directNegative=" << counts[8]
                          << " directInadmissible=" << counts[2]
                          << " capActive=" << counts[3]
                          << " residWarn=" << counts[4]
                          << "), worstScore=" << globalWorst.val
                          << " onRank=" << globalWorst.rank
                          << std::endl;
            if(counts[5] > 0)
                std::cout << "[Compton direct support-aware clamp] " << counts[5]
                          << " cells " << counts[6]
                          << " groups maxMassFrac=" << globalMaxClampFrac
                          << " histEpMismatch=" << counts[7]
                          << std::endl;
        }
        if(haveWorstBounded && rank == globalWorst.rank)
        {
            std::cout << "[Compton bounded correction worst local]"
                      << " rank=" << rank
                      << " cell=" << worstCellIndex
                      << " cellID=" << this->cells[worstCellIndex].ID
                      << " score=" << worstScore
                      << " relResid=" << worstBdiag.relativeResidual
                      << " unwRelResid=" << worstBdiag.unweightedRelativeResidual
                      << " maxGroupFrac=" << worstBdiag.maxGroupResidualFraction
                      << " active=" << worstBdiag.activeVariables
                      << " directFailed=" << worstBdiag.directSolveFailed
                      << " capActive=" << worstBdiag.materialCapActive
                      << " cap=" << worstBdiag.materialCap
                      << " sumEnergy=" << worstBdiag.sumEnergy
                      << " minPivot=" << worstBdiag.minPassivePivot
                      << " maxDirectDevFrac=" << worstBdiag.maxDirectDeviationFraction
                      << " maxDirectDevGroup=" << worstBdiag.maxDirectDeviationGroup
                      << std::endl;
        }

        {
            unsigned long long scCounts[8] = {
                static_cast<unsigned long long>(subcycleCellCount),
                static_cast<unsigned long long>(subcyclePartialCount),
                static_cast<unsigned long long>(subcycleReturnedRawCount),
                static_cast<unsigned long long>(subcycleRejectedCount),
                static_cast<unsigned long long>(subcycleDirectFailedCount),
                static_cast<unsigned long long>(subcycleDirectNegativeCount),
                static_cast<unsigned long long>(subcycleDirectCapCount),
                static_cast<unsigned long long>(subcycleDirectOtherCount)
            };
            if(rank == 0)
                MPI_Reduce(MPI_IN_PLACE, scCounts, 8, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
            else
                MPI_Reduce(scCounts, nullptr, 8, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

            unsigned long long globalMaxSub = static_cast<unsigned long long>(subcycleMaxCount);
            MPI_Reduce(rank == 0 ? MPI_IN_PLACE : &globalMaxSub,
                       rank == 0 ? &globalMaxSub : nullptr,
                       1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

            double scDoubles[4] = {
                minSubcycleFraction,
                maxSubcycleNegativeTrialMass,
                minSubcycleConsumedFraction,
                maxSubcycleConsumedFraction
            };
            for(int d = 0; d < 4; ++d)
            {
                MPI_Reduce(rank == 0 ? MPI_IN_PLACE : &scDoubles[d],
                           rank == 0 ? &scDoubles[d] : nullptr,
                           1, MPI_DOUBLE,
                           d % 2 == 0 ? MPI_MIN : MPI_MAX,
                           0, MPI_COMM_WORLD);
            }

            if(rank == 0 && scCounts[0] > 0)
            {
                double const printableMinFrac = std::isfinite(scDoubles[0])
                    ? scDoubles[0] : 0.0;
                double const printableMinConsumed = std::isfinite(scDoubles[2])
                    ? scDoubles[2] : 0.0;

                std::cout << "[Compton adaptive subcycle]"
                          << " cells=" << scCounts[0]
                          << " directFailed=" << scCounts[4]
                          << " directNegative=" << scCounts[5]
                          << " directCap=" << scCounts[6]
                          << " directOther=" << scCounts[7]
                          << " partial=" << scCounts[1]
                          << " returnedRaw=" << scCounts[2]
                          << " minConsumedFrac=" << printableMinConsumed
                          << " maxConsumedFrac=" << scDoubles[3]
                          << " maxSubcycles=" << globalMaxSub
                          << " rejected=" << scCounts[3]
                          << " minAcceptedFrac=" << printableMinFrac
                          << " maxNegativeTrialMass=" << scDoubles[1]
                          << std::endl;
            }
        }
#else
        if(boundedCellCount > 0)
        {
            std::cout << "[Compton bounded correction] " << boundedCellCount
                      << " cells (directFailed=" << directFailedCount
                      << " directNegative=" << directNegativeFallbackCount
                      << " directInadmissible=" << directInadmissibleCount
                      << " capActive=" << materialCapActiveCount
                      << " residWarn=" << residualWarningCount
                      << "), worst relResid=" << worstBdiag.relativeResidual
                      << " unwRelResid=" << worstBdiag.unweightedRelativeResidual
                      << " maxGroupFrac=" << worstBdiag.maxGroupResidualFraction
                      << " minPivot=" << worstBdiag.minPassivePivot
                      << " maxDirectDevFrac=" << worstBdiag.maxDirectDeviationFraction
                      << " maxDirectDevGroup=" << worstBdiag.maxDirectDeviationGroup
                      << " score=" << worstScore
                      << " cell=" << worstCellIndex << std::endl;
        }
        if(directSupportAwareClampCount > 0)
        {
            std::cout << "[Compton direct support-aware clamp] " << directSupportAwareClampCount
                      << " cells " << directSupportAwareClampGroupCount
                      << " groups maxMassFrac=" << maxDirectClampMassFraction
                      << " histEpMismatch=" << historyEndpointMismatchCount
                      << std::endl;
        }
        if(subcycleCellCount > 0)
        {
            double const printableMinFrac = std::isfinite(minSubcycleFraction)
                ? minSubcycleFraction : 0.0;
            double const printableMinConsumed = std::isfinite(minSubcycleConsumedFraction)
                ? minSubcycleConsumedFraction : 0.0;
            std::cout << "[Compton adaptive subcycle]"
                      << " cells=" << subcycleCellCount
                      << " directFailed=" << subcycleDirectFailedCount
                      << " directNegative=" << subcycleDirectNegativeCount
                      << " directCap=" << subcycleDirectCapCount
                      << " directOther=" << subcycleDirectOtherCount
                      << " partial=" << subcyclePartialCount
                      << " returnedRaw=" << subcycleReturnedRawCount
                      << " minConsumedFrac=" << printableMinConsumed
                      << " maxConsumedFrac=" << maxSubcycleConsumedFraction
                      << " maxSubcycles=" << subcycleMaxCount
                      << " rejected=" << subcycleRejectedCount
                      << " minAcceptedFrac=" << printableMinFrac
                      << " maxNegativeTrialMass=" << maxSubcycleNegativeTrialMass
                      << std::endl;
        }
#endif
    }
}

std::vector<typename RadiationIMC::Particle> RadiationIMC::generateParticles(double fullDt)
{
    if(this->withCompton)
        return this->generateComptonParticles(fullDt);

    lastGroupSamplingDiagnostics_ = GroupSamplingDiagnostics{};
    if (adaptiveSourceCellGroupScoresEnabled_)
        lastGroupSamplingDiagnostics_.cellsWithGroupScores = adaptiveSourceCellGroupScores_.size();

    std::vector<Particle> newParticles;
    size_t Ncells = this->grid.GetPointNo();

    std::vector<double> energyToCreateVec(Ncells);
    std::vector<double> gammaVec(Ncells);
    double localTotalEnergy = 0;
    for(size_t i = 0; i < Ncells; i++)
    {
        ComputationalCell3D &cell = this->cells[i];
        gammaVec[i] = (this->useTransportVelocities_ && !this->MMC) ? 1 / std::sqrt(1 - ScalarProd(cell.velocity, cell.velocity) * units::inv_clight2) : 1;
        energyToCreateVec[i] = this->factorFleck[i] * this->grid.GetVolume(i) * units::arad * boost::math::pow<4>(cell.temperature) * this->planckOpacities[i] * fullDt * units::clight;
        localTotalEnergy += energyToCreateVec[i];
    }

    double globalTotalEnergy = localTotalEnergy;
    size_t globalTotalCells = Ncells;
    #ifdef RICH_MPI
    MPI_Allreduce(MPI_IN_PLACE, &globalTotalEnergy, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &globalTotalCells, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    #endif

    size_t totalParticles = globalTotalCells * this->newPhotonsPerCell * 10;
    std::vector<size_t> nPhotons(Ncells);
    for(size_t i = 0; i < Ncells; i++)
    {
        size_t proportionalShare = (globalTotalEnergy > 0)
            ? static_cast<size_t>(energyToCreateVec[i] / globalTotalEnergy * totalParticles)
            : this->newPhotonsPerCell;
        nPhotons[i] = std::max(this->newPhotonsPerCell, std::min(proportionalShare, this->newPhotonsPerCell * 20));
    }

    if(sourceEmissionControlEnabled_)
    {
        double scoreSum = 0.0;
        for(auto const &kv : adaptiveSourceScores_)
            if(std::isfinite(kv.second) && kv.second > 0.0)
                scoreSum += kv.second;

        size_t const basePhotons = this->newPhotonsPerCell * sourceEmissionBaseMultiplier_;
        size_t const maxPhotons = static_cast<size_t>(std::ceil(
            static_cast<double>(std::max<size_t>(1, this->newPhotonsPerCell)) *
            adaptiveSourceMaxFactor_ * adaptiveSourceObserverBudgetMultiplier_));
        for(size_t i = 0; i < Ncells; ++i)
        {
            auto const it = adaptiveSourceScores_.find(this->cells[i].ID);
            bool const learned = adaptiveSourceScoresEnabled_ && it != adaptiveSourceScores_.end()
                && std::isfinite(it->second) && it->second > 0.0;

            size_t photons = sourceEmissionIncludeUniformBase_ ? basePhotons : 0;
            if(sourceEmissionUseLearnedScores_ && learned)
            {
                size_t learnedPhotons = this->newPhotonsPerCell * sourceEmissionLearnedBoostFactor_;
                if(scoreSum > 0.0 && sourceEmissionLearnedExtraBudget_ > 0)
                {
                    learnedPhotons += static_cast<size_t>(std::ceil(
                        adaptiveSourceStrength_ * static_cast<double>(sourceEmissionLearnedExtraBudget_) *
                        it->second / scoreSum));
                }
                size_t const minLearned = static_cast<size_t>(std::ceil(
                    static_cast<double>(std::max<size_t>(1, this->newPhotonsPerCell)) *
                    adaptiveSourceLearnedMinFactor_));
                learnedPhotons = std::max(learnedPhotons, minLearned);
                photons = std::max(photons, learnedPhotons);
            }
            nPhotons[i] = std::min(photons, std::max<size_t>(1, maxPhotons));
        }
    }

    lastSourceAllocationSummary_ = SourceAllocationSummary{};
    lastSourceAllocationSummary_.adaptiveEnabled =
        sourceEmissionControlEnabled_ && sourceEmissionUseLearnedScores_ && adaptiveSourceScoresEnabled_;
    lastSourceAllocationSummary_.minPhotons = std::numeric_limits<size_t>::max();
    lastSourceAllocationSummary_.learnedMinPhotons = std::numeric_limits<size_t>::max();
    for(size_t i = 0; i < Ncells; ++i)
    {
        size_t const photons = nPhotons[i];
        if(photons == 0)
            continue;
        ++lastSourceAllocationSummary_.sourceCells;
        lastSourceAllocationSummary_.totalPhotons += photons;
        lastSourceAllocationSummary_.minPhotons = std::min(lastSourceAllocationSummary_.minPhotons, photons);
        lastSourceAllocationSummary_.maxPhotons = std::max(lastSourceAllocationSummary_.maxPhotons, photons);
        if(photons > this->newPhotonsPerCell)
            ++lastSourceAllocationSummary_.boostedCells;

        auto const it = adaptiveSourceScores_.find(this->cells[i].ID);
        bool const learned = adaptiveSourceScoresEnabled_ && it != adaptiveSourceScores_.end()
            && std::isfinite(it->second) && it->second > 0.0;
        if(learned)
        {
            ++lastSourceAllocationSummary_.learnedCells;
            lastSourceAllocationSummary_.learnedPhotons += photons;
            lastSourceAllocationSummary_.adaptiveScoreSum += it->second;
            lastSourceAllocationSummary_.learnedMinPhotons =
                std::min(lastSourceAllocationSummary_.learnedMinPhotons, photons);
            lastSourceAllocationSummary_.learnedMaxPhotons =
                std::max(lastSourceAllocationSummary_.learnedMaxPhotons, photons);
            if (nPhotons[i] > this->newPhotonsPerCell)
            {
                ++lastSourceAllocationSummary_.learnedBoostedCells;
                lastSourceAllocationSummary_.learnedExtraPhotons += photons - this->newPhotonsPerCell;
            }
        }
    }
    if(lastSourceAllocationSummary_.minPhotons == std::numeric_limits<size_t>::max())
        lastSourceAllocationSummary_.minPhotons = 0;
    if(lastSourceAllocationSummary_.learnedMinPhotons == std::numeric_limits<size_t>::max())
        lastSourceAllocationSummary_.learnedMinPhotons = 0;
    lastSourcePhotonsPerCell_ = nPhotons;
    unsigned long long const localPhotonCount = lastSourceAllocationSummary_.totalPhotons;
    if (localPhotonCount > static_cast<unsigned long long>(newParticles.max_size()))
    {
        UniversalError eo("Too many local IMC source particles to allocate");
        eo.addEntry("Local photon count", static_cast<double>(localPhotonCount));
        eo.addEntry("Vector max size", static_cast<double>(newParticles.max_size()));
        throw eo;
    }
    newParticles.reserve(static_cast<size_t>(localPhotonCount));

    for(size_t i = 0; i < Ncells; i++)
    {
        ComputationalCell3D &cell = this->cells[i];
        double energyToCreate = energyToCreateVec[i];
        double gamma = gammaVec[i];
        size_t nPhotonsCell = nPhotons[i];
        if (nPhotonsCell == 0)
            continue;

        if(!this->noHydroFeedback)
        {
            this->conserved[i].internal_energy -= energyToCreate;
            this->conserved[i].energy -= energyToCreate * gamma;
            if(this->withHydro)
            {
                if(not this->diffusionPressureGradient)
                {
                    this->conserved[i].momentum -= energyToCreate * cell.velocity * units::inv_clight2 * gamma;
                }
            }
        }
        double energyPerPhoton = energyToCreate * gamma / nPhotonsCell;

        bool useGroupFreqSampling = adaptiveSourceCellGroupScoresEnabled_
            && this->multigroupOpacity
            && !this->withCompton;
        GroupArray physicalPdf{};
        GroupArray samplingPdf{};
        bool groupPdfValid = false;
        bool groupScoreAvailable = false;
        if (useGroupFreqSampling) {
            auto it = adaptiveSourceCellGroupScores_.find(cell.ID);
            if (it != adaptiveSourceCellGroupScores_.end()) {
                groupScoreAvailable = true;
                physicalPdf = this->multigroupOpacity->GetThermalGroupPdf(cell);
                double totalPhys = 0.0;
                size_t nPhysGroups = 0;
                for (size_t g = 0; g < ENERGY_GROUPS_NUM; ++g) {
                    if (physicalPdf[g] > 0.0) {
                        ++nPhysGroups;
                        totalPhys += physicalPdf[g];
                    }
                }
                if (totalPhys > 0.0 && nPhysGroups > 0) {
                    for (size_t g = 0; g < ENERGY_GROUPS_NUM; ++g)
                        physicalPdf[g] = (physicalPdf[g] > 0.0) ? physicalPdf[g] / totalPhys : 0.0;
                    GroupArray const& learnedScoreRaw = it->second;
                    double const scoreFloor = 1e-12;
                    GroupArray learnedPdf{};
                    double learnedTotal = 0.0;
                    for (size_t g = 0; g < ENERGY_GROUPS_NUM; ++g) {
                        if (physicalPdf[g] > 0.0) {
                            learnedPdf[g] = std::max(learnedScoreRaw[g], scoreFloor);
                            learnedTotal += learnedPdf[g];
                        }
                    }
                    if (learnedTotal > 0.0) {
                        for (size_t g = 0; g < ENERGY_GROUPS_NUM; ++g)
                            learnedPdf[g] /= learnedTotal;
                        for (size_t g = 0; g < ENERGY_GROUPS_NUM; ++g) {
                            samplingPdf[g] = (1.0 - adaptiveGroupStrength_) * physicalPdf[g]
                                           + adaptiveGroupStrength_ * learnedPdf[g];
                        }
                        double floorPerGroup = (nPhysGroups > 0) ? adaptiveGroupPdfFloor_ / static_cast<double>(nPhysGroups) : 0.0;
                        GroupArray lowerBound{};
                        GroupArray upperBound{};
                        double lowerTotal = 0.0;
                        double upperTotal = 0.0;
                        for (size_t g = 0; g < ENERGY_GROUPS_NUM; ++g) {
                            if (physicalPdf[g] > 0.0) {
                                lowerBound[g] = std::max(floorPerGroup,
                                    physicalPdf[g] / adaptiveGroupMaxWeightCorrection_);
                                upperBound[g] = std::min(1.0, adaptiveGroupMaxBias_ * physicalPdf[g]);
                                lowerBound[g] = std::min(lowerBound[g], upperBound[g]);
                                lowerTotal += lowerBound[g];
                                upperTotal += upperBound[g];
                            } else {
                                samplingPdf[g] = 0.0;
                            }
                        }

                        if (lowerTotal <= 1.0 + 1e-12 && upperTotal >= 1.0 - 1e-12) {
                            std::array<bool, ENERGY_GROUPS_NUM> fixed{};
                            double remaining = 1.0;
                            for (size_t g = 0; g < ENERGY_GROUPS_NUM; ++g) {
                                if (!(physicalPdf[g] > 0.0)) {
                                    fixed[g] = true;
                                    samplingPdf[g] = 0.0;
                                }
                            }

                            for (size_t iter = 0; iter < ENERGY_GROUPS_NUM + 2; ++iter) {
                                double freeTotal = 0.0;
                                for (size_t g = 0; g < ENERGY_GROUPS_NUM; ++g) {
                                    if (!fixed[g])
                                        freeTotal += std::max(samplingPdf[g], 0.0);
                                }
                                if (!(freeTotal > 0.0)) {
                                    groupPdfValid = false;
                                    break;
                                }

                                bool clamped = false;
                                double const scale = remaining / freeTotal;
                                for (size_t g = 0; g < ENERGY_GROUPS_NUM; ++g) {
                                    if (fixed[g])
                                        continue;
                                    double const candidate = std::max(samplingPdf[g], 0.0) * scale;
                                    if (candidate < lowerBound[g]) {
                                        samplingPdf[g] = lowerBound[g];
                                        fixed[g] = true;
                                        remaining -= lowerBound[g];
                                        clamped = true;
                                    } else if (candidate > upperBound[g]) {
                                        samplingPdf[g] = upperBound[g];
                                        fixed[g] = true;
                                        remaining -= upperBound[g];
                                        clamped = true;
                                    }
                                }

                                if (!clamped) {
                                    for (size_t g = 0; g < ENERGY_GROUPS_NUM; ++g) {
                                        if (!fixed[g])
                                            samplingPdf[g] = std::max(samplingPdf[g], 0.0) * scale;
                                    }
                                    remaining = 0.0;
                                    break;
                                }
                                if (remaining < 0.0)
                                    break;
                            }
                        } else {
                            for (size_t g = 0; g < ENERGY_GROUPS_NUM; ++g)
                                samplingPdf[g] = 0.0;
                        }

                        double sampTotal = 0.0;
                        for (size_t g = 0; g < ENERGY_GROUPS_NUM; ++g)
                            sampTotal += samplingPdf[g];
                        if (sampTotal > 0.0) {
                            for (size_t g = 0; g < ENERGY_GROUPS_NUM; ++g)
                                samplingPdf[g] /= sampTotal;
                            groupPdfValid = true;
                            for (size_t g = 0; g < ENERGY_GROUPS_NUM; ++g) {
                                if (physicalPdf[g] > 0.0) {
                                    if (!(samplingPdf[g] > 0.0)) {
                                        groupPdfValid = false;
                                        break;
                                    }
                                    double const correction = SafeGroupWeightCorrection(physicalPdf[g], samplingPdf[g]);
                                    if (!std::isfinite(correction) ||
                                        correction > adaptiveGroupMaxWeightCorrection_ * (1.0 + 1e-10) ||
                                        samplingPdf[g] > adaptiveGroupMaxBias_ * physicalPdf[g] * (1.0 + 1e-10)) {
                                        groupPdfValid = false;
                                        break;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        if (useGroupFreqSampling && groupScoreAvailable && !groupPdfValid) {
            ++lastGroupSamplingDiagnostics_.invalidPdfFallback;
            lastGroupSamplingDiagnostics_.invalidPdfFallbackPackets += nPhotonsCell;
        }

        for(size_t j = 0; j < nPhotonsCell; j++)
        {
            MCParticle particle = this->generateSingleParticle(i, cell);
            particle.cellID = cell.ID;
            particle.sourceCellID = cell.ID;
            particle.timeLeft = fullDt * this->dist(this->re);

            double weightCorrection = 1.0;
            bool usedGroupFrequencySampling = false;

            if (groupPdfValid) {
                double rndGroup = this->dist(this->re);
                double cumul = 0.0;
                size_t selectedGroup = ENERGY_GROUPS_NUM - 1;
                for (size_t g = 0; g < ENERGY_GROUPS_NUM; ++g) {
                    cumul += samplingPdf[g];
                    if (rndGroup <= cumul) {
                        selectedGroup = g;
                        break;
                    }
                }
                double freqCo = 0.0;

                if (samplingPdf[selectedGroup] > 0.0) {
                    weightCorrection = SafeGroupWeightCorrection(
                        physicalPdf[selectedGroup], samplingPdf[selectedGroup]);
                    if (weightCorrection > adaptiveGroupMaxWeightCorrection_) {
                        ++lastGroupSamplingDiagnostics_.weightCorrectionFallback;
                    } else if (weightCorrection > 0.0 && std::isfinite(weightCorrection)) {
                        if (lastGroupSamplingDiagnostics_.weightCorrectionCount == 0)
                            lastGroupSamplingDiagnostics_.weightCorrectionMin = weightCorrection;
                        else
                            lastGroupSamplingDiagnostics_.weightCorrectionMin = std::min(lastGroupSamplingDiagnostics_.weightCorrectionMin, weightCorrection);
                        if (lastGroupSamplingDiagnostics_.weightCorrectionCount == 0)
                            lastGroupSamplingDiagnostics_.weightCorrectionMax = weightCorrection;
                        else
                            lastGroupSamplingDiagnostics_.weightCorrectionMax = std::max(lastGroupSamplingDiagnostics_.weightCorrectionMax, weightCorrection);
                        lastGroupSamplingDiagnostics_.weightCorrectionSum += weightCorrection;
                        ++lastGroupSamplingDiagnostics_.weightCorrectionCount;
                        ++lastGroupSamplingDiagnostics_.totalSampled;
                        lastGroupSamplingDiagnostics_.sampledEnergy += energyPerPhoton;
                        double rndFreq = this->dist(this->re);
                        freqCo = this->multigroupOpacity->SampleThermalEnergyInGroup(cell, selectedGroup, rndFreq);
                        usedGroupFrequencySampling = true;
                    } else {
                        ++lastGroupSamplingDiagnostics_.weightCorrectionFallback;
                    }
                } else {
                    ++lastGroupSamplingDiagnostics_.weightCorrectionFallback;
                }

                if (usedGroupFrequencySampling && this->useTransportVelocities_ && !this->MMC) {
                    double D = DopplerShift(particle, cell.velocity);
                    particle.frequency = freqCo / D;
                    particle.weight = energyToCreate / (nPhotonsCell * D) * weightCorrection;
                } else if (usedGroupFrequencySampling) {
                    particle.frequency = freqCo;
                    particle.weight = energyPerPhoton * weightCorrection;
                }
            }

            if(!usedGroupFrequencySampling && this->useTransportVelocities_ && !this->MMC)
            {
                double D = DopplerShift(particle, cell.velocity);
                if(this->multigroupOpacity)
                {
                    double rnd = this->dist(this->re);
                    double freqCo = this->multigroupOpacity->GetThermalEnergy(cell, rnd);
                    particle.frequency = freqCo / D;
                }
                particle.weight = energyToCreate / (nPhotonsCell * D);
            }
            else if(!usedGroupFrequencySampling)
            {
                if(this->multigroupOpacity)
                {
                    particle.frequency = this->multigroupOpacity->GetThermalEnergy(cell, this->dist(this->re));
                }
                particle.weight = energyPerPhoton;
            }
            SetInitialWeightFromWeight(particle);
            newParticles.push_back(particle);
        }
    }
    if (lastGroupSamplingDiagnostics_.sampledEnergy > 0.0)
        lastGroupSamplingDiagnostics_.cappedEnergyFraction =
            lastGroupSamplingDiagnostics_.cappedEnergy /
            lastGroupSamplingDiagnostics_.sampledEnergy;
    lastGroupSamplingDiagnostics_.estimatorPotentiallyBiased =
        lastGroupSamplingDiagnostics_.weightCorrectionCapped > 0 ||
        lastGroupSamplingDiagnostics_.cappedEnergy > 0.0;
    return newParticles;
}

std::vector<typename RadiationIMC::Particle> RadiationIMC::generateComptonParticles(double fullDt)
{
    std::vector<Particle> newParticles;
    size_t const Ncells = this->grid.GetPointNo();
    if(this->comptonData.size() != Ncells)
        throw UniversalError("Compton data is not initialized in RadiationIMC::generateComptonParticles");

    std::vector<double> absEnergyToCreateVec(Ncells, 0.0);
    for(size_t i = 0; i < Ncells; i++)
    {
        for(size_t g = 0; g < ENERGY_GROUPS_NUM; g++)
            absEnergyToCreateVec[i] += this->comptonData[i].Bbase[g];
    }

    for(size_t i = 0; i < Ncells; i++)
    {
        ComputationalCell3D &cell = this->cells[i];
        ComptonCellData const &cd = this->comptonData[i];
        double const gamma = (this->withHydro && !this->MMC) ? 1 / std::sqrt(1 - ScalarProd(cell.velocity, cell.velocity) * units::inv_clight2) : 1;

        GroupArray absGroupEnergy{};
        std::array<size_t, ENERGY_GROUPS_NUM> groupCounts{};
        GroupArray fractional{};
        size_t nonzeroGroups = 0;
        for(size_t g = 0; g < ENERGY_GROUPS_NUM; g++)
        {
            absGroupEnergy[g] = cd.Bbase[g];
            if(absGroupEnergy[g] > 0.0)
                ++nonzeroGroups;
        }
        size_t const nPhotonsCell = std::max(this->newPhotonsPerCell, nonzeroGroups);
        if(nPhotonsCell == 0 || absEnergyToCreateVec[i] <= 0.0)
            continue;

        size_t allocated = 0;
        for(size_t g = 0; g < ENERGY_GROUPS_NUM; g++)
        {
            if(absGroupEnergy[g] <= 0.0)
                continue;
            groupCounts[g] = 1;
            ++allocated;
        }
        size_t const remainingBudget = nPhotonsCell - allocated;
        size_t extraAllocated = 0;
        for(size_t g = 0; g < ENERGY_GROUPS_NUM; g++)
        {
            if(absGroupEnergy[g] <= 0.0)
                continue;
            double const exactExtra = static_cast<double>(remainingBudget) * absGroupEnergy[g] / absEnergyToCreateVec[i];
            size_t const extra = static_cast<size_t>(std::floor(exactExtra));
            groupCounts[g] += extra;
            fractional[g] = exactExtra - static_cast<double>(extra);
            extraAllocated += extra;
        }
        while(extraAllocated < remainingBudget)
        {
            size_t bestGroup = ENERGY_GROUPS_NUM;
            double bestFraction = -1.0;
            for(size_t g = 0; g < ENERGY_GROUPS_NUM; g++)
            {
                if(absGroupEnergy[g] > 0.0 && fractional[g] > bestFraction)
                {
                    bestGroup = g;
                    bestFraction = fractional[g];
                }
            }
            if(bestGroup == ENERGY_GROUPS_NUM)
                break;
            ++groupCounts[bestGroup];
            fractional[bestGroup] = 0.0;
            ++extraAllocated;
        }

        size_t riskSourceBudget = std::max<size_t>(8, nPhotonsCell / 4);
        std::array<size_t, ENERGY_GROUPS_NUM> riskOrder{};
        for(size_t g = 0; g < ENERGY_GROUPS_NUM; g++)
            riskOrder[g] = g;
        std::sort(riskOrder.begin(), riskOrder.end(), [&](size_t a, size_t b)
        {
            return cd.riskScore[a] > cd.riskScore[b];
        });
        for(size_t orderIndex = 0; orderIndex < ENERGY_GROUPS_NUM && riskSourceBudget > 0; orderIndex++)
        {
            size_t const g = riskOrder[orderIndex];
            size_t const target = cd.riskTargetPackets[g];
            if(target == 0 || absGroupEnergy[g] <= 0.0 || groupCounts[g] >= target)
                continue;
            size_t const add = std::min(target - groupCounts[g], riskSourceBudget);
            groupCounts[g] += add;
            riskSourceBudget -= add;
        }

        for(size_t g = 0; g < ENERGY_GROUPS_NUM; g++)
        {
            size_t const ng = groupCounts[g];
            double const groupEnergy = cd.Bbase[g];
            if(ng == 0 || groupEnergy == 0.0)
                continue;

            if(!this->noHydroFeedback)
            {
                double const materialDeposit = -groupEnergy;
                this->conserved[i].internal_energy += materialDeposit;
                this->comptonSourceMaterialExchange += materialDeposit;
                this->conserved[i].energy -= groupEnergy * gamma;
                if(this->withHydro && !this->diffusionPressureGradient)
                    this->conserved[i].momentum -= groupEnergy * cell.velocity * units::inv_clight2 * gamma;
            }

            double const packetEnergy = groupEnergy / static_cast<double>(ng);
            for(size_t j = 0; j < ng; j++)
            {
                Particle particle = this->generateSingleParticle(i, cell);
                particle.timeLeft = fullDt * this->dist(this->re);
                particle.cellID = cell.ID;
                particle.sourceCellID = cell.ID;
                if(this->withHydro && !this->MMC)
                {
                    double const D = DopplerShift(particle, cell.velocity);
                    particle.frequency = this->frequencyForComptonGroup(g) / D;
                    particle.weight = packetEnergy / D;
                }
                else
                {
                    particle.frequency = this->frequencyForComptonGroup(g);
                    particle.weight = packetEnergy;
                }
                ClampFrequencyToBounds(particle.frequency);
                SetInitialWeightFromWeight(particle);
                if(particle.initialWeight > 0.0)
                    newParticles.push_back(particle);
            }
        }
    }

    return newParticles;
}

void RadiationIMC::initializeComptonGroups()
{
    if(this->comptonGroupsInitialized)
        return;

    for(size_t g = 0; g < ENERGY_GROUPS_NUM; g++)
    {
        double const left = ComputationalCell3D::energyBoundaries[g];
        double const right = ComputationalCell3D::energyBoundaries[g + 1];
        if(std::isnan(left) || std::isnan(right) || left >= right)
        {
            UniversalError eo("Invalid Compton energy group boundaries");
            eo.addEntry("Group", g);
            eo.addEntry("Left boundary", left);
            eo.addEntry("Right boundary", right);
            throw eo;
        }
        this->comptonGroupCenters[g] = 0.5 * (left + right);
        this->comptonGroupWidths[g] = right - left;
    }
    this->comptonGroupsInitialized = true;
}

void RadiationIMC::initializeComptonMatrixGenerator()
{
    if(this->comptonMatrixGen)
        return;

    this->initializeComptonGroups();
    this->comptonMatrixGen = std::make_unique<ComptonMatrixMC>(
        BuildComptonCentersVector(),
        BuildComptonBoundariesVector(),
        this->comptonMatrixSamples,
        true,
        1);
    this->comptonMatrixGen->set_tables(BuildComptonTemperatures());
}

RadiationIMC::GroupCdf RadiationIMC::buildSafeComptonCdf(const GroupArray &weights)
{
    GroupCdf cdf{};
    cdf[0] = 0.0;
    double total = 0.0;
    for(size_t g = 0; g < ENERGY_GROUPS_NUM; g++)
    {
        total += std::max(0.0, weights[g]);
        cdf[g + 1] = total;
    }
    if(total <= 0.0)
    {
        for(size_t g = 0; g <= ENERGY_GROUPS_NUM; g++)
            cdf[g] = static_cast<double>(g) / static_cast<double>(ENERGY_GROUPS_NUM);
        return cdf;
    }
    for(double &value : cdf)
        value /= total;
    cdf[ENERGY_GROUPS_NUM] = 1.0;
    return cdf;
}

double RadiationIMC::frequencyForComptonGroup(size_t group) const
{
    if(group >= ENERGY_GROUPS_NUM)
        throw UniversalError("Invalid Compton group in RadiationIMC::frequencyForComptonGroup");
    double frequency = 0.5 * (ComputationalCell3D::energyBoundaries[group] +
                              ComputationalCell3D::energyBoundaries[group + 1]);
    ClampFrequencyToBounds(frequency);
    return frequency;
}

void RadiationIMC::buildComptonMatricesForCell(const ComputationalCell3D &cell, size_t cellIndex, ComptonOccupationMode occupationMode, ComptonCellData &cd)
{
    this->initializeComptonMatrixGenerator();

    double constexpr fac = boost::math::pow<3>(units::clight) / (8.0 * M_PI * units::planck_constant);
    for(size_t g = 0; g < ENERGY_GROUPS_NUM; g++)
    {
        if(occupationMode == ComptonOccupationMode::RadiationField)
        {
            double const dnu = this->comptonGroupWidths[g] / units::planck_constant;
            double const nu = this->comptonGroupCenters[g] / units::planck_constant;
            double const Eg = std::max(0.0, cell.Eg[g] * cell.density);
            cd.occupation[g] = std::min(100.0, fac * Eg / (boost::math::pow<3>(nu) * dnu));
        }
        else if(occupationMode == ComptonOccupationMode::PlanckFunction)
        {
            double const dnu = this->comptonGroupWidths[g] / units::planck_constant;
            double const nu = this->comptonGroupCenters[g] / units::planck_constant;
            double const Eg = std::max(0.0, cd.planckFraction[g] * cd.Um);
            double const occupation = fac * Eg / (boost::math::pow<3>(nu) * dnu);
            cd.occupation[g] = std::clamp(occupation, 0.0, 100.0);
        }
        else
        {
            cd.occupation[g] = 0.0;
        }
    }

    Matrix tau(ENERGY_GROUPS_NUM, std::vector<double>(ENERGY_GROUPS_NUM, 0.0));
    Matrix dtau(ENERGY_GROUPS_NUM, std::vector<double>(ENERGY_GROUPS_NUM, 0.0));
    double const A = 1.0;
    double const Z = 1.0;
    double const temperature = std::min(this->comptonMatrixGen->get_maximum_temperature_grid() * 0.9999, cell.temperature);
    this->comptonMatrixGen->get_tau_matrix(temperature, cell.density, A, Z, tau, dtau);

    for(size_t h = 0; h < ENERGY_GROUPS_NUM; h++)
    {
        for(size_t g = 0; g < ENERGY_GROUPS_NUM; g++)
        {
            cd.tau[h][g] = tau[h][g];
            cd.dtau_dUm[h][g] = dtau[h][g];
        }
    }

    auto const lastGroup = this->comptonMatrixGen->get_last_group_upscattering_and_downscattering(temperature, cell.density, A, Z);
    double const upScatteringLast = lastGroup.first;
    double const downScatteringLast = lastGroup.second;

    ZeroGroupMatrix(cd.S);
    ZeroGroupMatrix(cd.dSdUm);

    for(size_t g = 0; g < ENERGY_GROUPS_NUM; g++)
    {
        for(size_t gt = 0; gt < ENERGY_GROUPS_NUM; gt++)
        {
            if(g + 1 == ENERGY_GROUPS_NUM && gt + 1 == ENERGY_GROUPS_NUM)
            {
                cd.S[g][g] += (upScatteringLast - downScatteringLast) * (1.0 + cd.occupation[g]);
                cd.dSdUm[g][g] += cd.dtau_dUm[g][g] * (1.0 + cd.occupation[g]);
                continue;
            }

            double const inScatteringFactor = this->comptonGroupCenters[g] / this->comptonGroupCenters[gt] * (1.0 + cd.occupation[g]);
            cd.S[gt][g] += cd.tau[gt][g] * inScatteringFactor;
            cd.dSdUm[gt][g] += cd.dtau_dUm[gt][g] * inScatteringFactor;

            double const outScatteringFactor = 1.0 + cd.occupation[gt];
            cd.S[g][g] -= cd.tau[g][gt] * outScatteringFactor;
            cd.dSdUm[g][g] -= cd.dtau_dUm[g][gt] * outScatteringFactor;
        }
    }

    double const UmFactor = 1.0 / (4.0 * units::arad * boost::math::pow<3>(cell.temperature));
    for(size_t h = 0; h < ENERGY_GROUPS_NUM; h++)
    {
        for(size_t g = 0; g < ENERGY_GROUPS_NUM; g++)
        {
            cd.dSdUm[h][g] *= UmFactor;
        }
    }

    (void) cellIndex;
}

void RadiationIMC::recomputeComptonContractions(ComptonCellData &cd)
{
    cd.Upsilon = 0.0;
    for(size_t g = 0; g < ENERGY_GROUPS_NUM; g++)
    {
        cd.D[g] = 0.0;
        for(size_t h = 0; h < ENERGY_GROUPS_NUM; h++)
            cd.D[g] += cd.dSdUm[h][g] * cd.oldRadiationEnergy[h];
        cd.Upsilon += cd.D[g];
        cd.M[g] = cd.absorptionOpacity[g] * cd.planckFraction[g] + cd.D[g];
    }

    for(size_t h = 0; h < ENERGY_GROUPS_NUM; h++)
    {
        cd.rowS[h] = 0.0;
        for(size_t g = 0; g < ENERGY_GROUPS_NUM; g++)
            cd.rowS[h] += cd.S[h][g];
        cd.Lambda[h] = cd.absorptionOpacity[h] - cd.rowS[h];
    }

    cd.Gamma = cd.planckOpacity + cd.Upsilon;
}

void RadiationIMC::buildComptonSources(double fullDt, ComptonCellData &cd)
{
    double const cdt = units::clight * fullDt;
    for(size_t g = 0; g < ENERGY_GROUPS_NUM; g++)
    {
        double const kgbg = cd.absorptionOpacity[g] * cd.planckFraction[g];
        cd.Bbase[g] = cd.volume * cdt * cd.fleck * kgbg * cd.Um;

        if(cd.planckOpacity > 0.0)
        {
            cd.Bcorr[g] = cd.volume * cdt * cd.planckOpacity * cd.Um *
                ((kgbg / cd.planckOpacity) * (1.0 - (1.0 + cd.beta * cdt * cd.planckOpacity) * cd.fleck)
                 - cd.beta * cdt * cd.fleck * cd.D[g]);
        }
        else
        {
            cd.Bcorr[g] = 0.0;
        }
        cd.Btotal[g] = cd.Bbase[g] + cd.Bcorr[g];
        cd.Bpos[g] = std::max(0.0, cd.Btotal[g]);
        cd.Bres[g] = cd.Btotal[g] - cd.Bpos[g];
    }
}

void RadiationIMC::computeComptonRiskForCell(size_t cellIndex, double fullDt, ComptonCellData &cd)
{
    (void)cellIndex;
    GroupArray rhs{};
    GroupArray predicted{};
    GroupMatrix residualMatrix{};
    double totalOldExtensive = 0.0;

    for(size_t g = 0; g < ENERGY_GROUPS_NUM; g++)
    {
        double const oldExtensive = std::max(0.0, cd.oldRadiationEnergy[g]) * cd.volume;
        rhs[g] = oldExtensive + cd.Bcorr[g];
        totalOldExtensive += oldExtensive;
        cd.riskScore[g] = 0.0;
        cd.riskTargetPackets[g] = 0;
    }
    if(totalOldExtensive <= 0.0)
        return;

    for(size_t g = 0; g < ENERGY_GROUPS_NUM; g++)
    {
        for(size_t h = 0; h < ENERGY_GROUPS_NUM; h++)
        {
            residualMatrix[g][h] = ((g == h) ? 1.0 : 0.0)
                - fullDt * units::clight * cd.residualKernel[h][g];
        }
    }

    SolverDiagnostics diag;
    bool const solved = SolveComptonGroupSystem(residualMatrix, rhs, predicted, diag);
    for(size_t g = 0; g < ENERGY_GROUPS_NUM; g++)
    {
        double const oldExtensive = std::max(0.0, cd.oldRadiationEnergy[g]) * cd.volume;
        double const groupFloor = std::max(1e-10 * totalOldExtensive, 1.0);
        double const scale = std::max(oldExtensive, groupFloor);
        double score = std::abs(cd.Bcorr[g]) / scale;
        if(solved)
        {
            double const depletion = oldExtensive - predicted[g];
            if(depletion > 0.0)
                score = std::max(score, depletion / scale);
            if(predicted[g] < 0.0)
                score = std::max(score, 1.0 + std::abs(predicted[g]) / scale);
        }
        else
        {
            score = std::max(score, 2.0);
        }

        if(oldExtensive <= 1e-8 * totalOldExtensive && score < 10.0)
            continue;
        if(score < 0.5)
            continue;

        cd.riskScore[g] = score;
        if(score >= 10.0)
            cd.riskTargetPackets[g] = 96;
        else if(score >= 3.0)
            cd.riskTargetPackets[g] = 64;
        else if(score >= 1.0)
            cd.riskTargetPackets[g] = 32;
        else
            cd.riskTargetPackets[g] = 16;
    }
}

void RadiationIMC::buildComptonEventData(size_t cellIndex, ComptonCellData &cd)
{
    (void)cellIndex;
    ZeroGroupMatrix(cd.segmentKernel);
    ZeroGroupMatrix(cd.residualKernel);

    for(size_t h = 0; h < ENERGY_GROUPS_NUM; h++)
    {
        GroupArray weights{};
        double outRateSum = 0.0;
        for(size_t g = 0; g < ENERGY_GROUPS_NUM; g++)
        {
            if(g == h)
                continue;
            weights[g] = std::max(0.0, cd.tau[h][g] * (1.0 + cd.occupation[g]));
            outRateSum += weights[g];
        }
        cd.comptonOutRate[h] = outRateSum;
        cd.comptonTargetCdf[h] = RadiationIMC::buildSafeComptonCdf(weights);

        cd.baseEffectiveOpacity[h] = 0.0;
        for(size_t g = 0; g < ENERGY_GROUPS_NUM; g++)
        {
            double const kgbg = cd.absorptionOpacity[g] * cd.planckFraction[g];
            cd.baseEffectiveOpacity[h] += cd.betaCdtF * cd.absorptionOpacity[h] * kgbg;
        }

        double const f = cd.fleck;

        double mu = 0.0;
        if(outRateSum > 0.0)
        {
            for(size_t g = 0; g < ENERGY_GROUPS_NUM; g++)
            {
                if(g == h)
                    continue;
                double const q_hg = weights[g] / outRateSum;
                double const r_hg = this->comptonGroupCenters[g] / this->comptonGroupCenters[h];
                mu += q_hg * r_hg;
            }
        }
        else
        {
            mu = 1.0;
        }
        cd.comptonMu[h] = mu;
        cd.comptonMh[h] = 1.0 + f * (mu - 1.0);

        for(size_t g = 0; g < ENERGY_GROUPS_NUM; g++)
        {
            double const kgbg = cd.absorptionOpacity[g] * cd.planckFraction[g];
            double const Ktotal_hg = cd.S[h][g] + cd.betaCdtF * cd.M[g] * cd.Lambda[h];
            double const Hbase_hg = cd.betaCdtF * cd.absorptionOpacity[h] * kgbg;
            double const Ktarget_hg = Ktotal_hg - Hbase_hg;
            cd.segmentKernel[h][g] = Ktarget_hg;

            double const diag = (h == g) ? -cd.absorptionOpacity[h] : 0.0;
            cd.Ktotal[h][g] = diag + Ktotal_hg;

            double Kimc_hg = (h == g)
                ? -cd.absorptionOpacity[h] + (1.0 - f) * cd.absorptionOpacity[h] * cd.baseSourceFraction[h]
                : (1.0 - f) * cd.absorptionOpacity[h] * cd.baseSourceFraction[g];
            double Kevent_new_hg;
            if(h == g)
                Kevent_new_hg = -cd.comptonOutRate[h];
            else
            {
                double const q_hg = (outRateSum > 0.0) ? weights[g] / outRateSum : 0.0;
                Kevent_new_hg = cd.comptonOutRate[h] * q_hg * cd.comptonMh[h];
            }
            cd.residualKernel[h][g] = cd.Ktotal[h][g] - Kimc_hg - Kevent_new_hg;
        }
    }
}

void RadiationIMC::applyComptonScatterEvent(size_t cellIndex, const ComputationalCell3D &cell, size_t sourceGroup, const Vector3D &oldVelocity, double oldWeight, double dopplerShift, Particle &particle)
{
    (void)dopplerShift;
    if(sourceGroup >= ENERGY_GROUPS_NUM)
        return;

    ComptonCellData &cd = this->comptonData[cellIndex];
    if(cd.comptonOutRate[sourceGroup] <= 0.0)
        return;

    size_t targetGroup = this->sampleComptonCdf(
        cd.comptonTargetCdf[sourceGroup], this->dist(this->re));
    if(targetGroup == sourceGroup || targetGroup >= ENERGY_GROUPS_NUM)
    {
        UniversalError eo("Compton CDF sampling returned invalid target group");
        eo.addEntry("Cell index", static_cast<double>(cellIndex));
        eo.addEntry("Source group", static_cast<double>(sourceGroup));
        eo.addEntry("Target group", static_cast<double>(targetGroup));
        eo.addEntry("comptonOutRate", cd.comptonOutRate[sourceGroup]);
        throw eo;
    }

    if(this->comptonAngleDependent)
    {
        static thread_local std::vector<double> angleCdf;
        this->comptonMatrixGen->get_angle_cdf(cd.temperature, sourceGroup, targetGroup, angleCdf);

        double const r = this->dist(this->re);
        std::size_t const N = ComptonMatrixMC::NUM_ANGLE_BINS;

        auto it = std::upper_bound(angleCdf.begin(), angleCdf.end(), r);
        std::size_t bin = 0;
        if(it != angleCdf.begin())
            bin = static_cast<std::size_t>(std::distance(angleCdf.begin(), it)) - 1;
        if(bin >= N)
            bin = N - 1;

        double const binWidth = 2.0 / static_cast<double>(N);
        double frac = 0.0;
        double const denom = angleCdf[bin + 1] - angleCdf[bin];
        if(denom > 0.0)
            frac = (r - angleCdf[bin]) / denom;
        double const cosTheta = -1.0 + (static_cast<double>(bin) + frac) * binWidth;
        double const sinTheta = std::sqrt(std::max(0.0, 1.0 - cosTheta * cosTheta));
        double const phi = 2.0 * M_PI * this->dist(this->re);

        Vector3D const oldDir = normalize(oldVelocity);

        Vector3D perp1;
        if(std::abs(oldDir.z) < 0.9)
            perp1 = normalize(CrossProduct(oldDir, Vector3D(0, 0, 1)));
        else
            perp1 = normalize(CrossProduct(oldDir, Vector3D(1, 0, 0)));
        Vector3D const perp2 = CrossProduct(oldDir, perp1);

        Vector3D const newDir = oldDir * cosTheta
            + (perp1 * std::cos(phi) + perp2 * std::sin(phi)) * sinTheta;

        particle.velocity = normalize(newDir) * units::clight;
    }
    else
    {
        particle.velocity = this->opacity->getNewScatterVelocity(cell, particle);
    }

    double const mh = cd.comptonMh[sourceGroup];
    double const newWeight = oldWeight * mh;
    double const materialDeposit = oldWeight * (1.0 - mh);

    particle.weight = newWeight;
    particle.frequency = this->frequencyForComptonGroup(targetGroup);

    if(!this->noHydroFeedback)
    {
        this->conserved[cellIndex].internal_energy += materialDeposit;
        this->conserved[cellIndex].energy += materialDeposit;
        this->comptonContinuousMaterialExchange += materialDeposit;
        if(this->withHydro && !this->diffusionPressureGradient)
        {
            this->conserved[cellIndex].momentum +=
                (oldWeight * oldVelocity - newWeight * particle.velocity) * units::inv_clight2;
        }
    }
}

void RadiationIMC::resetComptonDiagnostics()
{
    this->comptonSourceMaterialExchange = 0.0;
    this->comptonContinuousMaterialExchange = 0.0;
    this->comptonImplicitMaterialExchange = 0.0;
    this->comptonRemovalMaterialExchange = 0.0;
    this->comptonMinGroupEnergy = std::numeric_limits<double>::infinity();
    this->comptonMaxGroupEnergy = -std::numeric_limits<double>::infinity();
    this->comptonProjectedRadiationEnergy = 0.0;
    this->comptonMinFleck = std::numeric_limits<double>::infinity();
    this->comptonMaxFleck = -std::numeric_limits<double>::infinity();
    this->comptonMinGamma = std::numeric_limits<double>::infinity();
    this->comptonMaxGamma = -std::numeric_limits<double>::infinity();
    this->comptonMinUpsilon = std::numeric_limits<double>::infinity();
    this->comptonMaxUpsilon = -std::numeric_limits<double>::infinity();
    this->comptonNZeroFallbackCount = 0;
    this->comptonImplicitEventCount = 0;
    this->comptonOpacityLimitedGroupCount = 0;
    this->comptonProjectedNegativeGroupCount = 0;
}

void RadiationIMC::printComptonDiagnostics()
{
    if(!this->withCompton || !this->comptonDiagnostics)
        return;

    double sourceMaterialExchange = this->comptonSourceMaterialExchange;
    double continuousMaterialExchange = this->comptonContinuousMaterialExchange;
    double correctionMaterialExchange = this->comptonImplicitMaterialExchange;
    double removalMaterialExchange = this->comptonRemovalMaterialExchange;
    double minGroupEnergy = this->comptonMinGroupEnergy;
    double maxGroupEnergy = this->comptonMaxGroupEnergy;
    double projectedRadiationEnergy = this->comptonProjectedRadiationEnergy;
    double minFleck = this->comptonMinFleck;
    double maxFleck = this->comptonMaxFleck;
    double minGamma = this->comptonMinGamma;
    double maxGamma = this->comptonMaxGamma;
    double minUpsilon = this->comptonMinUpsilon;
    double maxUpsilon = this->comptonMaxUpsilon;
    size_t nZeroFallbackCount = this->comptonNZeroFallbackCount;
    size_t opacityLimitedGroupCount = this->comptonOpacityLimitedGroupCount;
    size_t projectedNegativeGroupCount = this->comptonProjectedNegativeGroupCount;
    int rank = 0;

    #ifdef RICH_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Allreduce(MPI_IN_PLACE, &sourceMaterialExchange, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &continuousMaterialExchange, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &correctionMaterialExchange, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &removalMaterialExchange, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &minGroupEnergy, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &maxGroupEnergy, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &projectedRadiationEnergy, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &minFleck, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &maxFleck, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &minGamma, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &maxGamma, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &minUpsilon, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &maxUpsilon, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &nZeroFallbackCount, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &opacityLimitedGroupCount, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &projectedNegativeGroupCount, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    #endif

    if(rank == 0)
    {
        std::cout << "Compton diagnostics:"
                  << " source_material_exchange=" << sourceMaterialExchange
                  << " continuous_material_exchange=" << continuousMaterialExchange
                  << " correction_material_exchange=" << correctionMaterialExchange
                  << " removal_material_exchange=" << removalMaterialExchange
                  << " min_group_energy=" << minGroupEnergy
                  << " max_group_energy=" << maxGroupEnergy
                  << " projected_radiation_energy=" << projectedRadiationEnergy
                  << " fleck_min=" << minFleck
                  << " fleck_max=" << maxFleck
                  << " gamma_min=" << minGamma
                  << " gamma_max=" << maxGamma
                  << " upsilon_min=" << minUpsilon
                  << " upsilon_max=" << maxUpsilon
                  << " n_zero_fallback_cells=" << nZeroFallbackCount
                  << " opacity_limited_groups=" << opacityLimitedGroupCount
                  << " projected_negative_groups=" << projectedNegativeGroupCount
                  << std::endl;
    }
    this->resetComptonDiagnostics();
}

size_t RadiationIMC::sampleComptonCdf(const GroupCdf &cdf, double random) const
{
    double const value = std::clamp(random, 0.0, std::nextafter(1.0, 0.0));
    auto it = std::upper_bound(cdf.begin(), cdf.end(), value);
    if(it == cdf.begin())
        return 0;
    size_t group = static_cast<size_t>(std::distance(cdf.begin(), it)) - 1;
    if(group >= ENERGY_GROUPS_NUM)
        group = ENERGY_GROUPS_NUM - 1;
    return group;
}

void RadiationIMC::precomputeComptonData(double fullDt)
{
    if(!this->withCompton)
    {
        this->comptonData.clear();
        return;
    }

    this->initializeComptonGroups();
    this->initializeComptonMatrixGenerator();

    size_t const Ncells = this->grid.GetPointNo();
    this->comptonData.assign(Ncells, ComptonCellData{});
    this->comptonRiskPrecomputeDt_ = fullDt;

    for(size_t i = 0; i < Ncells; i++)
    {
        ComputationalCell3D const &cell = this->cells[i];
        ComptonCellData &data = this->comptonData[i];
        data.volume = this->grid.GetVolume(i);
        data.temperature = cell.temperature;
        data.Um = units::arad * boost::math::pow<4>(cell.temperature);
        data.cv = this->eos->dT2cv(cell.density, cell.temperature, cell.tracers, cell.tracerNames);
        if(data.cv <= 0.0)
        {
            UniversalError eo("Invalid heat capacity in RadiationIMC::precomputeComptonData");
            eo.addEntry("Cell index", i);
            eo.addEntry("cv", data.cv);
            throw eo;
        }
        data.beta = 4.0 * units::arad * boost::math::pow<3>(cell.temperature) / data.cv;

        double const kT = units::k_boltz * cell.temperature;
        double planckIntegralTotal = 0.0;
        if(kT > 0.0)
        {
            for(size_t g = 0; g < ENERGY_GROUPS_NUM; g++)
            {
                double const a = ComputationalCell3D::energyBoundaries[g] / kT;
                double const b = ComputationalCell3D::energyBoundaries[g + 1] / kT;
                data.planckFraction[g] = planck_integral::planck_integral(a, b);
                planckIntegralTotal += data.planckFraction[g];
            }
        }

        for(size_t g = 0; g < ENERGY_GROUPS_NUM; g++)
        {
            if(planckIntegralTotal > 0.0)
                data.planckFraction[g] /= planckIntegralTotal;
            else
                data.planckFraction[g] = 0.0;

            double absorptionOpacity = this->opacity->CalcAbsorptionOpacity(cell, this->comptonGroupCenters[g]);
            double const groupRadiationEnergy = std::max(0.0, cell.Eg[g] * cell.density);
            if(this->capAbsorptionOpacity)
            {
                absorptionOpacity = std::min(absorptionOpacity, CG::max_coupling_strength / (units::clight * fullDt));

                double const groupExcess = groupRadiationEnergy - data.planckFraction[g] * data.Um;
                if(cell.density > 1e-12 &&
                   groupExcess > 0.0 &&
                   units::clight * fullDt * absorptionOpacity * groupExcess > 2.0 * data.cv * cell.temperature)
                {
                    double const limitedOpacity =
                        2.0 * data.cv * cell.temperature / (units::clight * fullDt * groupExcess);
                    if(limitedOpacity < absorptionOpacity)
                    {
                        absorptionOpacity = limitedOpacity;
                    }
                }
            }
            if(absorptionOpacity < 0.0)
            {
                UniversalError eo("Negative absorption coefficient in RadiationIMC::precomputeComptonData");
                eo.addEntry("Cell index", i);
                eo.addEntry("Group", g);
                eo.addEntry("Absorption opacity", absorptionOpacity);
                throw eo;
            }
            data.absorptionOpacity[g] = absorptionOpacity;
            data.planckOpacity += data.absorptionOpacity[g] * data.planckFraction[g];
            data.oldRadiationEnergy[g] = groupRadiationEnergy;
        }
        this->planckOpacities[i] = data.planckOpacity;

        for(size_t g = 0; g < ENERGY_GROUPS_NUM; g++)
        {
            if(data.planckOpacity > 0.0)
                data.baseSourceFraction[g] = data.absorptionOpacity[g] * data.planckFraction[g] / data.planckOpacity;
            else
                data.baseSourceFraction[g] = 0.0;
        }

        data.planckCdf = RadiationIMC::buildSafeComptonCdf(data.planckFraction);
        data.baseSourceCdf = RadiationIMC::buildSafeComptonCdf(data.baseSourceFraction);

        ComptonOccupationMode const initialOccupationMode = this->comptonUseInduced
            ? ComptonOccupationMode::RadiationField
            : ComptonOccupationMode::Zero;
        this->buildComptonMatricesForCell(cell, i, initialOccupationMode, data);
        this->recomputeComptonContractions(data);

        double const gamma = (this->useTransportVelocities_ && !this->MMC)
            ? 1.0 / std::sqrt(1.0 - ScalarProd(cell.velocity, cell.velocity) * units::inv_clight2)
            : 1.0;
        double const cdtEff = units::clight * fullDt * gamma;
        double denom = 1.0 + data.beta * cdtEff * data.Gamma;
        bool const negativeUpsilon = data.Upsilon < 0.0;
        if((denom <= 0.0 || negativeUpsilon) &&
           this->comptonAllowNZeroFallback)
        {
            ComptonOccupationMode fallbackOccupationMode = ComptonOccupationMode::Zero;
            if(negativeUpsilon &&
               this->comptonUseInduced &&
               this->comptonInducedMode == ComptonInducedMode::AdaptivePlanckFallback &&
               data.planckOpacity * units::clight * fullDt >= 1.0)
            {
                fallbackOccupationMode = ComptonOccupationMode::PlanckFunction;
            }

            this->buildComptonMatricesForCell(cell, i, fallbackOccupationMode, data);
            data.useNZero = fallbackOccupationMode == ComptonOccupationMode::Zero;
            data.usePlanckInduced = fallbackOccupationMode == ComptonOccupationMode::PlanckFunction;
            this->recomputeComptonContractions(data);
            denom = 1.0 + data.beta * cdtEff * data.Gamma;
        }
        if(denom <= 0.0)
        {
            UniversalError eo("Compton Fleck denominator is nonpositive in RadiationIMC::precomputeComptonData");
            eo.addEntry("Cell index", i);
            eo.addEntry("Denominator", denom);
            eo.addEntry("Gamma", data.Gamma);
            eo.addEntry("Upsilon", data.Upsilon);
            eo.addEntry("Planck opacity", data.planckOpacity);
            eo.addEntry("Full dt", fullDt);
            throw eo;
        }
        data.fleck = 1.0 / denom;
        if(data.fleck < 0.0 || data.fleck > 1.0)
        {
            UniversalError eo("Invalid Compton-modified Fleck factor in RadiationIMC::precomputeComptonData");
            eo.addEntry("Cell index", i);
            eo.addEntry("Fleck", data.fleck);
            eo.addEntry("Gamma", data.Gamma);
            eo.addEntry("Upsilon", data.Upsilon);
            eo.addEntry("Planck opacity", data.planckOpacity);
            throw eo;
        }
        this->factorFleck[i] = data.fleck;
        data.betaCdtF = data.beta * cdtEff * data.fleck;
        if(std::abs(data.Gamma) > 1e-200)
            data.betaCdtF = (1.0 - data.fleck) / data.Gamma;

        this->buildComptonEventData(i, data);
        this->buildComptonSources(fullDt, data);
        this->computeComptonRiskForCell(i, fullDt, data);
        data.active = true;
    }
}

std::vector<typename RadiationIMC::Particle> RadiationIMC::preStep(double fullDt)
{
    if (postProcess_.enabled && !observer_)
        throw UniversalError("RadiationIMC post-process mode requires observer");

    assert(this->cells.size() >= this->grid.GetPointNo());
    assert(this->conserved.size() >= this->grid.GetPointNo());

    size_t Ncells = this->grid.GetPointNo();
    double const emissionDt = postProcess_.enabled ? postProcess_.sourceDt : fullDt;
    bool const reuseComptonPrecompute =
        !postProcess_.enabled &&
        this->withCompton &&
        this->multigroupOpacity &&
        this->comptonDataReusableInPreStep_ &&
        this->comptonData.size() == Ncells &&
        this->factorFleck.size() == Ncells &&
        this->planckOpacities.size() == Ncells &&
        this->comptonRiskPrecomputeDt_ == emissionDt;

    if(!reuseComptonPrecompute)
    {
        this->factorFleck = std::vector<double>(Ncells);
        this->planckOpacities = std::vector<double>(Ncells);
    }
    this->Erad_time_avg = std::vector<double>(Ncells, 0);
    if((this->withEgTimeAvg || this->withCompton) && this->multigroupOpacity)
    {
        std::array<double, ENERGY_GROUPS_NUM> zeros{};
        this->Eg_time_avg.assign(Ncells, zeros);
    }
    if(this->multigroupOpacity)
    {
        this->multigroupOpacity->ResetCummulativeOpacityCellID();
    }
    for(size_t i = 0; i < Ncells; i++)
    {
        const ComputationalCell3D &cell = this->cells[i];
        double gamma = (this->withCompton ? (this->withHydro && !this->MMC) : (this->useTransportVelocities_ && !this->MMC))? 1 / std::sqrt(1 - ScalarProd(cell.velocity, cell.velocity) * units::inv_clight2) : 1;

        if(this->withCompton)
        {
            if(!reuseComptonPrecompute)
            {
                this->planckOpacities[i] = 0.0;
                this->factorFleck[i] = 1.0;
            }
        }
        else
        {
            this->planckOpacities[i] = this->opacity->CalcPlanckOpacity(this->cells[i]);
            double cv = this->eos->dT2cv(this->cells[i].density, this->cells[i].temperature, this->cells[i].tracers, this->cells[i].tracerNames);
            this->factorFleck[i] = 1 / (1 + (4 * units::arad * boost::math::pow<3>(this->cells[i].temperature) * this->planckOpacities[i] * units::clight * fullDt * gamma) / cv);
            if(this->factorFleck[i] < 0 or this->factorFleck[i] > 1)
            {
                UniversalError eo("Invalid factor fleck in RadiationIMC::preStep");
                eo.addEntry("Factor fleck", this->factorFleck[i]);
                eo.addEntry("Planck opacity", this->planckOpacities[i]);
                eo.addEntry("Temperature", this->cells[i].temperature);
                eo.addEntry("Density", this->cells[i].density);
                eo.addEntry("Gamma", gamma);
                eo.addEntry("cv", cv);
                eo.addEntry("Full dt", fullDt);
                throw eo;
            }
        }
    }

    if (postProcess_.enabled && !withCompton && postProcess_.forceGreyFleckOne)
    {
        for (size_t i = 0; i < Ncells; ++i)
            factorFleck[i] = 1.0;
    }

    if(this->withCompton && !reuseComptonPrecompute)
        this->precomputeComptonData(emissionDt);
    this->comptonDataReusableInPreStep_ = false;

    if(this->withRandomWalk)
    {
        this->precomputeRandomWalkData();
        this->rwStepCount = 0;
    }
    if(this->withDDMC)
    {
#ifdef RICH_IMC_DDMC_ENABLED
        this->precomputeDDMCData();
        size_t ddmcFluxCellCount = Ncells;
#ifdef RICH_MPI
        ddmcFluxCellCount = this->grid.GetTotalPointNumber();
#endif
        this->ddmcFluxRhsIntegrated.assign(ddmcFluxCellCount, Vector3D(0.0, 0.0, 0.0));
        this->ddmcStepCount = 0;
        this->ddmcLeakCount = 0;
        this->ddmcResidentLeakCount = 0;
        this->ddmcTransportLeakCount = 0;
        this->ddmcRemoteResidentLeakCount = 0;
        this->ddmcMomentumFeedbackCount = 0;
        this->ddmcMomentumMatrixFallbackCount = 0;
        this->ddmcMovingMediumUpdateCount = 0;
        this->ddmcFaceFrameShiftCount = 0;
        this->ddmcMaxMovingMediumLogShift = 0.0;
        this->ddmcMaxFaceFrameLogShift = 0.0;
        this->ddmcFaceFluxEnergy = 0.0;
        this->ddmcFaceFluxMpiEnergy = 0.0;
        this->ddmcMaterialEnergyExchangeCo = 0.0;
        this->ddmcMaterialEnergyExchangeLab = 0.0;
        this->ddmcMaterialMomentumExchangeLab = Vector3D(0.0, 0.0, 0.0);
        this->ddmcFluxMomentumExchangeLab = Vector3D(0.0, 0.0, 0.0);
        this->ddmcAppliedMomentumExchangeLab = Vector3D(0.0, 0.0, 0.0);
        this->ddmcLocalFaceFluxPairResidualMax = 0.0;
        this->ddmcWeightRatioMax = 0.0;
        this->ddmcWeightRatioSum = 0.0;
        this->ddmcWeightRatioSamples.clear();
        this->ddmcCensusCount = 0;
        this->ddmcUpscatterCount = 0;
        this->ddmcFallbackCount = 0;
        this->ddmcMpiFaceFluxReductionCount = 0;
        this->ddmcInterfaceFluxTallyCount = 0;
        this->ddmcBoundaryFluxTallyCount = 0;
        this->ddmcObserverEnergyOnlyTallyCount = 0;
        this->ddmcLocalFaceFluxPairCheckCount = 0;
        this->ddmcWeightRatioCount = 0;
        this->ddmcWeightRatioSamplesDropped = 0;
        this->ddmcWeightRatioOutlierCount = 0;
#else
        throw UniversalError("RadiationIMC DDMC precompute requested, but DDMC support was not compiled");
#endif
    }

    std::vector<Particle> newParticles = this->generateParticles(emissionDt);

    if (postProcess_.enabled)
    {
        double emitted = PostProcessIMC::PrepareGeneratedParticles(newParticles, postProcess_.transportTime);
        if (observer_)
            observer_->addEmittedEnergy(emitted);

        return newParticles;
    }

    std::vector<Particle> newParticles2 = this->boundary->generateNewBoundaryParticles(fullDt);
    for(Particle &particle : newParticles2)
    {
        SetInitialWeightFromWeight(particle);
    }
    newParticles.insert(newParticles.end(), newParticles2.begin(), newParticles2.end());
    return newParticles;
}

std::vector<typename RadiationIMC::Particle> RadiationIMC::generateInitialParticles(size_t particlesPerCell)
{
    std::vector<Particle> result;
    if(particlesPerCell == 0)
        return result;

    size_t Ncells = this->grid.GetPointNo();

    std::array<double, ENERGY_GROUPS_NUM + 1> cumulPlanck;
    bool hasPlanckTable = false;
    double cachedTemperature = -1;

    for(size_t i = 0; i < Ncells; i++)
    {
        double totalErad = this->cells[i].Erad * this->cells[i].density * this->grid.GetVolume(i);
        double weightPerPhoton = totalErad / particlesPerCell;
        if(weightPerPhoton <= 0)
            continue;

        if(this->multigroupOpacity && !this->withCompton && (!hasPlanckTable || this->cells[i].temperature != cachedTemperature))
        {
            double kT = units::k_boltz * this->cells[i].temperature;
            cumulPlanck[0] = 0.0;
            for(size_t g = 1; g <= ENERGY_GROUPS_NUM; g++)
            {
                double a = ComputationalCell3D::energyBoundaries[g - 1] / kT;
                double b = ComputationalCell3D::energyBoundaries[g] / kT;
                cumulPlanck[g] = planck_integral::planck_integral(a, b) + cumulPlanck[g - 1];
            }
            cachedTemperature = this->cells[i].temperature;
            hasPlanckTable = true;
        }
        GroupCdf initialGroupEnergyCdf{};
        if(this->withCompton && this->multigroupOpacity)
        {
            initialGroupEnergyCdf[0] = 0.0;
            double cumulativeEnergy = 0.0;
            for(size_t g = 0; g < ENERGY_GROUPS_NUM; g++)
            {
                cumulativeEnergy += std::max(0.0, this->cells[i].Eg[g] * this->cells[i].density * this->grid.GetVolume(i));
                initialGroupEnergyCdf[g + 1] = cumulativeEnergy;
            }
            if(cumulativeEnergy <= 0.0)
            {
                UniversalError eo("Compton initial census has nonpositive multigroup radiation energy");
                eo.addEntry("Cell index", i);
                eo.addEntry("Erad", this->cells[i].Erad);
                throw eo;
            }
        }

        for(size_t j = 0; j < particlesPerCell; j++)
        {
            Particle p = this->generateSingleParticle(i, this->cells[i]);
            if(this->withCompton && this->multigroupOpacity)
            {
                double const r = this->dist(this->re) * initialGroupEnergyCdf.back();
                p.frequency = LinearInterpolation(initialGroupEnergyCdf, ComputationalCell3D::energyBoundaries, r);
            }
            else if(this->multigroupOpacity)
            {
                double rnd = this->dist(this->re);
                double r = rnd * cumulPlanck.back();
                p.frequency = LinearInterpolation(cumulPlanck, ComputationalCell3D::energyBoundaries, r);
            }
            p.cellID = this->cells[i].ID;
            p.sourceCellID = this->cells[i].ID;
            p.weight = weightPerPhoton;
            SetInitialWeightFromWeight(p);
            result.push_back(p);
        }
    }
    return result;
}

typename RadiationIMC::Particle RadiationIMC::generateSingleParticle(size_t cellIndex, const ComputationalCell3D &cell) const
{
    Particle particle;
    particle.id = std::numeric_limits<size_t>::max();
    particle.frequency = 0; // TODO
    particle.location = RandomPointInCell(this->grid, cellIndex);
    // particle.location = particle.location * (1 - MONTECARLO_EPS) + MONTECARLO_EPS * this->grid.GetMeshPoint(cellIndex);
    particle.timeLeft = 0;
    assert(this->grid.IsPointInCell(particle.location, cellIndex));
    assert(not this->grid.IsPointOutsideBox(particle.location));
    particle.velocity = this->opacity->getRandomVelocity(cell);
    if(this->useTransportVelocities_ && !this->MMC)
    {
        ComovingToLabPacket(particle, cell.velocity);
    }
    particle.cellIndex = cellIndex;
    // nudge a little bit towards the cell's point
    static constexpr double nudge = 1e-10;
    particle.location = particle.location * (1 - nudge) + nudge * this->grid.GetMeshPoint(cellIndex);
    return particle;
}

void RadiationIMC::reconcileComptonParticles(std::vector<Particle> &particles)
{
    if(!this->withCompton || !this->multigroupOpacity)
        return;

    size_t const Ncells = this->grid.GetPointNo();
    std::vector<GroupArray> rawGroupEnergy(Ncells, GroupArray{});
    for(const Particle &particle : particles)
    {
        if(particle.cellIndex >= Ncells)
            continue;
        if(particle.weight < 0.0)
        {
            UniversalError eo("Negative particle weight in positive-only Compton reconciliation");
            eo.addEntry("Cell index", particle.cellIndex);
            eo.addEntry("Particle weight", particle.weight);
            throw eo;
        }
        double frequency = particle.frequency;
        ClampFrequencyToBounds(frequency);
        size_t const g = this->opacity->findGroup(frequency);
        rawGroupEnergy[particle.cellIndex][g] += particle.weight;
    }

    std::vector<GroupArray> scale(Ncells, GroupArray{});
    for(size_t i = 0; i < Ncells; i++)
    {
        for(size_t g = 0; g < ENERGY_GROUPS_NUM; g++)
        {
            double const target = std::max(0.0, this->conserved[i].Eg[g]);
            double const raw = rawGroupEnergy[i][g];
            scale[i][g] = 1.0;
            if(raw > 0.0 && target < raw)
                scale[i][g] = target / raw;
        }
    }

    auto it = particles.begin();
    while(it != particles.end())
    {
        Particle &particle = *it;
        if(particle.cellIndex < Ncells)
        {
            double frequency = particle.frequency;
            ClampFrequencyToBounds(frequency);
            size_t const g = this->opacity->findGroup(frequency);
            particle.weight *= scale[particle.cellIndex][g];
            SetInitialWeightFromWeight(particle);
        }
        if(particle.weight <= 0.0)
            it = particles.erase(it);
        else
            ++it;
    }

    for(size_t i = 0; i < Ncells; i++)
    {
        ComputationalCell3D const &cell = this->cells[i];
        GroupArray deficits{};
        GroupArray fractional{};
        std::array<size_t, ENERGY_GROUPS_NUM> groupCounts{};
        double totalDeficit = 0.0;
        size_t nonzeroDeficitGroups = 0;
        for(size_t g = 0; g < ENERGY_GROUPS_NUM; g++)
        {
            double const target = std::max(0.0, this->conserved[i].Eg[g]);
            double const represented = (rawGroupEnergy[i][g] > target) ? target : rawGroupEnergy[i][g];
            double const deficit = target - represented;
            double const tolerance = 1e-12 * std::max(target, 1.0);
            if(deficit <= tolerance)
                continue;

            deficits[g] = deficit;
            totalDeficit += deficit;
            ++nonzeroDeficitGroups;
        }
        if(totalDeficit <= 0.0)
            continue;

        size_t const correctionBudget = std::max(this->newPhotonsPerCell, nonzeroDeficitGroups);
        size_t allocated = 0;
        for(size_t g = 0; g < ENERGY_GROUPS_NUM; g++)
        {
            if(deficits[g] <= 0.0)
                continue;
            groupCounts[g] = 1;
            ++allocated;
        }

        size_t const remainingBudget = correctionBudget - allocated;
        size_t extraAllocated = 0;
        for(size_t g = 0; g < ENERGY_GROUPS_NUM; g++)
        {
            if(deficits[g] <= 0.0)
                continue;
            double const exactExtra = static_cast<double>(remainingBudget) * deficits[g] / totalDeficit;
            size_t const extra = static_cast<size_t>(std::floor(exactExtra));
            groupCounts[g] += extra;
            fractional[g] = exactExtra - static_cast<double>(extra);
            extraAllocated += extra;
        }
        while(extraAllocated < remainingBudget)
        {
            size_t bestGroup = ENERGY_GROUPS_NUM;
            double bestFraction = -1.0;
            for(size_t g = 0; g < ENERGY_GROUPS_NUM; g++)
            {
                if(deficits[g] > 0.0 && fractional[g] > bestFraction)
                {
                    bestGroup = g;
                    bestFraction = fractional[g];
                }
            }
            if(bestGroup == ENERGY_GROUPS_NUM)
                break;
            ++groupCounts[bestGroup];
            fractional[bestGroup] = 0.0;
            ++extraAllocated;
        }

        for(size_t g = 0; g < ENERGY_GROUPS_NUM; g++)
        {
            size_t const ng = groupCounts[g];
            if(ng == 0)
                continue;
            double const packetWeight = deficits[g] / static_cast<double>(ng);
            for(size_t j = 0; j < ng; j++)
            {
                Particle particle = this->generateSingleParticle(i, cell);
                particle.cellID = cell.ID;
                particle.sourceCellID = cell.ID;
                particle.frequency = this->frequencyForComptonGroup(g);
                particle.weight = packetWeight;
                SetInitialWeightFromWeight(particle);
                particles.push_back(particle);
            }
        }
    }
}

void RadiationIMC::splitComptonRiskyParticles(std::vector<Particle> &particles, double fullDt)
{
    this->comptonDataReusableInPreStep_ = false;
    if(postProcess_.enabled || !this->withCompton || !this->multigroupOpacity)
        return;

    size_t const Ncells = this->grid.GetPointNo();
    this->factorFleck.assign(Ncells, 1.0);
    this->planckOpacities.assign(Ncells, 0.0);
    this->multigroupOpacity->ResetCummulativeOpacityCellID();
    this->precomputeComptonData(fullDt);
    this->comptonDataReusableInPreStep_ = true;

    size_t const binCount = Ncells * ENERGY_GROUPS_NUM;
    std::vector<std::vector<size_t>> bins(binCount);
    for(size_t p = 0; p < particles.size(); p++)
    {
        Particle &particle = particles[p];
        if(particle.cellIndex >= Ncells || particle.weight <= 0.0)
            continue;
        double frequency = particle.frequency;
        ClampFrequencyToBounds(frequency);
        size_t const g = this->opacity->findGroup(frequency);
        if(this->comptonData[particle.cellIndex].riskTargetPackets[g] == 0)
            continue;
        bins[particle.cellIndex * ENERGY_GROUPS_NUM + g].push_back(p);
    }

    size_t const maxLocalExtra = std::max<size_t>(1, particles.size() / 10);
    size_t constexpr maxExtraPerCell = 200;
    std::vector<size_t> extraPerCell(Ncells, 0);
    size_t localExtra = 0;
    size_t localRiskBinsSplit = 0;
    particles.reserve(particles.size() + maxLocalExtra);

    for(size_t i = 0; i < Ncells && localExtra < maxLocalExtra; i++)
    {
        std::array<size_t, ENERGY_GROUPS_NUM> riskOrder{};
        for(size_t g = 0; g < ENERGY_GROUPS_NUM; g++)
            riskOrder[g] = g;
        std::sort(riskOrder.begin(), riskOrder.end(), [&](size_t a, size_t b)
        {
            return this->comptonData[i].riskScore[a] > this->comptonData[i].riskScore[b];
        });
        for(size_t orderIndex = 0; orderIndex < ENERGY_GROUPS_NUM && localExtra < maxLocalExtra; orderIndex++)
        {
            size_t const g = riskOrder[orderIndex];
            size_t const target = this->comptonData[i].riskTargetPackets[g];
            if(target == 0)
                continue;
            std::vector<size_t> const &bin = bins[i * ENERGY_GROUPS_NUM + g];
            size_t const count = bin.size();
            if(count == 0 || count >= target)
                continue;

            size_t allowed = target - count;
            allowed = std::min(allowed, maxLocalExtra - localExtra);
            allowed = std::min(allowed, maxExtraPerCell - std::min(extraPerCell[i], maxExtraPerCell));
            if(allowed == 0)
                continue;

            std::vector<size_t> copiesPerOriginal(count, 0);
            for(size_t k = 0; k < allowed; k++)
                ++copiesPerOriginal[k % count];

            for(size_t j = 0; j < count; j++)
            {
                size_t const copies = copiesPerOriginal[j];
                if(copies == 0)
                    continue;
                size_t const particleIndex = bin[j];
                size_t const pieces = copies + 1;
                double const splitWeight = particles[particleIndex].weight / static_cast<double>(pieces);
                particles[particleIndex].weight = splitWeight;
                SetInitialWeightFromWeight(particles[particleIndex]);
                for(size_t k = 0; k < copies; k++)
                {
                    Particle particleCopy = particles[particleIndex];
                    particleCopy.weight = splitWeight;
                    SetInitialWeightFromWeight(particleCopy);
                    particleCopy.id = std::numeric_limits<size_t>::max();
                    #ifdef RICH_MPI
                    particleCopy.rank = std::numeric_limits<rank_t>::max();
                    #endif
                    particleCopy.steps = 0;
                    particles.push_back(particleCopy);
                }
            }

            localExtra += allowed;
            extraPerCell[i] += allowed;
            ++localRiskBinsSplit;
        }
    }

    if(this->comptonDiagnostics)
    {
        unsigned long long globalExtra = static_cast<unsigned long long>(localExtra);
        unsigned long long globalRiskBinsSplit = static_cast<unsigned long long>(localRiskBinsSplit);
        int rank = 0;
        #ifdef RICH_MPI
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Allreduce(MPI_IN_PLACE, &globalExtra, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, &globalRiskBinsSplit, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
        #endif
        if(rank == 0 && globalExtra > 0)
        {
            std::cout << "Compton risk splitter added " << globalExtra
                      << " particles across " << globalRiskBinsSplit
                      << " risky cell-groups" << std::endl;
        }
    }
}

void RadiationIMC::adjustExistingParticles(std::vector<Particle> &particles, double fullDt)
{
    this->splitComptonRiskyParticles(particles, fullDt);

    if(!this->MMC)
    {
        return;
    }

    size_t Ncells = this->grid.GetPointNo();
    std::vector<double> divV(Ncells, 0);

    std::vector<size_t> neigh;
    for(size_t i = 0; i < Ncells; i++)
    {
        this->grid.GetNeighbors(i, neigh);
        const auto &faces = this->grid.GetCellFaces(i);
        Vector3D r_i = this->grid.GetMeshPoint(i);
        for(size_t j = 0; j < neigh.size(); j++)
        {
            size_t neighbor_j = neigh[j];
            auto r_ij = normalize(r_i - this->grid.GetMeshPoint(neighbor_j));
            double A_ij = this->grid.GetArea(faces[j]);
            Vector3D v_j = (neighbor_j >= Ncells && this->grid.IsPointOutsideBox(neighbor_j))
                           ? this->cells[i].velocity
                           : this->cells[neighbor_j].velocity;
            divV[i] -= 0.5 * ScalarProd(this->cells[i].velocity + v_j, r_ij) * A_ij;
        }
        divV[i] /= this->grid.GetVolume(i);
    }

    const auto [ll, ur] = this->grid.GetBoxCoordinates();

    auto it = particles.begin();
    while(it != particles.end())
    {
        Particle &p = *it;
        size_t ci = p.cellIndex;
        p.location += this->cells[ci].velocity * fullDt;
        p.weight += -p.weight * fullDt * divV[ci] / 3.0;

        if(this->grid.IsPointOutsideBox(p.location))
        {
            p.location.x = std::max(ll.x, std::min(ur.x, p.location.x));
            p.location.y = std::max(ll.y, std::min(ur.y, p.location.y));
            p.location.z = std::max(ll.z, std::min(ur.z, p.location.z));
            MonteCarloParticleStatus status = this->boundary->apply(p);
            if(status == MonteCarloParticleStatus::REMOVE)
            {
                it = particles.erase(it);
                continue;
            }
        }
        ++it;
    }

    UpdateNewCells(this->grid, particles, this->cells);
}

std::ostream &operator<<(std::ostream &os, const RadiationIMCParameters &parameters)
{
    os << "IMC, with parameters:" << std::endl;
    os << "\t" << "new photons per cell: " << parameters.newPhotonsPerCell << std::endl;
    os << "\t" << "with hydro: " << parameters.withHydro << std::endl;
    os << "\t" << "diffusion pressure gradient: " << parameters.diffusionPressureGradient << std::endl;
    os << "\t" << "MMC: " << parameters.MMC << std::endl;
    os << "\t" << "with multigroup opacity: " << parameters.withMultigroupOpacity << std::endl;
    os << "\t" << "with random walk: " << parameters.withRandomWalk << std::endl;
    os << "\t" << "with DDMC: " << parameters.withDDMC << std::endl;
    os << "\t" << "with Compton: " << parameters.withCompton << std::endl;
    if(parameters.withCompton)
    {
        os << "\t" << "Compton induced terms: " << parameters.comptonUseInduced << std::endl;
        os << "\t" << "Compton n=0 fallback: " << parameters.comptonAllowNZeroFallback << std::endl;
        os << "\t" << "Compton debug parity check: " << parameters.comptonDebugParityCheck << std::endl;
        os << "\t" << "Compton transport mode: implicit in-place signed scattering" << std::endl;
        os << "\t" << "Compton diagnostics: " << parameters.comptonDiagnostics << std::endl;
        os << "\t" << "Compton matrix samples: " << parameters.comptonMatrixSamples << std::endl;
    }
    if(parameters.withRandomWalk)
    {
        os << "\t" << "RW min cell optical depth: " << parameters.rwMinCellOpticalDepth << std::endl;
        os << "\t" << "RW min particle optical depth: " << parameters.rwMinParticleOpticalDepth << std::endl;
    }
    if(parameters.withDDMC)
    {
        os << "\t" << "DDMC min cell optical depth: " << parameters.ddmcMinCellOpticalDepth << std::endl;
        os << "\t" << "DDMC multigroup PGRW: " << parameters.ddmcUseMultigroupPGRW << std::endl;
        os << "\t" << "DDMC maximum group cutoff: " << parameters.ddmcMaxGroupCutoff << std::endl;
        os << "\t" << "DDMC interface diagnostics: " << parameters.ddmcInterfaceDiagnostics << std::endl;
    }
    os << "\t" << "no hydro feedback: " << parameters.noHydroFeedback << std::endl;
    if(parameters.postProcess.enabled)
    {
        os << "\t" << "post-process: enabled" << std::endl;
        os << "\t" << "post-process use cell velocities: " << parameters.postProcess.useCellVelocities << std::endl;
        os << "\t" << "post-process polarization: " << parameters.postProcess.polarization.enabled << std::endl;
        os << "\t" << "post-process polarization manual scatterings: "
           << parameters.postProcess.polarization.manualScatteringsAfterAcceleration << std::endl;
        os << "\t" << "post-process polarization depolarization scatterings: "
           << parameters.postProcess.polarization.depolarizationScatterings << std::endl;
        os << "\t" << "post-process polarization accelerated closure: "
           << parameters.postProcess.polarization.acceleratedClosure << std::endl;
    }
    return os;
}
