#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <limits>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "3D/tessellation/Voronoi3D.hpp"
#include "newtonian/common/ideal_gas.hpp"
#include "newtonian/three_dimensional/Ghost3D.hpp"
#include "newtonian/three_dimensional/LinearGauss3D.hpp"
#include "newtonian/three_dimensional/ManualTimeStep.hpp"
#include "newtonian/three_dimensional/computational_cell.hpp"
#include "newtonian/three_dimensional/default_cell_updater.hpp"
#include "newtonian/three_dimensional/default_extensive_updater.hpp"
#include "newtonian/three_dimensional/simulation/ActiveMeshView.hpp"
#include "newtonian/three_dimensional/simulation/IndividualTimeStep.hpp"
#include "newtonian/three_dimensional/simulation/Simulation.hpp"
#include "newtonian/three_dimensional/simulation/steps/RadiationStep.hpp"
#include "newtonian/three_dimensional/simulation/steps/RadiationMCStep.hpp"
#include "newtonian/three_dimensional/conserved_3d.hpp"
#include "Radiation/Diffusion.hpp"
#include "Radiation/RadiationDriverTestHooks.hpp"
#include "Radiation/conj_grad_solve.hpp"
#include "Radiation/SpectralPositivity.hpp"
#include "CMMC/src/compton_matrix_mc.hpp"
#ifdef RICH_MPI
#include <mpi.h>
#endif

namespace {

static_assert(!RadiationMCStep::individual_time_steps_supported,
              "Monte Carlo radiation must remain incompatible with individual timesteps");

class MonteCarloStepStub final : public PhysicsStep
{
public:
    void step(double) override {}
    double suggestTimeStep(void) const override {return 1;}
    bool supportsIndividualTimeSteps(void) const override
    {return RadiationMCStep::individual_time_steps_supported;}
    std::string individualTimeStepUnsupportedReason(void) const override
    {return RadiationMCStep::individual_time_step_error;}
    std::string getName(void) const override {return RadiationMCStep::step_name;}

#ifdef RICH_MPI
    bool allowRebalance(void) override {return false;}
    std::string getRequiredLB(void) const override {return std::string();}
    std::vector<double> getLoadBalanceWeights(void) override
    {return std::vector<double>();}
#endif
};

class RetryingRadiationDriver final : public RadiationDriver
{
public:
    explicit RetryingRadiationDriver(
        EquationOfState const& eos,
        bool const release_restriction_after_acceptance = true) :
        RadiationDriver(eos, std::vector<std::string>(), false, false, false),
        release_restriction_after_acceptance_(
            release_restriction_after_acceptance)
    {}

    bool prestep(Tessellation3D const&,
                 std::vector<ComputationalCell3D> const&) const override
    {return true;}

    bool step(double, int& total_iters, Tessellation3D const&,
              std::vector<ComputationalCell3D>&,
              std::vector<Conserved3D>&, double dt, double time) const override
    {
        clearStepFailure();
        ++calls;
        total_iters = 0;
        time_consistent = time_consistent &&
            std::abs(time - accepted_time) <= 2e-15;
        if(!restriction_released && dt > accepted_dt)
        {
            ++rejections;
            setStepFailure("forced retry below the former fraction cutoff", 7);
            return false;
        }
        ++acceptances;
        accepted_time += dt;
        if(release_restriction_after_acceptance_)
            restriction_released = true;
        return true;
    }

    bool poststep() const override {return true;}

    double calculate_dt(double dt, Tessellation3D&,
                        std::vector<ComputationalCell3D>&) const override
    {return dt;}

    void BuildMatrix(Tessellation3D const&, CG::mat&, CG::size_t_mat&,
                     std::vector<ComputationalCell3D> const&, double,
                     std::vector<double>&, std::vector<double>&,
                     double) const override
    {}

    void PostCG(Tessellation3D const&, std::vector<Conserved3D>&, double,
                std::vector<ComputationalCell3D>&,
                std::vector<double> const&,
                std::vector<double> const&) const override
    {}

    static constexpr double accepted_dt = 1.0 / 2048.0;
    mutable std::size_t calls = 0;
    mutable std::size_t rejections = 0;
    mutable std::size_t acceptances = 0;
    mutable double accepted_time = 0;
    mutable bool restriction_released = false;
    mutable bool time_consistent = true;

private:
    bool const release_restriction_after_acceptance_;
};

class RetryingIndividualRadiationDriver final : public RadiationDriver
{
public:
    RetryingIndividualRadiationDriver(EquationOfState const& eos,
                                      bool cell_local_failure,
                                      bool attributed_collective_failure) :
        RadiationDriver(eos, std::vector<std::string>(), false, false, false),
        cell_local_failure_(cell_local_failure),
        attributed_collective_failure_(attributed_collective_failure)
    {}

    bool prestep(Tessellation3D const&,
                 std::vector<ComputationalCell3D> const&) const override
    {return true;}

    bool step(double, int&, Tessellation3D const&,
              std::vector<ComputationalCell3D>&,
              std::vector<Conserved3D>&, double, double) const override
    {return true;}

    bool poststep() const override {return true;}

    double calculate_dt(double dt, Tessellation3D&,
                        std::vector<ComputationalCell3D>&) const override
    {return dt;}

    void BuildMatrix(Tessellation3D const&, CG::mat&, CG::size_t_mat&,
                     std::vector<ComputationalCell3D> const&, double,
                     std::vector<double>&, std::vector<double>&,
                     double) const override
    {}

    void PostCG(Tessellation3D const&, std::vector<Conserved3D>&, double,
                std::vector<ComputationalCell3D>&,
                std::vector<double> const&,
                std::vector<double> const&) const override
    {}

    bool supportsIndividualTimeSteps() const override {return true;}

    bool stepIndividual(
        double, int& total_iters, Tessellation3D const&,
        std::vector<ComputationalCell3D>& cells,
        std::vector<Conserved3D>&, IndividualStepContext const&,
        double interval_fraction, double,
        std::vector<ComputationalCell3D> const*,
        std::vector<Conserved3D>*,
        std::vector<std::size_t> const*) const override
    {
        clearStepFailure();
        total_iters = 0;
        if(interval_fraction > 0.5) {
            ++rejections;
            if(cell_local_failure_)
                setCellLocalStepFailure(
                    "forced individual retry", cells.at(0).ID);
            else
                setStepFailure("forced individual retry",
                    attributed_collective_failure_ ?
                    cells.at(0).ID :
                    std::numeric_limits<std::size_t>::max());
            return false;
        }
        ++acceptances;
        return true;
    }

    void calculateIndividualTimeSteps(
        IndividualStepContext const& context, Tessellation3D&,
        std::vector<ComputationalCell3D>&,
        std::vector<double>& time_step_limits,
        std::vector<ComputationalCell3D> const*,
        std::vector<std::size_t> const*) const override
    {
        for(std::size_t cell : context.active_indices)
            time_step_limits.at(cell) = 8 * context.cellTimeStep(cell);
    }

    mutable std::size_t rejections = 0;
    mutable std::size_t acceptances = 0;

private:
    bool cell_local_failure_;
    bool attributed_collective_failure_;
};

class RetryOnceGreyDiffusion final : public Diffusion
{
public:
    RetryOnceGreyDiffusion(OpacityCalculator const& opacity,
                           DiffusionBoundaryCalculator const& boundary,
                           EquationOfState const& eos,
                           bool reject_first_candidate) :
        Diffusion(opacity, boundary, eos, std::vector<std::string>(),
                  false, true, false, false),
        reject_next_(reject_first_candidate)
    {}

    mutable std::size_t candidate_preparations = 0;
    mutable std::size_t rejected_candidates = 0;
    mutable std::size_t accepted_intervals = 0;

    void PostCG(Tessellation3D const& tess,
                std::vector<Conserved3D>& extensives,
                double dt,
                std::vector<ComputationalCell3D>& cells,
                std::vector<double> const& result,
                std::vector<double> const& full_result) const override
    {
        Diffusion::PostCG(tess, extensives, dt, cells, result, full_result);
        ++accepted_intervals;
    }

private:
    void prepareIndividualCandidate(
        Tessellation3D const& tess,
        std::vector<ComputationalCell3D> const& cells) const override
    {
        Diffusion::prepareIndividualCandidate(tess, cells);
        ++candidate_preparations;
    }

    bool validateIndividualCoefficients(
        IndividualStepContext const& context,
        std::vector<ComputationalCell3D> const& cells) const override
    {
        if(reject_next_)
        {
            reject_next_ = false;
            ++rejected_candidates;
            std::size_t const cell_id = context.active_indices.empty() ?
                std::numeric_limits<std::size_t>::max() :
                cells.at(context.active_indices.front()).ID;
            setStepFailure("forced grey candidate retry", cell_id);
            return false;
        }
        return true;
    }

    mutable bool reject_next_;
};

class OrderedSpectralRepairDriver final : public RadiationDriver
{
public:
    explicit OrderedSpectralRepairDriver(EquationOfState const& eos) :
        RadiationDriver(eos, std::vector<std::string>(), false, false, false)
    {}

    bool prestep(Tessellation3D const&,
                 std::vector<ComputationalCell3D> const&) const override
    {return true;}
    bool step(double, int&, Tessellation3D const&,
              std::vector<ComputationalCell3D>&,
              std::vector<Conserved3D>&, double, double) const override
    {return true;}
    bool poststep() const override {return true;}
    double calculate_dt(double dt, Tessellation3D&,
                        std::vector<ComputationalCell3D>&) const override
    {return dt;}
    bool supportsIndividualTimeSteps() const override {return true;}
    std::size_t individualUnknownsPerCell() const override {return 32;}

    void BuildMatrix(Tessellation3D const& tess,
                     CG::mat& matrix,
                     CG::size_t_mat& columns,
                     std::vector<ComputationalCell3D> const&,
                     double,
                     std::vector<double>& rhs,
                     std::vector<double>& initial,
                     double) const override
    {
        std::size_t const block_size = individualUnknownsPerCell();
        std::size_t const rows = block_size * tess.GetPointNo();
        matrix.assign(rows, CG::mat::value_type());
        columns.assign(rows, CG::size_t_mat::value_type());
        rhs.resize(rows);
        initial.assign(rows, 0);
        for(std::size_t row = 0; row < rows; ++row) {
            std::size_t const local_row = row % block_size;
            matrix[row].push_back(2.001);
            columns[row].push_back(row);
            if(local_row > 0) {
                matrix[row].push_back(-1);
                columns[row].push_back(row - 1);
            }
            if(local_row + 1 < block_size) {
                matrix[row].push_back(-1);
                columns[row].push_back(row + 1);
            }
            rhs[row] = 1 + 0.01 * static_cast<double>(local_row + 1);
        }
    }

    void PostCG(Tessellation3D const&,
                std::vector<Conserved3D>& extensives,
                double,
                std::vector<ComputationalCell3D>& cells,
                std::vector<double> const&,
                std::vector<double> const&) const override
    {
        ++absorption_diffusion_commits;
        double constexpr positive_extent = 2.5792383680236759e29;
        double constexpr negative_extent = 6.3799634091935332e27;
        extensives.at(0).Eg[0] = positive_extent;
        extensives.at(0).Eg[1] = -negative_extent;
        for(std::size_t group = 2;
            group < extensives.at(0).Eg.size(); ++group)
            extensives.at(0).Eg[group] = 0;
        extensives.at(0).Erad = positive_extent - negative_extent;
        cells.at(0).Eg[0] = extensives.at(0).Eg[0] / extensives.at(0).mass;
        cells.at(0).Eg[1] = extensives.at(0).Eg[1] / extensives.at(0).mass;
        cells.at(0).Erad = extensives.at(0).Erad / extensives.at(0).mass;
    }

    mutable std::size_t absorption_diffusion_commits = 0;
    mutable std::size_t post_solve_calls = 0;
    mutable std::size_t compton_substep_validations = 0;
    mutable std::size_t dormant_global_storage_releases = 0;
    mutable bool absorption_repaired_before_compton = false;
    mutable bool aggregate_consistent_before_compton = false;
    mutable bool every_compton_substep_repaired = true;

protected:
    void ReleaseDormantGlobalSolverStorage() const override
    {
        ++dormant_global_storage_releases;
    }

    bool applyIndividualPostSolvePhysics(
        Tessellation3D const&,
        std::vector<ComputationalCell3D>&,
        std::vector<Conserved3D>& extensives,
        double,
        double const global_maximum_cell_radiation_extent) const override
    {
        ++post_solve_calls;
        absorption_repaired_before_compton =
            std::all_of(extensives.at(0).Eg.begin(),
                        extensives.at(0).Eg.end(),
                        [](double const extent) {return extent >= 0;});
        double const group_sum = std::accumulate(
            extensives.at(0).Eg.begin(), extensives.at(0).Eg.end(), 0.0);
        aggregate_consistent_before_compton =
            extensives.at(0).Erad == group_sum;

        for(std::size_t substep = 0; substep < 2; ++substep) {
            std::vector<double> candidate{
                2.5792383680236759e29,
                -6.3799634091935332e27};
            double candidate_total =
                std::accumulate(candidate.begin(), candidate.end(), 0.0);
            auto const controlled =
                RadiationPositivity::RepairControlledNegativeGroupExtents(
                    candidate, candidate_total,
                    RadiationPositivity::spectral_repair_relative_limit,
                    global_maximum_cell_radiation_extent);
            every_compton_substep_repaired =
                every_compton_substep_repaired &&
                controlled.repair.valid && controlled.repair.repaired &&
                controlled.used_global_negative_exception &&
                candidate_total ==
                    std::accumulate(candidate.begin(), candidate.end(), 0.0);
            ++compton_substep_validations;
        }
        return absorption_repaired_before_compton &&
            aggregate_consistent_before_compton &&
            every_compton_substep_repaired &&
            absorption_diffusion_commits == 1;
    }
};

void require(bool condition, std::string const& message)
{
    if(!condition)
        throw std::runtime_error(message);
}

bool close(double left, double right, double relative = 2e-8)
{
    return std::abs(left - right) <=
        relative * std::max({1.0, std::abs(left), std::abs(right)});
}

bool close(Vector3D const& left, Vector3D const& right, double relative = 2e-8)
{
    return close(left.x, right.x, relative) &&
           close(left.y, right.y, relative) &&
           close(left.z, right.z, relative);
}

class ScopedEnvironmentVariable
{
public:
    explicit ScopedEnvironmentVariable(char const* name) :
        name_(name), had_value_(std::getenv(name) != nullptr),
        saved_value_(had_value_ ? std::getenv(name) : "")
    {}

    ~ScopedEnvironmentVariable()
    {
        if(had_value_)
            ::setenv(name_.c_str(), saved_value_.c_str(), 1);
        else
            ::unsetenv(name_.c_str());
    }

    void set(char const* value)
    {
        int const status = value == nullptr ?
            ::unsetenv(name_.c_str()) :
            ::setenv(name_.c_str(), value, 1);
        if(status != 0)
            throw std::runtime_error(
                "Could not set individual cell-update test environment");
    }

private:
    std::string name_;
    bool had_value_;
    std::string saved_value_;
};

bool sameCellState(ComputationalCell3D const& left,
                   ComputationalCell3D const& right)
{
    return left.density == right.density &&
           left.pressure == right.pressure &&
           left.internal_energy == right.internal_energy &&
           left.temperature == right.temperature &&
           left.ID == right.ID &&
           left.velocity.x == right.velocity.x &&
           left.velocity.y == right.velocity.y &&
           left.velocity.z == right.velocity.z &&
           left.dt == right.dt &&
           left.Erad == right.Erad &&
           left.Eg == right.Eg &&
           left.Erad_dt == right.Erad_dt &&
           left.Erad_dt_dt == right.Erad_dt_dt &&
           left.cs == right.cs &&
           left.tracers == right.tracers &&
           left.stickers == right.stickers;
}

bool sameConservedState(Conserved3D const& left, Conserved3D const& right)
{
    return left.mass == right.mass &&
           left.momentum.x == right.momentum.x &&
           left.momentum.y == right.momentum.y &&
           left.momentum.z == right.momentum.z &&
           left.energy == right.energy &&
           left.internal_energy == right.internal_energy &&
           left.Erad == right.Erad &&
           left.Eg == right.Eg &&
           left.Erad_dt == right.Erad_dt &&
           left.Erad_dt_dt == right.Erad_dt_dt &&
           left.tracers == right.tracers;
}

void requireCellUpdateStateEqual(
    std::vector<ComputationalCell3D> const& expected_cells,
    std::vector<Conserved3D> const& expected_extensives,
    std::vector<ComputationalCell3D> const& candidate_cells,
    std::vector<Conserved3D> const& candidate_extensives,
    std::string const& message)
{
    require(expected_cells.size() == candidate_cells.size() &&
            expected_extensives.size() == candidate_extensives.size(),
            message + " size mismatch");
    for(std::size_t index = 0; index < expected_cells.size(); ++index)
        require(sameCellState(expected_cells[index], candidate_cells[index]),
                message + " primitive mismatch");
    for(std::size_t index = 0; index < expected_extensives.size(); ++index)
        require(sameConservedState(
                    expected_extensives[index], candidate_extensives[index]),
                message + " extensive mismatch");
}

void compareDerivative(ComputationalCell3D const& expected,
                       ComputationalCell3D const& candidate)
{
    double const tolerance = 2e-7;
    require(close(expected.density, candidate.density, tolerance),
            "full/partial density gradient mismatch");
    require(close(expected.pressure, candidate.pressure, tolerance),
            "full/partial pressure gradient mismatch");
    require(close(expected.internal_energy, candidate.internal_energy, tolerance),
            "full/partial energy gradient mismatch");
    require(close(expected.velocity, candidate.velocity, tolerance),
            "full/partial velocity gradient mismatch");
}

void compareSlope(Slope3D const& expected, Slope3D const& candidate)
{
    compareDerivative(expected.xderivative, candidate.xderivative);
    compareDerivative(expected.yderivative, candidate.yderivative);
    compareDerivative(expected.zderivative, candidate.zderivative);
}

std::size_t mappedNeighbor(Tessellation3D const& tess,
                           std::size_t local_neighbor,
                           std::size_t point_count)
{
    Tessellation3D::AllPointsMap const& map = tess.GetIndicesInAllPoints();
    auto const found = map.find(local_neighbor);
    if(found == map.end() || found->second >= point_count)
        return std::numeric_limits<std::size_t>::max();
    return found->second;
}

std::vector<std::size_t> reconstructionClosure(
    Tessellation3D const& full,
    std::vector<std::size_t> const& active,
    std::size_t point_count)
{
    ActiveMeshView const view(full, point_count);
    std::vector<unsigned char> included(point_count, 0);
    std::vector<std::size_t> target;
    std::vector<std::size_t> frontier;
    for(std::size_t global : active)
    {
        if(included.at(global) == 0)
        {
            included[global] = 1;
            target.push_back(global);
            frontier.push_back(global);
        }
    }

    // Active-face reconstruction needs slopes on both face endpoints.  The
    // second layer supplies complete centroid stencils for passive endpoints.
    for(std::size_t depth = 0; depth < 2; ++depth)
    {
        std::vector<std::size_t> next;
        std::vector<std::size_t> neighbors;
        for(std::size_t global : frontier)
        {
            full.GetNeighbors(view.globalToLocal(global), neighbors);
            for(std::size_t local_neighbor : neighbors)
            {
                std::size_t const neighbor =
                    mappedNeighbor(full, local_neighbor, point_count);
                if(neighbor < point_count && included[neighbor] == 0)
                {
                    included[neighbor] = 1;
                    target.push_back(neighbor);
                    next.push_back(neighbor);
                }
            }
        }
        frontier.swap(next);
    }
    return target;
}

struct FaceGeometry
{
    std::size_t neighbor;
    double area;
    Vector3D centroid;
    Vector3D normal;
};

std::vector<FaceGeometry> cellFaces(Tessellation3D const& tess,
                                    std::size_t local,
                                    std::size_t point_count)
{
    std::vector<std::size_t> neighbors;
    tess.GetNeighbors(local, neighbors);
    face_vec const& faces = tess.GetCellFaces(local);
    require(neighbors.size() == faces.size(), "face/neighbor size mismatch");
    std::vector<FaceGeometry> result;
    result.reserve(faces.size());
    for(std::size_t i = 0; i < faces.size(); ++i)
    {
        Vector3D normal = tess.Normal(faces[i]);
        double const normal_size = abs(normal);
        require(normal_size > 0, "zero face normal");
        normal *= 1.0 / normal_size;
        if(neighbors[i] < tess.getMeshPoints().size())
        {
            Vector3D const outward = tess.GetMeshPoint(neighbors[i]) -
                                     tess.GetMeshPoint(local);
            if(ScalarProd(normal, outward) < 0)
                normal *= -1;
        }
        result.push_back({mappedNeighbor(tess, neighbors[i], point_count),
                          tess.GetArea(faces[i]), tess.FaceCM(faces[i]), normal});
    }
    return result;
}

double fluxProxy(Tessellation3D const& tess,
                 std::size_t local,
                 std::size_t global,
                 std::size_t point_count)
{
    std::vector<std::size_t> neighbors;
    tess.GetNeighbors(local, neighbors);
    face_vec const& faces = tess.GetCellFaces(local);
    double result = 0;
    for(std::size_t i = 0; i < neighbors.size(); ++i)
    {
        std::size_t const neighbor_global =
            mappedNeighbor(tess, neighbors[i], point_count);
        if(neighbor_global == std::numeric_limits<std::size_t>::max())
            continue;
        double const distance = abs(tess.GetMeshPoint(local) -
                                    tess.GetMeshPoint(neighbors[i]));
        double const left = 1.0 + 0.013 * static_cast<double>(global);
        double const right = 1.0 + 0.013 * static_cast<double>(neighbor_global);
        result += tess.GetArea(faces[i]) * (right - left) / distance;
    }
    return result;
}

void comparePartial(std::vector<Vector3D> const& points,
                    std::vector<std::size_t> const& active)
{
    Voronoi3D full(Vector3D(0, 0, 0), Vector3D(1, 1, 1));
    Voronoi3D partial(Vector3D(0, 0, 0), Vector3D(1, 1, 1));
    full.Build(points);
    std::vector<std::size_t> const target =
        reconstructionClosure(full, active, points.size());
    partial.BuildPartially(points, target);
    ActiveMeshView const view(partial, points.size());

    for(std::size_t global : active)
    {
        require(view.containsGlobal(global), "active point missing from partial map");
        std::size_t const local = view.globalToLocal(global);
        require(close(full.GetVolume(global), partial.GetVolume(local)),
                "full/partial volume mismatch");
        require(close(full.GetCellCM(global), partial.GetCellCM(local)),
                "full/partial cell centroid mismatch");

        std::vector<FaceGeometry> reference = cellFaces(full, global, points.size());
        std::vector<FaceGeometry> candidate = cellFaces(partial, local, points.size());
        require(reference.size() == candidate.size(), "full/partial face count mismatch");
        std::vector<unsigned char> used(candidate.size(), 0);
        for(FaceGeometry const& expected : reference)
        {
            std::size_t best = candidate.size();
            double best_distance = std::numeric_limits<double>::max();
            for(std::size_t i = 0; i < candidate.size(); ++i)
            {
                if(used[i] || candidate[i].neighbor != expected.neighbor)
                    continue;
                double const distance = abs(candidate[i].centroid - expected.centroid);
                if(distance < best_distance)
                {
                    best = i;
                    best_distance = distance;
                }
            }
            require(best < candidate.size(), "full/partial neighbor ID mismatch");
            used[best] = 1;
            require(close(expected.area, candidate[best].area),
                    "full/partial face area mismatch");
            require(close(expected.centroid, candidate[best].centroid),
                    "full/partial face centroid mismatch");
            require(close(expected.normal, candidate[best].normal),
                    "full/partial face normal mismatch");
        }
        require(close(fluxProxy(full, global, global, points.size()),
                      fluxProxy(partial, local, global, points.size())),
                "full/partial conservative face-flux proxy mismatch");
    }

    if(active.size() == points.size())
    {
        double full_balance = 0;
        double partial_balance = 0;
        for(std::size_t global : active)
        {
            full_balance += fluxProxy(full, global, global, points.size());
            partial_balance += fluxProxy(partial, view.globalToLocal(global),
                                         global, points.size());
        }
        require(close(full_balance, 0, 2e-8),
                "full closed-domain flux proxy is not conservative");
        require(close(partial_balance, full_balance, 2e-8),
                "partial closed-domain flux conservation mismatch");
    }

    IdealGas eos(5.0 / 3.0);
    RigidWallGenerator3D ghost;
    std::vector<ComputationalCell3D> cells(points.size());
    for(std::size_t i = 0; i < cells.size(); ++i)
    {
        cells[i].density = 1.0 + 0.2 * points[i].x + 0.1 * points[i].y;
        cells[i].pressure = 0.8 + 0.1 * points[i].x + 0.15 * points[i].z;
        cells[i].internal_energy = cells[i].pressure /
            (cells[i].density * (5.0 / 3.0 - 1.0));
        cells[i].velocity = Vector3D(0.1 * points[i].y,
                                     -0.08 * points[i].x,
                                     0.04 * points[i].z);
    }

    LinearGauss3D full_reconstruction(eos, ghost);
    std::vector<unsigned char> full_active(points.size(), 0);
    for(std::size_t global : active)
        full_active[global] = 1;
    std::vector<std::pair<ComputationalCell3D, ComputationalCell3D>>
        full_face_values;
    full_reconstruction.InterpolateIndividual(
        full, cells, 0, full_active, full_face_values);
    std::vector<Slope3D> const full_slopes = full_reconstruction.GetSlopes();

    std::vector<ComputationalCell3D> partial_cells(view.localSize());
    for(std::size_t local = 0; local < view.localSize(); ++local)
        partial_cells[local] = cells[view.localToGlobal(local)];
    std::vector<ComputationalCell3D> synchronized_cells = cells;
    partial.SyncPartialBuildData(partial_cells, synchronized_cells);
    LinearGauss3D partial_reconstruction(eos, ghost);
    std::vector<unsigned char> partial_active(view.localSize(), 0);
    for(std::size_t global : active)
        partial_active[view.globalToLocal(global)] = 1;
    std::vector<std::pair<ComputationalCell3D, ComputationalCell3D>>
        partial_face_values;
    partial_reconstruction.InterpolateIndividual(
        partial, partial_cells, 0, partial_active, partial_face_values);
    std::vector<Slope3D> const partial_slopes = partial_reconstruction.GetSlopes();
    for(std::size_t global : active)
        compareSlope(full_slopes.at(global),
                     partial_slopes.at(view.globalToLocal(global)));
}

std::vector<Vector3D> makePoints(bool moved)
{
    std::vector<Vector3D> points;
    for(std::size_t ix = 0; ix < 5; ++ix)
        for(std::size_t iy = 0; iy < 5; ++iy)
            for(std::size_t iz = 0; iz < 5; ++iz)
            {
                std::size_t const index = points.size();
                double const jitter = 2e-7 *
                    (static_cast<double>((index * 37) % 17) - 8.0);
                Vector3D point((ix + 0.5) / 5.0 + jitter,
                               (iy + 0.5) / 5.0 - 0.5 * jitter,
                               (iz + 0.5) / 5.0 + 0.25 * jitter);
                if(moved)
                {
                    point.x += 0.008 * std::sin(0.7 * static_cast<double>(index));
                    point.y += 0.006 * std::cos(0.3 * static_cast<double>(index));
                    point.z += 0.004 * std::sin(0.5 * static_cast<double>(index));
                }
                points.push_back(point);
            }
    return points;
}

void testPartialGeometry()
{
    double const fractions[] = {0.01, 0.05, 0.10, 0.25, 0.50, 1.00};
    for(bool moved : {false, true})
    {
        std::vector<Vector3D> const points = makePoints(moved);
        for(double fraction : fractions)
        {
            std::size_t const count = std::max<std::size_t>(
                1, static_cast<std::size_t>(std::ceil(fraction * points.size())));
            std::vector<std::size_t> active;
            active.reserve(count);
            for(std::size_t i = 0; i < count; ++i)
                active.push_back((i * 53) % points.size());
            comparePartial(points, active);
        }
    }
}

void testSchedulerAndAMR()
{
    std::vector<Vector3D> points = {
        Vector3D(0.2, 0.2, 0.2), Vector3D(0.8, 0.2, 0.2),
        Vector3D(0.2, 0.8, 0.2), Vector3D(0.8, 0.8, 0.2),
        Vector3D(0.2, 0.2, 0.8), Vector3D(0.8, 0.2, 0.8),
        Vector3D(0.2, 0.8, 0.8), Vector3D(0.8, 0.8, 0.8)};
    Voronoi3D tess(Vector3D(0, 0, 0), Vector3D(1, 1, 1));
    tess.Build(points);
    std::vector<ComputationalCell3D> cells(points.size());
    for(std::size_t i = 0; i < cells.size(); ++i)
        cells[i].ID = 1000 + i;

    IndividualTimeStepOptions options;
    options.time_quantum = 1;
    options.initial_bin = 3;
    options.maximum_bin = 6;
    options.maximum_neighbor_bin_difference = 2;
    IndividualTimeStepScheduler scheduler(options);
    scheduler.initialize(cells, 2.5, 8);
    require(!scheduler.forceAllActiveLatched(),
            "forced-active latch was set on a fresh scheduler");
    scheduler.setForceAllActiveLatched(false);
    require(!scheduler.forceAllActiveLatched(),
            "false unexpectedly set the forced-active latch");
    scheduler.setForceAllActiveLatched(true);
    scheduler.setForceAllActiveLatched(false);
    require(scheduler.forceAllActiveLatched(),
            "forced-active latch was not monotonic");
    IndividualStepContext first = scheduler.prepareEvent(cells);
    require(first.event_tick == 8 && first.active_indices.size() == cells.size(),
            "initial scheduler synchronization is wrong");
    std::vector<double> limits(cells.size(), std::numeric_limits<double>::infinity());
    limits[0] = 2;
    first.cached_accelerations[0] = Vector3D(0.25, -0.5, 0.75);
    first.gravity_half_kick_pending[0] = 1;
    scheduler.commitEvent(first, tess, cells, limits);
    require(scheduler.states()[0].time_bin == 1,
            "scheduler did not decrease a bin immediately");
    for(std::size_t i = 1; i < cells.size(); ++i)
        require(scheduler.states()[i].time_bin <= 3,
                "neighbor-bin limiter allowed a gap larger than two");

    IndividualStepContext second = scheduler.prepareEvent(cells);
    require(second.event_tick == 10 && second.active_indices.size() == 1,
            "scheduler selected the wrong next event");
    require(close(second.cached_accelerations[0], Vector3D(0.25, -0.5, 0.75)) &&
            second.gravity_half_kick_pending[0] != 0,
            "scheduler did not preserve the gravity endpoint cache");
    limits.assign(cells.size(), std::numeric_limits<double>::infinity());
    limits[1] = 1;
    scheduler.commitEvent(second, tess, cells, limits);
    require(scheduler.states()[0].time_bin == 1,
            "scheduler increased a bin at a non-aligned tick");
    require(scheduler.states()[1].end_tick == 11,
            "scheduler did not wake an inactive cell for a shorter limit");

    RadiationRepairAccounting& accounting =
        scheduler.radiationRepairAccounting();
    accounting.repaired_cells = 17;
    accounting.repaired_groups = 23;
    accounting.cumulative_injected_energy = 0.125;
    accounting.maximum_relative_deficit = 5e-7;
    accounting.representative_cell_id = cells[0].ID;
    accounting.representative_group = 3;
    accounting.representative_rank = 2;
    accounting.representative_original_extent = -0.25;
    accounting.representative_floor_extent = 1e-10;
    accounting.representative_injected_extent = 0.2500000001;
    accounting.maximum_global_radiation_energy = 4096;
    accounting.next_warning_fraction = 4e-4;
    accounting.residual_correction_limited_groups = 11;
    accounting.residual_correction_signed_energy_bias = 0.25;
    accounting.residual_correction_absolute_energy_bias = 0.75;
    accounting.residual_correction_signed_bias_by_group = {0.5, -0.25};
    accounting.residual_correction_absolute_bias_by_group = {0.5, 0.25};
    accounting.residual_correction_minimum_scale = 0.5;
    accounting.positivity_rescue_events = 4;
    accounting.positivity_rescue_blocks = 9;
    accounting.positivity_rescue_additional_iterations = 83;
    accounting.residual_positive_floor_events = 3;
    accounting.residual_positive_floor_cells = 7;
    accounting.residual_positive_floor_groups = 12;
    accounting.residual_positive_floor_cumulative_injected_energy = 1e-8;
    accounting.residual_positive_floor_maximum_cell_injection_ratio = 4e-8;
    accounting.residual_positive_floor_maximum_global_injection_ratio = 2e-9;
    accounting.residual_positive_floor_maximum_post_true_residual_error = 3e-6;
    accounting.residual_positive_floor_initial_global_radiation_energy = 8192;
    std::vector<CellTimeState> saved = scheduler.states();
    IndividualTimeStepScheduler restored(options);
    restored.restore(cells, scheduler.timeOrigin(), scheduler.timeQuantum(),
                     scheduler.currentTick(), saved, accounting,
                     scheduler.radiationDefectAccounting(),
                     scheduler.forceAllActiveLatched());
    require(restored.states()[0].end_tick == scheduler.states()[0].end_tick,
            "scheduler restart changed exact ticks");
    require(restored.forceAllActiveLatched(),
            "scheduler restart lost the forced-active latch");
    IndividualTimeStepScheduler legacy_restored(options);
    legacy_restored.restore(cells, scheduler.timeOrigin(),
                            scheduler.timeQuantum(), scheduler.currentTick(),
                            saved, accounting);
    require(!legacy_restored.forceAllActiveLatched(),
            "legacy scheduler restart did not default the latch to false");
    RadiationRepairAccounting const& restored_accounting =
        restored.radiationRepairAccounting();
    require(restored_accounting.repaired_cells == accounting.repaired_cells &&
            restored_accounting.repaired_groups == accounting.repaired_groups &&
            restored_accounting.cumulative_injected_energy ==
                accounting.cumulative_injected_energy &&
            restored_accounting.maximum_relative_deficit ==
                accounting.maximum_relative_deficit &&
            restored_accounting.representative_cell_id ==
                accounting.representative_cell_id &&
            restored_accounting.representative_group ==
                accounting.representative_group &&
            restored_accounting.representative_rank ==
                accounting.representative_rank &&
            restored_accounting.representative_original_extent ==
                accounting.representative_original_extent &&
            restored_accounting.representative_floor_extent ==
                accounting.representative_floor_extent &&
            restored_accounting.representative_injected_extent ==
                accounting.representative_injected_extent &&
            restored_accounting.maximum_global_radiation_energy ==
                accounting.maximum_global_radiation_energy &&
            restored_accounting.next_warning_fraction ==
                accounting.next_warning_fraction &&
            restored_accounting.residual_correction_limited_groups ==
                accounting.residual_correction_limited_groups &&
            restored_accounting.residual_correction_signed_energy_bias ==
                accounting.residual_correction_signed_energy_bias &&
            restored_accounting.residual_correction_absolute_energy_bias ==
                accounting.residual_correction_absolute_energy_bias &&
            restored_accounting.residual_correction_signed_bias_by_group ==
                accounting.residual_correction_signed_bias_by_group &&
            restored_accounting.residual_correction_absolute_bias_by_group ==
                accounting.residual_correction_absolute_bias_by_group &&
            restored_accounting.residual_correction_minimum_scale ==
                accounting.residual_correction_minimum_scale &&
            restored_accounting.positivity_rescue_events ==
                accounting.positivity_rescue_events &&
            restored_accounting.positivity_rescue_blocks ==
                accounting.positivity_rescue_blocks &&
            restored_accounting.positivity_rescue_additional_iterations ==
                accounting.positivity_rescue_additional_iterations &&
            restored_accounting.residual_positive_floor_events ==
                accounting.residual_positive_floor_events &&
            restored_accounting.residual_positive_floor_cells ==
                accounting.residual_positive_floor_cells &&
            restored_accounting.residual_positive_floor_groups ==
                accounting.residual_positive_floor_groups &&
            restored_accounting.
                residual_positive_floor_cumulative_injected_energy ==
                accounting.
                    residual_positive_floor_cumulative_injected_energy &&
            restored_accounting.
                residual_positive_floor_maximum_cell_injection_ratio ==
                accounting.
                    residual_positive_floor_maximum_cell_injection_ratio &&
            restored_accounting.
                residual_positive_floor_maximum_global_injection_ratio ==
                accounting.
                    residual_positive_floor_maximum_global_injection_ratio &&
            restored_accounting.
                residual_positive_floor_maximum_post_true_residual_error ==
                accounting.
                    residual_positive_floor_maximum_post_true_residual_error &&
            restored_accounting.
                residual_positive_floor_initial_global_radiation_energy ==
                accounting.
                    residual_positive_floor_initial_global_radiation_energy,
            "scheduler restart changed spectral-repair accounting bits");

    std::vector<ComputationalCell3D> refined(cells.begin(), cells.end() - 1);
    ComputationalCell3D child = cells[0];
    child.ID = 9001;
    refined.push_back(child);
    IndividualAMRChangeSet changes;
    changes.child_parent_ids.push_back({child.ID, cells[0].ID});
    changes.removed_cell_ids.push_back(cells.back().ID);
    restored.applyAMRChangeSet(refined, changes);
    require(restored.states().back().cell_id == child.ID &&
            restored.states().back().time_bin == restored.states().front().time_bin &&
            restored.states().back().begin_tick == restored.states().front().begin_tick,
            "AMR child did not inherit the parent tick/bin");
}

void testSynchronizedScheduler()
{
    std::vector<Vector3D> points = {
        Vector3D(0.2, 0.2, 0.2), Vector3D(0.8, 0.2, 0.2),
        Vector3D(0.2, 0.8, 0.2), Vector3D(0.8, 0.8, 0.2),
        Vector3D(0.2, 0.2, 0.8), Vector3D(0.8, 0.2, 0.8),
        Vector3D(0.2, 0.8, 0.8), Vector3D(0.8, 0.8, 0.8)};
    Voronoi3D tess(Vector3D(0, 0, 0), Vector3D(1, 1, 1));
    tess.Build(points);
    std::vector<ComputationalCell3D> cells(points.size());
    for(std::size_t i = 0; i < cells.size(); ++i)
        cells[i].ID = 2000 + i;

    IndividualTimeStepOptions options;
    options.time_quantum = 1;
    options.initial_bin = 3;
    options.maximum_bin = 6;
    options.force_synchronized = true;
    IndividualTimeStepScheduler scheduler(options);
    scheduler.initialize(cells, 0, 8);
    IndividualStepContext event = scheduler.prepareEvent(cells);
    std::vector<double> limits(cells.size(),
        std::numeric_limits<double>::infinity());
    limits[3] = 2;
    scheduler.commitEvent(event, tess, cells, limits);
    for(CellTimeState const& state : scheduler.states())
        require(state.time_bin == 1 && state.end_tick == 10,
            "synchronized scheduler did not apply the shared minimum limit");

    event = scheduler.prepareEvent(cells);
    require(event.active_indices.size() == cells.size(),
        "synchronized scheduler did not activate every cell");
    limits.assign(cells.size(), std::numeric_limits<double>::infinity());
    scheduler.commitEvent(event, tess, cells, limits);
    for(CellTimeState const& state : scheduler.states())
        require(state.time_bin == 1,
            "synchronized scheduler increased at a non-aligned tick");

    event = scheduler.prepareEvent(cells);
    scheduler.commitEvent(event, tess, cells, limits);
    for(CellTimeState const& state : scheduler.states())
        require(state.time_bin == 2 && state.end_tick == 16,
            "synchronized scheduler did not increase together at alignment");

    std::size_t const clamped = scheduler.clampToTerminalTick(15);
    require(clamped == cells.size() && scheduler.nextEventTick() == 15,
        "terminal synchronization did not clamp every crossing interval");
    event = scheduler.prepareEvent(cells);
    require(event.event_tick == 15 &&
            event.active_indices.size() == cells.size(),
        "terminal synchronization did not activate every cell");
    for(std::size_t i = 0; i < cells.size(); ++i)
        require(close(event.cellTimeStep(i), 3),
            "terminal synchronization lost the shortened interval");
}

void testInactiveWakeAlignment()
{
    std::vector<Vector3D> const points = {
        Vector3D(0.25, 0.5, 0.5), Vector3D(0.75, 0.5, 0.5)};
    Voronoi3D tess(Vector3D(0, 0, 0), Vector3D(1, 1, 1));
    tess.Build(points);
    std::vector<ComputationalCell3D> cells(points.size());
    cells[0].ID = 3000;
    cells[1].ID = 3001;

    IndividualTimeStepOptions options;
    options.time_quantum = 1;
    options.initial_bin = 3;
    options.maximum_bin = 6;
    IndividualTimeStepScheduler initial_scheduler(options);
    initial_scheduler.initialize(cells, 0, 8);

    std::vector<CellTimeState> states = initial_scheduler.states();
    states[0].begin_tick = 8;
    states[0].end_tick = 9;
    states[0].time_bin = 0;
    states[1].begin_tick = 8;
    states[1].end_tick = 16;
    states[1].time_bin = 3;
    IndividualTimeStepScheduler scheduler(options);
    scheduler.restore(cells, 0, 1, 8, states);

    IndividualStepContext event = scheduler.prepareEvent(cells);
    require(event.event_tick == 9 && event.active_indices.size() == 1,
            "wake-alignment setup selected the wrong event");
    std::vector<double> limits(cells.size(),
        std::numeric_limits<double>::infinity());
    limits[1] = 2;
    scheduler.commitEvent(event, tess, cells, limits);
    require(scheduler.states()[1].time_bin == 1 &&
            scheduler.states()[1].end_tick == 10,
            "inactive wake did not choose the next aligned endpoint");

    event = scheduler.prepareEvent(cells);
    require(event.event_tick == 10 && event.active_indices.size() == cells.size(),
            "aligned wake did not activate at the next bin boundary");
    limits.assign(cells.size(), std::numeric_limits<double>::infinity());
    scheduler.commitEvent(event, tess, cells, limits);

    event = scheduler.prepareEvent(cells);
    require(event.event_tick == 12 && event.active_indices.size() == cells.size(),
            "aligned cells did not remain synchronized");
    scheduler.commitEvent(event, tess, cells, limits);
    for(CellTimeState const& state : scheduler.states())
        require(state.time_bin == 2 && state.end_tick == 16,
                "woken cell could not increase its bin after realignment");
}

void testForcedAllActiveEventOverlay()
{
    std::vector<Vector3D> const points = {
        Vector3D(0.25, 0.5, 0.5), Vector3D(0.75, 0.5, 0.5)};
    Voronoi3D tess(Vector3D(0, 0, 0), Vector3D(1, 1, 1));
    tess.Build(points);
    std::vector<ComputationalCell3D> cells(2);
    cells[0].ID = 3100;
    cells[1].ID = 3101;

    IndividualTimeStepOptions options;
    options.time_quantum = 0.5;
    options.initial_bin = 3;
    options.maximum_bin = 6;
    IndividualTimeStepScheduler initial_scheduler(options);
    initial_scheduler.initialize(cells, 0, 4);

    std::vector<CellTimeState> states = initial_scheduler.states();
    states[0].begin_tick = 8;
    states[0].end_tick = 10;
    states[0].time_bin = 1;
    states[1].begin_tick = 8;
    states[1].end_tick = 16;
    states[1].time_bin = 3;
    IndividualTimeStepScheduler scheduler(options);
    scheduler.restore(cells, 0, 0.5, 8, states);

    IndividualStepContext const normal = scheduler.prepareEvent(cells, 10);
    require(normal.active_indices.size() == 1 && normal.isActive(0) &&
            !normal.isActive(1) && close(normal.cellTimeStep(1), 4),
        "ordinary event behavior changed by forced-overlay support");

    IndividualStepContext const forced = scheduler.prepareEvent(cells, 10, true);
    require(forced.active_indices.size() == cells.size() &&
            forced.isActive(0) && forced.isActive(1),
        "forced event did not transiently activate every cell");
    require(close(forced.cellTimeStep(0), 1) &&
            close(forced.cellTimeStep(1), 1),
        "forced event did not use each cell's begin-to-event interval");
    require(scheduler.states()[0].begin_tick == 8 &&
            scheduler.states()[0].end_tick == 10 &&
            scheduler.states()[1].begin_tick == 8 &&
            scheduler.states()[1].end_tick == 16,
        "preparation of a forced event mutated persistent scheduler state");

    std::vector<double> limits(cells.size(),
        std::numeric_limits<double>::infinity());
    scheduler.commitEvent(forced, tess, cells, limits);
    require(scheduler.states()[0].time_bin == 1 &&
            scheduler.states()[0].end_tick == 12 &&
            scheduler.states()[1].time_bin == 1 &&
            scheduler.states()[1].end_tick == 12,
        "forced event grew from a cell's stale scheduled bin instead of its completed interval");
}

void testDefaultCellUpdaterAllActiveCommit()
{
    std::vector<Vector3D> const points = {
        Vector3D(0.2, 0.2, 0.2), Vector3D(0.8, 0.2, 0.2),
        Vector3D(0.2, 0.8, 0.2), Vector3D(0.8, 0.8, 0.2),
        Vector3D(0.2, 0.2, 0.8), Vector3D(0.8, 0.2, 0.8),
        Vector3D(0.2, 0.8, 0.8), Vector3D(0.8, 0.8, 0.8)};
    Voronoi3D tess(Vector3D(0, 0, 0), Vector3D(1, 1, 1));
    tess.Build(points);

    std::size_t const owned_count = tess.GetPointNo();
    std::size_t const mesh_count = tess.getMeshPoints().size();
    require(owned_count == points.size() && mesh_count >= owned_count,
            "all-active cell-update test built an unexpected mesh");
    IdealGas eos(5.0 / 3.0);
    std::vector<ComputationalCell3D> initial_cells(mesh_count);
    std::vector<Conserved3D> initial_extensives(mesh_count);
    for(std::size_t index = 0; index < owned_count; ++index)
    {
        ComputationalCell3D& cell = initial_cells[index];
        cell.ID = 3500 + index;
        cell.density = 1.0 + 0.05 * static_cast<double>(index);
        cell.pressure = 0.8 + 0.03 * static_cast<double>(index);
        cell.internal_energy = cell.pressure /
            (cell.density * (5.0 / 3.0 - 1.0));
        cell.velocity = Vector3D(0.01 * static_cast<double>(index),
                                 -0.02 * static_cast<double>(index),
                                 0.03 * static_cast<double>(index));
        cell.Erad = 0.25 + 0.01 * static_cast<double>(index);
        for(std::size_t group = 0; group < ENERGY_GROUPS_NUM; ++group)
            cell.Eg[group] = cell.Erad /
                static_cast<double>(ENERGY_GROUPS_NUM);
        PrimitiveToConserved(
            cell, tess.GetVolume(index), initial_extensives[index]);
    }
    for(std::size_t index = owned_count; index < mesh_count; ++index)
    {
        ComputationalCell3D& cell = initial_cells[index];
        Conserved3D& extensive = initial_extensives[index];
        cell.ID = 9000 + index;
        cell.density = 100.0 + static_cast<double>(index);
        cell.pressure = 200.0 + static_cast<double>(index);
        cell.internal_energy = 300.0 + static_cast<double>(index);
        cell.temperature = 400.0 + static_cast<double>(index);
        cell.velocity = Vector3D(1, 2, 3);
        cell.Erad = 500.0 + static_cast<double>(index);
        extensive.mass = 600.0 + static_cast<double>(index);
        extensive.momentum = Vector3D(4, 5, 6);
        extensive.energy = 700.0 + static_cast<double>(index);
        extensive.internal_energy = 800.0 + static_cast<double>(index);
        extensive.Erad = 900.0 + static_cast<double>(index);
    }

    IndividualStepContext all_active;
    all_active.active_indices.resize(owned_count);
    std::iota(all_active.active_indices.begin(),
              all_active.active_indices.end(), 0);
    all_active.active_mask.assign(mesh_count, 0);
    std::fill(all_active.active_mask.begin(),
              all_active.active_mask.begin() + owned_count, 1);

    DefaultCellUpdater updater;
    ScopedEnvironmentVariable option(
        "RICH_INDIVIDUAL_ALL_ACTIVE_CELL_UPDATE");

    option.set(nullptr);
    std::vector<ComputationalCell3D> legacy_cells = initial_cells;
    std::vector<Conserved3D> legacy_extensives = initial_extensives;
    ComputationalCell3D const* const legacy_cells_data = legacy_cells.data();
    Conserved3D const* const legacy_extensives_data =
        legacy_extensives.data();
    updater.UpdateIndividual(
        legacy_cells, eos, tess, legacy_extensives, all_active);
    require(legacy_cells.data() == legacy_cells_data &&
            legacy_extensives.data() == legacy_extensives_data,
            "default-off cell update unexpectedly replaced whole vectors");

    option.set("1");
    std::vector<ComputationalCell3D> fast_cells = initial_cells;
    std::vector<Conserved3D> fast_extensives = initial_extensives;
    ComputationalCell3D const* const fast_cells_data = fast_cells.data();
    Conserved3D const* const fast_extensives_data = fast_extensives.data();
    updater.UpdateIndividual(fast_cells, eos, tess, fast_extensives, all_active);
    require(fast_cells.data() != fast_cells_data &&
            fast_extensives.data() != fast_extensives_data,
            "all-active cell update did not use whole-vector commit");
    requireCellUpdateStateEqual(
        legacy_cells, legacy_extensives, fast_cells, fast_extensives,
        "all-active cell update differs from legacy");

    IndividualStepContext partial = all_active;
    partial.active_indices.pop_back();
    partial.active_mask[owned_count - 1] = 0;
    std::vector<ComputationalCell3D> partial_cells = initial_cells;
    std::vector<Conserved3D> partial_extensives = initial_extensives;
    ComputationalCell3D const* const partial_cells_data = partial_cells.data();
    Conserved3D const* const partial_extensives_data =
        partial_extensives.data();
    updater.UpdateIndividual(
        partial_cells, eos, tess, partial_extensives, partial);
    require(partial_cells.data() == partial_cells_data &&
            partial_extensives.data() == partial_extensives_data,
            "partial cell update did not retain the generic commit path");

    std::vector<ComputationalCell3D> invalid_cells = initial_cells;
    std::vector<Conserved3D> invalid_extensives = initial_extensives;
    invalid_extensives[0].mass = 0;
    std::vector<ComputationalCell3D> const invalid_cells_before = invalid_cells;
    std::vector<Conserved3D> const invalid_extensives_before =
        invalid_extensives;
    bool threw = false;
    try
    {
        updater.UpdateIndividual(
            invalid_cells, eos, tess, invalid_extensives, all_active);
    }
    catch(...)
    {
        threw = true;
    }
    require(threw, "all-active cell-update exception test did not throw");
    requireCellUpdateStateEqual(
        invalid_cells_before, invalid_extensives_before,
        invalid_cells, invalid_extensives,
        "all-active cell-update exception changed caller state");
}

void testHydroRadiationPositivityLimiter()
{
    std::vector<Vector3D> points{
        Vector3D(0.2, 0.2, 0.2), Vector3D(0.8, 0.2, 0.2),
        Vector3D(0.2, 0.8, 0.2), Vector3D(0.8, 0.8, 0.2),
        Vector3D(0.2, 0.2, 0.8), Vector3D(0.8, 0.2, 0.8),
        Vector3D(0.2, 0.8, 0.8), Vector3D(0.8, 0.8, 0.8)};
    Voronoi3D tess(Vector3D(0, 0, 0), Vector3D(1, 1, 1));
    tess.Build(points);

    std::vector<ComputationalCell3D> cells(points.size());
    std::vector<Conserved3D> extensives(points.size());
    for(std::size_t cell = 0; cell < cells.size(); ++cell)
    {
        cells[cell].ID = 4000 + cell;
        extensives[cell].mass = 1;
        extensives[cell].energy = 1;
        extensives[cell].internal_energy = 1;
        extensives[cell].Erad = ENERGY_GROUPS_NUM;
        for(std::size_t group = 0; group < ENERGY_GROUPS_NUM; ++group)
            extensives[cell].Eg[group] = 1;
    }

    std::size_t target_face = tess.GetTotalFacesNumber();
    std::pair<std::size_t, std::size_t> target_neighbors;
    for(std::size_t face = 0; face < tess.GetTotalFacesNumber(); ++face)
    {
        const auto neighbors = tess.GetFaceNeighbors(face);
        if(neighbors.first < points.size() && neighbors.second < points.size())
        {
            target_face = face;
            target_neighbors = neighbors;
            break;
        }
    }
    require(target_face < tess.GetTotalFacesNumber(),
            "positivity test could not find an internal Voronoi face");

    IndividualStepContext context;
    context.active_indices.push_back(target_neighbors.first);
    context.active_mask.assign(points.size(), 0);
    context.active_mask[target_neighbors.first] = 1;
    context.cell_time_steps.assign(points.size(), 1);

    std::vector<Conserved3D> fluxes(tess.GetTotalFacesNumber());
    const double requested_transfer = 2;
    const double flux_density = requested_transfer / tess.GetArea(target_face);
    fluxes[target_face].Erad = flux_density;
    fluxes[target_face].Eg[0] = flux_density;

    double group_before = 0;
    double total_before = 0;
    for(const Conserved3D& extensive : extensives)
    {
        group_before += extensive.Eg[0];
        total_before += extensive.Erad;
    }

    DefaultExtensiveUpdater updater;
    updater.UpdateIndividual(fluxes, tess, context, cells, extensives, 0,
                             std::vector<Vector3D>(), std::vector<Vector3D>(),
                             std::vector<std::pair<ComputationalCell3D,
                                                   ComputationalCell3D> >());

    double group_after = 0;
    double total_after = 0;
    for(const Conserved3D& extensive : extensives)
    {
        require(extensive.Eg[0] >= 0,
                "individual hydro left a negative radiation-group extent");
        group_after += extensive.Eg[0];
        total_after += extensive.Erad;
    }
    require(close(group_before, group_after, 1e-12),
            "hydro radiation-group limiter is not conservative");
    require(close(total_before, total_after, 1e-12),
            "hydro total-radiation limiter is not conservative");
}

void testSpectralRoundoffPositivityRepair()
{
    std::vector<double> spectrum{4, -1, 2};
    std::vector<double> const original_spectrum = spectrum;
    double aggregate = 5;
    auto const controlled =
        RadiationPositivity::RepairControlledNegativeGroupExtents(
            spectrum, aggregate, 0.2, 1e30);
    auto const& repaired = controlled.repair;
    require(repaired.valid && repaired.repaired,
            "small spectral negativity was not repaired");
    require(repaired.repaired_groups == 1 &&
            repaired.most_negative_group == 1,
            "spectral repair reported the wrong negative group");
    require(spectrum[0] == original_spectrum[0] &&
            spectrum[2] == original_spectrum[2] &&
            spectrum[1] == 6e-11,
            "spectral repair changed a positive group or used the wrong floor");
    require(close(repaired.injected_extent, 1 + 6e-11, 1e-14),
            "spectral repair reported the wrong injected extent");
    require(aggregate ==
                std::accumulate(spectrum.begin(), spectrum.end(), 0.0),
            "controlled repair did not make Erad exactly equal sum(Eg)");

    std::vector<double> rejected{4, -1, 2};
    std::vector<double> const original = rejected;
    auto const too_large =
        RadiationPositivity::RepairSmallNegativeGroupExtents(
            rejected, 5, 0.01);
    require(!too_large.valid && rejected == original &&
                too_large.failure ==
                    RadiationPositivity::SpectralRepairFailure::
                        NegativeExtentExceedsTolerance,
            "large spectral negativity lacked its classified rejection");

    double constexpr tolerance =
        RadiationPositivity::spectral_repair_relative_limit;
    for(std::size_t const group_count : {2u, 3u, 7u, 16u}) {
        std::vector<double> groups(group_count, 1);
        double const positive = static_cast<double>(group_count - 1);
        groups.back() = -tolerance * positive;
        std::vector<double> const positive_reference = groups;
        auto const at_boundary =
            RadiationPositivity::RepairSmallNegativeGroupExtents(
                groups, positive, tolerance);
        require(at_boundary.valid && at_boundary.repaired,
                "inclusive spectral-repair boundary was rejected");
        require(groups.back() ==
                    RadiationPositivity::spectral_repair_floor_fraction *
                        positive,
                "spectral floor was divided by the runtime group count");
        for(std::size_t group = 0; group + 1 < group_count; ++group)
            require(groups[group] == positive_reference[group],
                    "spectral repair changed a positive runtime group");

        std::vector<double> over_boundary(group_count, 1);
        over_boundary.back() = -std::nextafter(
            tolerance * positive,
            std::numeric_limits<double>::infinity());
        std::vector<double> const over_reference = over_boundary;
        auto const rejected_boundary =
            RadiationPositivity::RepairSmallNegativeGroupExtents(
                over_boundary, positive, tolerance);
        require(!rejected_boundary.valid &&
                over_boundary == over_reference,
                "spectral deficit just above the inclusive limit was repaired");
    }

    std::vector<double> multiple{1, 1, 1, 1, 1, -2e-6, -3e-6};
    auto const multiple_repair =
        RadiationPositivity::RepairSmallNegativeGroupExtents(
            multiple, 5, tolerance);
    require(multiple_repair.valid && multiple_repair.repaired &&
            multiple_repair.repaired_groups == 2,
            "multiple negative runtime groups were not repaired");
    require(multiple[5] == multiple_repair.floor_extent &&
            multiple[6] == multiple_repair.floor_extent &&
            multiple_repair.floor_extent ==
                RadiationPositivity::spectral_repair_floor_fraction *
                    multiple_repair.positive_extent,
            "multiple negative runtime groups did not receive the full floor");

    std::vector<double> zero_and_positive{0, 2, 3};
    std::vector<double> const zero_reference = zero_and_positive;
    auto const no_repair =
        RadiationPositivity::RepairSmallNegativeGroupExtents(
            zero_and_positive, 5, tolerance);
    require(no_repair.valid && !no_repair.repaired &&
            zero_and_positive == zero_reference,
            "zero or positive spectral groups were changed");

    std::vector<double> negative_total_groups{1, -1e-3};
    auto const negative_total =
        RadiationPositivity::RepairSmallNegativeGroupExtents(
            negative_total_groups, -1, tolerance);
    require(!negative_total.valid &&
                negative_total.repaired_groups == 0 &&
                negative_total.failure ==
                    RadiationPositivity::SpectralRepairFailure::
                        NegativeTotalExtent &&
                std::string(
                    RadiationPositivity::SpectralRepairFailureLabel(
                        negative_total.failure)) == "negative_total_extent",
            "negative total radiation extent lacked a classified failure");

    std::vector<double> nonfinite_groups{
        std::numeric_limits<double>::quiet_NaN(), 1};
    auto const nonfinite_group =
        RadiationPositivity::RepairSmallNegativeGroupExtents(
            nonfinite_groups, 1, tolerance);
    require(!nonfinite_group.valid &&
                nonfinite_group.failure ==
                    RadiationPositivity::SpectralRepairFailure::
                        NonfiniteGroupExtent &&
                nonfinite_group.failure_group == 0,
            "nonfinite radiation group lacked a group-local failure");

    auto const passive_roundoff =
        RadiationPositivity::ClassifyPassiveRoundoff(
            -0.0078125, 4.6172029679959712e16, 5.9775037752781104e16);
    require(passive_roundoff.valid && passive_roundoff.repairable,
            "face-scale passive roundoff was not classified as repairable");
    auto const significant_negative =
        RadiationPositivity::ClassifyPassiveRoundoff(
            -7.4859487514877788e18, 2.45728709578e8, 3.62702771734e11);
    require(significant_negative.valid && !significant_negative.repairable,
            "significant passive negativity was misclassified as roundoff");

    RadiationRepairAccounting warning_accounting;
    require(!AdvanceRadiationRepairWarning(warning_accounting, 9.999e-5) &&
            warning_accounting.next_warning_fraction == 1e-4,
            "spectral-repair warning fired below the first threshold");
    require(AdvanceRadiationRepairWarning(warning_accounting, 1e-4) &&
            warning_accounting.next_warning_fraction == 2e-4,
            "spectral-repair first warning threshold is not inclusive");
    require(AdvanceRadiationRepairWarning(warning_accounting, 3.999e-4) &&
            warning_accounting.next_warning_fraction == 4e-4,
            "spectral-repair warning schedule did not advance by doubling");
    require(AdvanceRadiationRepairWarning(warning_accounting, 4e-4) &&
            warning_accounting.next_warning_fraction == 8e-4,
            "spectral-repair doubling threshold is not inclusive");
    require(!AdvanceRadiationRepairWarning(
                warning_accounting,
                std::numeric_limits<double>::quiet_NaN()) &&
            warning_accounting.next_warning_fraction == 8e-4,
            "non-finite injection fraction changed the warning schedule");
}

void testBiCGSTABWorkspaceRelease()
{
    CG::BiCGSTABWorkspace Workspace;
    Workspace.A.assign(2, CG::vec(3, 1));
    Workspace.A_indeces.assign(2, CG::vec_size_t(3, 0));
    std::vector<double>* DoubleBuffers[] = {
        &Workspace.A_diag, &Workspace.b, &Workspace.sub_x, &Workspace.M,
        &Workspace.r_old, &Workspace.sub_a_times_p, &Workspace.sub_r,
        &Workspace.sub_p, &Workspace.sub_r0, &Workspace.y, &Workspace.z,
        &Workspace.v, &Workspace.h, &Workspace.s, &Workspace.t,
        &Workspace.scratch_rescale1, &Workspace.scratch_rescale2,
        &Workspace.old_x};
    for(std::vector<double>* Buffer : DoubleBuffers)
        Buffer->assign(8, 1);
    Workspace.A_row_ptr.assign(9, 0);
    Workspace.A_col_idx.assign(16, 0);
    Workspace.A_values.assign(16, 1);
    Workspace.fixed16_block_stencil.LocalCellCount = 1;
    Workspace.fixed16_block_stencil.VectorCellCount = 1;
    Workspace.fixed16_block_stencil.LocalBlockValues.assign(256, 1);
    Workspace.fixed16_block_stencil.NeighborOffsets = {0, 0};
    Workspace.historical_correction.pre_correction_solution.assign(8, 1);
    require(Workspace.HasAllocatedStorage(),
            "BiCGSTAB workspace fixture did not allocate storage");

    Workspace.Release();

    require(!Workspace.HasAllocatedStorage() && Workspace.A.capacity() == 0 &&
            Workspace.A_indeces.capacity() == 0 &&
            !Workspace.fixed16_block_stencil.HasAllocatedStorage() &&
            Workspace.historical_correction.pre_correction_solution.empty() &&
            Workspace.historical_correction.pre_correction_solution.capacity() == 0,
            "BiCGSTAB workspace release retained solver storage");
}

void testFixed16BlockStencilCore()
{
    std::size_t constexpr block_size =
        CG::Fixed16BlockStencilMatrix::BlockSize;
    std::size_t constexpr cell_count = 2;
    std::size_t constexpr row_count = cell_count * block_size;
    std::vector<std::size_t> row_offsets(row_count + 1, 0);
    std::vector<std::size_t> columns;
    std::vector<double> values;
    columns.reserve(row_count * (block_size + 1));
    values.reserve(row_count * (block_size + 1));
    for(std::size_t cell = 0; cell < cell_count; ++cell)
        for(std::size_t row_group = 0; row_group < block_size; ++row_group)
        {
            std::size_t const row = cell * block_size + row_group;
            columns.push_back(row);
            values.push_back(4.0 + static_cast<double>(row) / 32.0);
            for(std::size_t group = 0; group < block_size; ++group)
                if(group != row_group)
                {
                    columns.push_back(cell * block_size + group);
                    values.push_back(-0.001 * static_cast<double>(group + 1));
                }
            std::size_t const neighbor_cell = 1 - cell;
            columns.push_back(neighbor_cell * block_size + row_group);
            values.push_back(-0.125 -
                0.001 * static_cast<double>(row_group));
            row_offsets[row + 1] = columns.size();
        }

    CG::Fixed16BlockStencilMatrix stencil;
    require(CG::BuildFixed16BlockStencilFromCSR(
                row_offsets, columns, values, row_count, stencil) ==
                CG::Fixed16BlockStencilFallback::None &&
            CG::ValidateFixed16BlockStencil(stencil) ==
                CG::Fixed16BlockStencilFallback::None &&
            CG::ValidateFixed16BlockStencilCSRShadow(
                stencil, row_offsets, columns, values) ==
                CG::Fixed16BlockStencilFallback::None,
            "valid fixed-16 block stencil failed construction");
    require(stencil.LocalCellCount == cell_count &&
            stencil.VectorCellCount == cell_count &&
            stencil.LocalBlockValues.size() == cell_count * 256 &&
            stencil.NeighborOffsets ==
                std::vector<std::size_t>({0, 1, 2}) &&
            stencil.NeighborCells ==
                std::vector<std::size_t>({1, 0}) &&
            stencil.NeighborValues.size() == row_count,
            "fixed-16 block-stencil layout changed");

    std::vector<double> input(row_count);
    for(std::size_t row = 0; row < row_count; ++row)
        input[row] = 0.25 + static_cast<double>(row) / 17.0;
    std::vector<double> csr_output(row_count, 0.0);
    std::vector<double> stencil_output;
    for(std::size_t row = 0; row < row_count; ++row)
        for(std::size_t entry = row_offsets[row];
            entry < row_offsets[row + 1]; ++entry)
            csr_output[row] += values[entry] * input[columns[entry]];
    CG::mat_times_vec_fixed16_block_stencil(
        stencil, input, stencil_output);
    require(stencil_output == csr_output,
            "fixed-16 block stencil changed exact CSR operation order");
    std::vector<double> vectorized_stencil_output;
    CG::mat_times_vec_fixed16_block_stencil(
        stencil, input, vectorized_stencil_output, true);
    require(vectorized_stencil_output == stencil_output,
            "fixed-16 AVX2 neighbor kernel changed scalar operation order");

    std::vector<double> aliased = input;
    CG::mat_times_vec_fixed16_block_stencil(stencil, aliased, aliased);
    require(aliased == csr_output,
            "fixed-16 block-stencil matvec is not alias safe");
    std::vector<double> vectorized_aliased = input;
    CG::mat_times_vec_fixed16_block_stencil(
        stencil, vectorized_aliased, vectorized_aliased, true);
    require(vectorized_aliased == csr_output,
            "fixed-16 AVX2 neighbor kernel is not alias safe");

    std::vector<double> inverse_diagonal(row_count);
    for(std::size_t row = 0; row < row_count; ++row)
        inverse_diagonal[row] = 1.0 / values[row_offsets[row]];
    CG::CellBlockJacobiPreconditioner csr_preconditioner;
    CG::CellBlockJacobiPreconditioner stencil_preconditioner;
    require(csr_preconditioner.SetupCSR(
                row_offsets, columns, values, block_size,
                CG::PreconditionerKind::CellBlockJacobi,
                inverse_diagonal) &&
            stencil_preconditioner.SetupFixed16BlockStencil(
                stencil, CG::PreconditionerKind::CellBlockJacobi,
                inverse_diagonal),
            "fixed-16 block-stencil preconditioner setup failed");
    std::vector<double> csr_preconditioned;
    std::vector<double> stencil_preconditioned;
    csr_preconditioner.Apply(input, csr_preconditioned);
    stencil_preconditioner.Apply(input, stencil_preconditioned);
    require(stencil_preconditioned.size() == csr_preconditioned.size(),
            "fixed-16 block preconditioner changed output size");
    for(std::size_t row = 0; row < row_count; ++row)
        require(close(stencil_preconditioned[row],
                      csr_preconditioned[row], 2e-14),
                "fixed-16 block preconditioner diverged from CSR");

    CG::Fixed16BlockStencilMatrix shadow_candidate = stencil;
    shadow_candidate.NeighborValues[0] += 1.0;
    require(CG::ValidateFixed16BlockStencilCSRShadow(
                shadow_candidate, row_offsets, columns, values) ==
                CG::Fixed16BlockStencilFallback::CSRValueMismatch,
            "fixed-16 CSR shadow accepted a value mismatch");
    std::vector<double> shadow_output;
    CG::mat_times_vec_fixed16_block_stencil(
        shadow_candidate, input, shadow_output);
    require(shadow_output != csr_output,
            "fixed-16 block-stencil shadow did not detect a value mismatch");

    CG::Fixed16BlockStencilMatrix invalid = stencil;
    invalid.NeighborCells[0] = 0;
    require(CG::ValidateFixed16BlockStencil(invalid) ==
                CG::Fixed16BlockStencilFallback::NeighborCells,
            "same-cell spatial neighbor did not fail closed");
    std::vector<std::size_t> malformed_columns = columns;
    malformed_columns[row_offsets[1] + block_size] = block_size + 2;
    require(CG::BuildFixed16BlockStencilFromCSR(
                row_offsets, malformed_columns, values, row_count, invalid) ==
                CG::Fixed16BlockStencilFallback::CSRNeighborOrder &&
            !invalid.HasAllocatedStorage(),
            "malformed neighbor ordering retained a stencil candidate");

    CG::Fixed16BlockStencilMatrix empty;
    require(CG::BuildFixed16BlockStencilFromCSR(
                {0}, {}, {}, 0, empty) ==
                CG::Fixed16BlockStencilFallback::None &&
            CG::ValidateFixed16BlockStencil(empty) ==
                CG::Fixed16BlockStencilFallback::None,
            "zero-owned fixed-16 block stencil failed validation");
    std::vector<double> empty_output{1.0};
    CG::mat_times_vec_fixed16_block_stencil(empty, {}, empty_output);
    require(empty_output.empty(),
            "zero-owned fixed-16 block stencil produced output");
    CG::mat_times_vec_fixed16_block_stencil(
        empty, empty_output, empty_output);
    require(empty_output.empty(),
            "zero-owned aliased block-stencil matvec produced output");
}

void testPrecomputedComptonLinearTables()
{
    std::vector<double> const centers = {1.0e-11, 4.0e-11, 1.0e-10};
    std::vector<double> const boundaries = {
        1.0e-12, 2.0e-11, 6.0e-11, 2.0e-10};
    ComptonMatrixMC generator(centers, boundaries, 64, true, 7);
    generator.set_tables({1.0e5, 2.0e5, 4.0e5});

    using ComptonMatrix = std::vector<std::vector<double>>;
    ComptonMatrix legacy_tau(3, std::vector<double>(3, 0.0));
    ComptonMatrix legacy_derivative(3, std::vector<double>(3, 0.0));
    generator.get_tau_matrix(
        1.5e5, 1.0e-7, 1.0, 1.0, legacy_tau, legacy_derivative);

    generator.set_precomputed_linear_tables(true);
    ComptonMatrix cached_tau(3, std::vector<double>(3, 0.0));
    ComptonMatrix cached_derivative(3, std::vector<double>(3, 0.0));
    generator.get_tau_matrix(
        1.5e5, 1.0e-7, 1.0, 1.0, cached_tau, cached_derivative);

    require(cached_tau == legacy_tau,
            "precomputed Compton interpolation changed tau");
    require(cached_derivative == legacy_derivative,
            "precomputed Compton interpolation changed d(tau)/dUm");

}

void testCompactWideRadiationMatrixRows()
{
    static_assert(sizeof(std::size_t) >= 8,
                  "wide distributed cell IDs require 64-bit size_t");
    static_assert(std::is_same<CG::vec, std::vector<double> >::value,
                  "radiation rows must not embed a fixed inline payload");
    static_assert(
        std::is_same<CG::vec_size_t,
                     std::vector<CG::matrix_index_t> >::value,
        "radiation index rows must not embed a fixed inline payload");

    std::size_t const wide_cell_id =
        (std::size_t{1} << 32) + std::size_t{12345};
    CG::mat matrix(1);
    CG::size_t_mat columns(1);
    matrix[0] = {2.0, -0.25};
    columns[0] = {0, wide_cell_id};
    require(columns[0][1] == wide_cell_id,
            "radiation matrix column truncated above 2^32");

    CG::CellBlockJacobiPreconditioner preconditioner;
    require(preconditioner.Setup(
                matrix, columns, 1,
                CG::PreconditionerKind::CellBlockJacobi, {0.5}),
            "wide-column block preconditioner setup failed");
    std::vector<double> output;
    preconditioner.Apply({4.0}, output);
    require(output.size() == 1 && close(output[0], 2.0, 2e-15),
            "wide remote column changed the same-cell block solve");

    CG::CellBlockJacobiPreconditioner csr_preconditioner;
    std::vector<std::size_t> const row_offsets{0, 2};
    require(csr_preconditioner.SetupCSR(
                row_offsets, columns[0], matrix[0], 1,
                CG::PreconditionerKind::CellBlockJacobi, {0.5}),
            "wide-column CSR block preconditioner setup failed");
    std::vector<double> csr_output;
    csr_preconditioner.Apply({4.0}, csr_output);
    require(csr_output == output,
            "CSR block preconditioner changed the same-cell solve");
}

void testCellBlockCSRProductionShapes()
{
    std::size_t constexpr block_size = 16;
    std::size_t constexpr block_count = 3;
    std::size_t constexpr row_count = block_size * block_count;
    CG::mat matrix(row_count);
    CG::size_t_mat columns(row_count);
    std::vector<double> inverse_diagonal(row_count, 0);
    std::vector<double> input(row_count, 0);

    auto append = [&](std::size_t const row, std::size_t const column,
                      double const value)
    {
        columns[row].push_back(column);
        matrix[row].push_back(value);
    };

    // Block 0 is nonsingular, requires a pivot-row swap in column zero,
    // contains duplicate same-cell entries, and has ignored remote columns.
    for(std::size_t group = 0; group < block_size; ++group) {
        std::size_t const row = group;
        double row_sum = 0;
        if(group == 0) {
            append(row, 0, 0);
            append(row, 1, 1);
            append(row, 1, 1);
            row_sum = 2;
        }
        else if(group == 1) {
            append(row, 0, 3);
            append(row, 1, 4);
            row_sum = 7;
        }
        else {
            append(row, row, 4);
            append(row, row, 0.5);
            append(row, row - 1, -0.25);
            row_sum = 4.25;
        }
        append(row, row_count + group, 123);
        inverse_diagonal[row] = 0.25;
        input[row] = row_sum;
    }

    // Block 1 is rank deficient but has nonzero row scales.  It must take the
    // unsafe-pivot scalar fallback for the whole cell block.
    for(std::size_t group = 0; group < block_size; ++group) {
        std::size_t const row = block_size + group;
        append(row, block_size, 1);
        append(row, row_count + row, -91);
        inverse_diagonal[row] = 0.01 * static_cast<double>(group + 1);
        input[row] = 2 + static_cast<double>(group);
    }

    // Block 2 overflows only while duplicate finite entries are accumulated.
    // It must fall back without allowing a nonfinite factor to escape.
    for(std::size_t group = 0; group < block_size; ++group) {
        std::size_t const row = 2 * block_size + group;
        if(group == 0) {
            append(row, row, std::numeric_limits<double>::max());
            append(row, row, std::numeric_limits<double>::max());
        }
        else
            append(row, row, 2);
        append(row, row_count + row, 37);
        inverse_diagonal[row] = 0.02 * static_cast<double>(group + 1);
        input[row] = 3 + static_cast<double>(group);
    }

    std::vector<std::size_t> row_offsets(row_count + 1, 0);
    std::vector<CG::matrix_index_t> csr_columns;
    std::vector<double> csr_values;
    for(std::size_t row = 0; row < row_count; ++row) {
        row_offsets[row] = csr_values.size();
        csr_columns.insert(csr_columns.end(), columns[row].begin(),
                           columns[row].end());
        csr_values.insert(csr_values.end(), matrix[row].begin(),
                          matrix[row].end());
        row_offsets[row + 1] = csr_values.size();
    }

    CG::CellBlockJacobiPreconditioner row_preconditioner;
    CG::CellBlockJacobiPreconditioner csr_preconditioner;
    std::vector<std::uint32_t> narrow_csr_columns(
        csr_columns.begin(), csr_columns.end());
    CG::CellBlockJacobiPreconditioner narrow_csr_preconditioner;
    require(row_preconditioner.Setup(
                matrix, columns, block_size,
                CG::PreconditionerKind::CellBlockJacobi,
                inverse_diagonal) &&
            csr_preconditioner.SetupCSR(
                row_offsets, csr_columns, csr_values, block_size,
                CG::PreconditionerKind::CellBlockJacobi,
                inverse_diagonal) &&
            narrow_csr_preconditioner.SetupCSR(
                row_offsets, narrow_csr_columns, csr_values, block_size,
                CG::PreconditionerKind::CellBlockJacobi,
                inverse_diagonal),
            "16-group row/CSR block preconditioner setup failed");
    require(row_preconditioner.BlockCount() == block_count &&
            row_preconditioner.FactorizedBlockCount() == 1 &&
            row_preconditioner.FallbackBlockCount() == 2 &&
            row_preconditioner.FirstFallbackBlock() == 1 &&
            row_preconditioner.FirstFallbackReason() ==
                CG::CellBlockFallbackReason::UnsafePivot &&
            csr_preconditioner.FactorizedBlockCount() ==
                row_preconditioner.FactorizedBlockCount() &&
            csr_preconditioner.FallbackBlockCount() ==
                row_preconditioner.FallbackBlockCount() &&
            csr_preconditioner.FirstFallbackBlock() ==
                row_preconditioner.FirstFallbackBlock() &&
            csr_preconditioner.FirstFallbackReason() ==
                row_preconditioner.FirstFallbackReason() &&
            narrow_csr_preconditioner.FactorizedBlockCount() ==
                row_preconditioner.FactorizedBlockCount() &&
            narrow_csr_preconditioner.FallbackBlockCount() ==
                row_preconditioner.FallbackBlockCount() &&
            narrow_csr_preconditioner.FirstFallbackReason() ==
                row_preconditioner.FirstFallbackReason(),
            "16-group CSR block diagnostics diverged from row storage");

    std::vector<double> row_output;
    std::vector<double> csr_output;
    std::vector<double> narrow_csr_output;
    row_preconditioner.Apply(input, row_output);
    csr_preconditioner.Apply(input, csr_output);
    narrow_csr_preconditioner.Apply(input, narrow_csr_output);
    require(row_output.size() == row_count &&
            csr_output.size() == row_count &&
            narrow_csr_output.size() == row_count,
            "16-group block preconditioner returned the wrong size");
    for(std::size_t row = 0; row < row_count; ++row) {
        require(close(row_output[row], csr_output[row], 2e-14),
                "16-group CSR and row block solves differ");
        require(close(row_output[row], narrow_csr_output[row], 2e-14),
                "16-group narrow CSR and row block solves differ");
        double const expected = row < block_size ? 1 :
            inverse_diagonal[row] * input[row];
        require(close(csr_output[row], expected, 2e-13),
                "16-group block solve or scalar fallback is incorrect");
    }

    std::vector<double> invalid_values = csr_values;
    invalid_values.front() = std::numeric_limits<double>::quiet_NaN();
    CG::CellBlockJacobiPreconditioner invalid_preconditioner;
    require(!invalid_preconditioner.SetupCSR(
                row_offsets, csr_columns, invalid_values, block_size,
                CG::PreconditionerKind::CellBlockJacobi,
                inverse_diagonal),
            "CSR block setup accepted a nonfinite matrix entry");
}

void testCellBlockForwardGaussSeidel()
{
    std::size_t constexpr block_size = 2;
    std::vector<std::vector<std::size_t>> row_columns(6);
    std::vector<std::vector<double>> row_values(6);
    auto append = [&](std::size_t const row, std::size_t const column,
                      double const value)
    {
        row_columns[row].push_back(column);
        row_values[row].push_back(value);
    };

    append(0, 0, 4); append(0, 1, 1); append(0, 2, 0.6); append(0, 6, 9);
    append(1, 1, 3); append(1, 0, 1); append(1, 3, -0.4); append(1, 7, -8);
    append(2, 2, 5); append(2, 3, 0.5); append(2, 0, 0.4); append(2, 4, -0.2);
    append(3, 3, 2); append(3, 2, 0.5); append(3, 1, -0.2); append(3, 5, 0.3);
    append(4, 4, 3); append(4, 5, -0.25); append(4, 0, 0.1); append(4, 2, 0.2);
    append(5, 5, 4); append(5, 4, -0.25); append(5, 1, -0.3); append(5, 3, 0.15);

    std::vector<std::size_t> row_offsets(7, 0), columns;
    std::vector<double> values;
    for(std::size_t row = 0; row < row_columns.size(); ++row)
    {
        columns.insert(columns.end(), row_columns[row].begin(),
                       row_columns[row].end());
        values.insert(values.end(), row_values[row].begin(),
                      row_values[row].end());
        row_offsets[row + 1] = columns.size();
    }

    CG::CellBlockJacobiPreconditioner preconditioner;
    require(preconditioner.SetupCSR(
                row_offsets, columns, values, block_size,
                CG::PreconditionerKind::CellBlockGaussSeidel,
                {0.25, 1.0 / 3, 0.2, 0.5, 1.0 / 3, 0.25}),
            "forward block-Gauss-Seidel setup failed");
    require(preconditioner.Kind() ==
                CG::PreconditionerKind::CellBlockGaussSeidel &&
            preconditioner.RequestedKindSupported() &&
            preconditioner.LocalLowerCouplingCount() == 6 &&
            preconditioner.IgnoredLocalUpperCouplingCount() == 4 &&
            preconditioner.IgnoredRemoteCouplingCount() == 2 &&
            CG::CellBlockJacobiSweepCount(preconditioner.Kind()) == 1 &&
            !CG::UsesCellBlockNeighborCorrection(preconditioner.Kind()),
            "forward block-Gauss-Seidel structure accounting is wrong");

    auto solve2 = [](double const a, double const b, double const c,
                     double const d, double const r0, double const r1)
    {
        double const determinant = a * d - b * c;
        return std::array<double, 2>{{
            (d * r0 - b * r1) / determinant,
            (a * r1 - c * r0) / determinant}};
    };
    std::vector<double> const input = {1, 2, 3, 4, 5, 6};
    std::array<double, 2> const x0 = solve2(4, 1, 1, 3, input[0], input[1]);
    std::array<double, 2> const x1 = solve2(
        5, 0.5, 0.5, 2, input[2] - 0.4 * x0[0],
        input[3] + 0.2 * x0[1]);
    std::array<double, 2> const x2 = solve2(
        3, -0.25, -0.25, 4,
        input[4] - 0.1 * x0[0] - 0.2 * x1[0],
        input[5] + 0.3 * x0[1] - 0.15 * x1[1]);
    std::vector<double> const expected = {
        x0[0], x0[1], x1[0], x1[1], x2[0], x2[1]};
    std::vector<double> output;
    preconditioner.ApplyCSRForwardGaussSeidel(
        input, output, row_offsets, columns, values);
    for(std::size_t row = 0; row < output.size(); ++row)
        require(close(output[row], expected[row], 2e-14),
                "forward block-Gauss-Seidel differs from dense reference");

    std::vector<double> aliased = input;
    preconditioner.ApplyCSRForwardGaussSeidel(
        aliased, aliased, row_offsets, columns, values);
    for(std::size_t row = 0; row < aliased.size(); ++row)
        require(close(aliased[row], expected[row], 2e-14),
                "forward block-Gauss-Seidel is not alias safe");

    std::vector<double> second = {-1, 0.5, 2, -3, 1.5, 0.25};
    std::vector<double> combined(input.size()), second_output, combined_output;
    for(std::size_t row = 0; row < combined.size(); ++row)
        combined[row] = input[row] + 2 * second[row];
    preconditioner.ApplyCSRForwardGaussSeidel(
        second, second_output, row_offsets, columns, values);
    preconditioner.ApplyCSRForwardGaussSeidel(
        combined, combined_output, row_offsets, columns, values);
    for(std::size_t row = 0; row < combined_output.size(); ++row)
        require(close(combined_output[row],
                      output[row] + 2 * second_output[row], 3e-14),
                "forward block-Gauss-Seidel is not linear");

    std::vector<std::size_t> invalid_columns = columns;
    for(std::size_t entry = row_offsets[2]; entry < row_offsets[3]; ++entry)
        if(invalid_columns[entry] == 0)
            invalid_columns[entry] = 1;
    CG::CellBlockJacobiPreconditioner unsupported;
    require(unsupported.SetupCSR(
                row_offsets, invalid_columns, values, block_size,
                CG::PreconditionerKind::CellBlockGaussSeidel,
                {0.25, 1.0 / 3, 0.2, 0.5, 1.0 / 3, 0.25}) &&
            !unsupported.RequestedKindSupported() &&
            unsupported.Kind() == CG::PreconditionerKind::CellBlockJacobi,
            "malformed block-Gauss-Seidel layout did not fail closed");

    CG::CellBlockJacobiPreconditioner empty;
    require(empty.SetupCSR(
                std::vector<std::size_t>{0}, std::vector<std::size_t>{},
                std::vector<double>{}, block_size,
                CG::PreconditionerKind::CellBlockGaussSeidel, {}) &&
            empty.RequestedKindSupported(),
            "zero-owned block-Gauss-Seidel setup failed");
    std::vector<double> empty_output;
    empty.ApplyCSRForwardGaussSeidel(
        {}, empty_output, {0}, {}, {});
    require(empty_output.empty(),
            "zero-owned block-Gauss-Seidel produced output");
}

void testRankLocalILU0()
{
    std::vector<std::size_t> const row_offsets = {0, 3, 6, 10, 12};
    std::vector<std::size_t> const columns = {
        0, 1, 4,
        0, 1, 2,
        1, 2, 3, 5,
        2, 3};
    std::vector<double> const values = {
        4, 1, 9,
        2, 3, 1,
        1, 2, 1, -7,
        1, 2};
    std::vector<double> const inverse_diagonal = {0.25, 1.0 / 3, 0.5, 0.5};

    CG::CellBlockJacobiPreconditioner preconditioner;
    require(preconditioner.SetupCSR(
                row_offsets, columns, values, 1,
                CG::PreconditionerKind::RankLocalILU0,
                inverse_diagonal) &&
            preconditioner.Kind() == CG::PreconditionerKind::RankLocalILU0 &&
            preconditioner.RequestedKindSupported() &&
            preconditioner.LocalLowerCouplingCount() == 3 &&
            preconditioner.IgnoredLocalUpperCouplingCount() == 3 &&
            preconditioner.IgnoredRemoteCouplingCount() == 2,
            "rank-local ILU(0) setup or structure accounting failed");

    std::vector<double> const right_hand_side = {6, 11, 12, 11};
    std::vector<double> output;
    preconditioner.ApplyCSRRankLocalILU0(
        right_hand_side, output, row_offsets, columns);
    std::vector<double> const expected = {1, 2, 3, 4};
    for(std::size_t row = 0; row < output.size(); ++row)
        require(close(output[row], expected[row], 2e-14),
                "rank-local ILU(0) differs from the exact tridiagonal solve");

    std::vector<double> aliased = right_hand_side;
    preconditioner.ApplyCSRRankLocalILU0(
        aliased, aliased, row_offsets, columns);
    for(std::size_t row = 0; row < aliased.size(); ++row)
        require(close(aliased[row], expected[row], 2e-14),
                "rank-local ILU(0) is not alias safe");

    std::vector<double> const second = {1, -2, 0.5, 3};
    std::vector<double> combined(second.size()), second_output, combined_output;
    for(std::size_t row = 0; row < combined.size(); ++row)
        combined[row] = right_hand_side[row] - 3 * second[row];
    preconditioner.ApplyCSRRankLocalILU0(
        second, second_output, row_offsets, columns);
    preconditioner.ApplyCSRRankLocalILU0(
        combined, combined_output, row_offsets, columns);
    for(std::size_t row = 0; row < combined_output.size(); ++row)
        require(close(combined_output[row],
                      output[row] - 3 * second_output[row], 3e-14),
                "rank-local ILU(0) is not linear");

    std::vector<std::size_t> duplicate_columns = columns;
    duplicate_columns[2] = 1;
    CG::CellBlockJacobiPreconditioner duplicate;
    require(duplicate.SetupCSR(
                row_offsets, duplicate_columns, values, 1,
                CG::PreconditionerKind::RankLocalILU0,
                inverse_diagonal) &&
            !duplicate.RequestedKindSupported() &&
            duplicate.Kind() == CG::PreconditionerKind::CellBlockJacobi,
            "rank-local ILU(0) duplicate-column layout did not fail closed");

    CG::CellBlockJacobiPreconditioner unsafe_pivot;
    require(unsafe_pivot.SetupCSR(
                std::vector<std::size_t>{0, 2, 4},
                std::vector<std::size_t>{0, 1, 0, 1},
                std::vector<double>{1, 1, 1, 1}, 1,
                CG::PreconditionerKind::RankLocalILU0, {1, 1}) &&
            !unsafe_pivot.RequestedKindSupported() &&
            unsafe_pivot.Kind() == CG::PreconditionerKind::CellBlockJacobi,
            "rank-local ILU(0) unsafe pivot did not fail closed");

    CG::CellBlockJacobiPreconditioner empty;
    require(empty.SetupCSR(
                std::vector<std::size_t>{0}, std::vector<std::size_t>{},
                std::vector<double>{}, 1,
                CG::PreconditionerKind::RankLocalILU0, {}) &&
            empty.RequestedKindSupported(),
            "zero-owned rank-local ILU(0) setup failed");
    std::vector<double> empty_output;
    empty.ApplyCSRRankLocalILU0({}, empty_output, {0}, {});
    require(empty_output.empty(),
            "zero-owned rank-local ILU(0) produced output");
}

#ifdef RICH_MPI
void testPermutationAwareAllActiveMapping()
{
    std::vector<ComputationalCell3D> canonical_cells(3);
    canonical_cells[0].ID = 101;
    canonical_cells[1].ID = 202;
    canonical_cells[2].ID = 303;

    std::vector<ComputationalCell3D> owned_cells(3);
    owned_cells[0].ID = 303;
    owned_cells[1].ID = 101;
    owned_cells[2].ID = 202;
    auto const permutation =
        RadiationDriverTestHooks::ProbeOwnedCanonicalMapping(
            owned_cells, canonical_cells, {2, 0, 1});
    require(permutation.valid && !permutation.identity,
            "all-active mapping rejected a valid owned permutation");

    auto const identity =
        RadiationDriverTestHooks::ProbeOwnedCanonicalMapping(
            canonical_cells, canonical_cells, {0, 1, 2});
    require(identity.valid && identity.identity,
            "all-active mapping rejected the identity mapping");

    auto const invalid =
        RadiationDriverTestHooks::ProbeOwnedCanonicalMapping(
            owned_cells, canonical_cells, {2, 0, 3});
    require(!invalid.valid,
            "all-active mapping accepted an out-of-range canonical index");

    IndividualStepContext activity;
    activity.active_indices = {2, 0, 1};
    activity.active_mask = {1, 1, 1};
    require(RadiationDriverTestHooks::ProbeCompleteOwnedActivity(activity, 3),
            "all-active dispatch rejected a valid scheduler permutation");
    activity.active_indices = {0, 0, 2};
    require(!RadiationDriverTestHooks::ProbeCompleteOwnedActivity(activity, 3),
            "all-active dispatch accepted duplicate scheduler indices");
    activity.active_indices = {0, 1, 3};
    require(!RadiationDriverTestHooks::ProbeCompleteOwnedActivity(activity, 3),
            "all-active dispatch accepted an out-of-range scheduler index");

    IndividualStepContext interval;
    interval.previous_event_tick = std::uint64_t{640} << 40;
    interval.event_tick = interval.previous_event_tick +
        (std::uint64_t{1} << 40);
    interval.time_quantum = 2.8475356730643257e-13;
    interval.previous_event_time = interval.time_quantum *
        static_cast<double>(interval.previous_event_tick);
    interval.event_time = interval.time_quantum *
        static_cast<double>(interval.event_tick);
    double const tick_interval = interval.time_quantum *
        static_cast<double>(interval.event_tick - interval.previous_event_tick);
    auto const valid_interval =
        RadiationDriverTestHooks::ProbeSchedulerEventInterval(
            interval, tick_interval);
    require(valid_interval.valid && valid_interval.matches_cell_interval &&
            valid_interval.interval == tick_interval,
            "all-active interval did not use the exact scheduler tick difference");
    require(!RadiationDriverTestHooks::ProbeSchedulerEventInterval(
                interval, 2 * tick_interval).matches_cell_interval,
            "all-active interval accepted a factor-two timestep mismatch");
    interval.time_quantum = 0;
    require(!RadiationDriverTestHooks::ProbeSchedulerEventInterval(
                interval, 0).valid,
            "all-active interval accepted a zero scheduler quantum");
    interval.time_quantum = std::numeric_limits<double>::quiet_NaN();
    require(!RadiationDriverTestHooks::ProbeSchedulerEventInterval(
                interval, 0).valid,
            "all-active interval accepted a nonfinite scheduler quantum");
    interval.previous_event_tick = 0;
    interval.event_tick = 2;
    interval.time_quantum = std::numeric_limits<double>::max();
    require(!RadiationDriverTestHooks::ProbeSchedulerEventInterval(
                interval, 0).valid,
            "all-active interval accepted an overflowing scheduler duration");
}

void testDistributedActiveWideIdExchangeMPI()
{
    int rank = 0;
    int rank_count = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &rank_count);
    if(rank_count < 2)
        return;

    std::size_t const wide_id_base =
        (std::size_t{1} << 32) + std::size_t{700001};
    auto const narrow_column_probe =
        RadiationDriverTestHooks::ProbeDistributedCSRColumns(
            {0, 17,
             static_cast<std::size_t>(
                 std::numeric_limits<std::uint32_t>::max())});
    require(narrow_column_probe.uses_narrow_columns &&
            narrow_column_probe.columns ==
                std::vector<std::size_t>({0, 17,
                    static_cast<std::size_t>(
                        std::numeric_limits<std::uint32_t>::max())}),
            "distributed CSR did not retain checked 32-bit column slots");
    auto const wide_column_probe =
        RadiationDriverTestHooks::ProbeDistributedCSRColumns(
            {0,
             static_cast<std::size_t>(
                 std::numeric_limits<std::uint32_t>::max()) + 1,
             wide_id_base});
    require(!wide_column_probe.uses_narrow_columns &&
            wide_column_probe.columns ==
                std::vector<std::size_t>({0,
                    static_cast<std::size_t>(
                        std::numeric_limits<std::uint32_t>::max()) + 1,
                    wide_id_base}),
            "distributed CSR wide-column fallback truncated an index");
    std::vector<ComputationalCell3D> owned_active_cells;
    std::vector<double> local_values;
    std::vector<int> remote_owners;
    std::vector<std::size_t> remote_cell_ids;
    std::vector<std::vector<double> > matrix;
    std::vector<std::vector<std::size_t> > columns;
    if(rank < 2) {
        for(std::size_t local = 0; local < 2; ++local) {
            ComputationalCell3D cell;
            cell.ID = wide_id_base + 2 * static_cast<std::size_t>(rank) +
                local;
            owned_active_cells.push_back(cell);
        }
        local_values.push_back(static_cast<double>(rank + 1));
        local_values.push_back(static_cast<double>(10 * (rank + 1)));
        remote_owners.push_back(1 - rank);
        remote_cell_ids.push_back(
            wide_id_base + 2 * static_cast<std::size_t>(1 - rank));
        remote_owners.push_back(1 - rank);
        remote_cell_ids.push_back(
            wide_id_base + 2 * static_cast<std::size_t>(1 - rank) + 1);
        matrix.push_back(std::vector<double>{2.0});
        columns.push_back(std::vector<std::size_t>{0});
        matrix.push_back(std::vector<double>{2.0, 3.0, 5.0});
        columns.push_back(std::vector<std::size_t>{1, 2, 3});
    }
    RadiationDriverTestHooks::DistributedActiveMatVecProbeResult const probe =
        RadiationDriverTestHooks::ProbeDistributedActiveMatVec(
            owned_active_cells, local_values, remote_owners,
            remote_cell_ids, matrix, columns, 1);
    require(probe.exchange_initialized && probe.multiplied &&
            probe.second_multiply_succeeded &&
            probe.exchange_storage_reused && probe.global_size == 4 &&
            probe.csr_uses_narrow_columns &&
            (rank >= 2 || (probe.request_send_chunks == 2 &&
                           probe.request_receive_chunks == 2)) &&
            probe.exchange_validity_reductions == 2,
            "wide-ID chunked distributed-active exchange did not initialize");
    if(rank == 0) {
        require(probe.output.size() == 2 && close(probe.output[0], 2.0) &&
                close(probe.output[1], 126.0) &&
                probe.second_output.size() == 2 &&
                close(probe.second_output[0], 4.0) &&
                close(probe.second_output[1], 136.0),
                "rank 0 distributed matvec lost the wide-ID remote value");
        require(probe.csr_nonzeros == 4 && probe.local_row_count == 1 &&
                probe.remote_row_count == 1,
                "rank 0 CSR rows were not split for communication overlap");
    }
    else if(rank == 1) {
        require(probe.output.size() == 2 && close(probe.output[0], 4.0) &&
                close(probe.output[1], 93.0) &&
                probe.second_output.size() == 2 &&
                close(probe.second_output[0], 6.0) &&
                close(probe.second_output[1], 103.0),
                "rank 1 distributed matvec lost the wide-ID remote value");
        require(probe.csr_nonzeros == 4 && probe.local_row_count == 1 &&
                probe.remote_row_count == 1,
                "rank 1 CSR rows were not split for communication overlap");
    }
    else
        require(probe.output.empty() && probe.second_output.empty() &&
                probe.csr_nonzeros == 0 && probe.local_row_count == 0 &&
                probe.remote_row_count == 0,
                "zero-owned rank acquired a distributed-active output row");

    owned_active_cells.clear();
    local_values.clear();
    matrix.clear();
    columns.clear();
    remote_owners.clear();
    remote_cell_ids.clear();
    if(rank == 0) {
        ComputationalCell3D cell;
        cell.ID = wide_id_base + 10;
        owned_active_cells.push_back(cell);
        local_values.push_back(2.0);
        matrix.push_back(std::vector<double>{4.0});
        columns.push_back(std::vector<std::size_t>{0});
    }
    RadiationDriverTestHooks::DistributedActiveMatVecProbeResult const
        zero_owned_probe =
            RadiationDriverTestHooks::ProbeDistributedActiveMatVec(
                owned_active_cells, local_values, remote_owners,
                remote_cell_ids, matrix, columns);
    require(zero_owned_probe.exchange_initialized &&
            zero_owned_probe.multiplied &&
            zero_owned_probe.second_multiply_succeeded &&
            zero_owned_probe.exchange_storage_reused &&
            zero_owned_probe.exchange_validity_reductions == 2 &&
            zero_owned_probe.global_size == 1,
            "distributed-active zero-owned-rank probe failed");
    if(rank == 0)
        require(zero_owned_probe.output.size() == 1 &&
                close(zero_owned_probe.output[0], 8.0) &&
                zero_owned_probe.second_output.size() == 1 &&
                close(zero_owned_probe.second_output[0], 12.0) &&
                zero_owned_probe.local_row_count == 1 &&
                zero_owned_probe.remote_row_count == 0,
                "owned row changed in the zero-owned-rank probe");
    else
        require(zero_owned_probe.output.empty() &&
                zero_owned_probe.second_output.empty() &&
                zero_owned_probe.local_row_count == 0 &&
                zero_owned_probe.remote_row_count == 0,
                "zero-owned rank was not safe in distributed matvec");
}

void testDistributedActiveExchangeFailuresMPI()
{
    int rank = 0;
    int rank_count = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &rank_count);
    if(rank_count < 2)
        return;

    std::size_t const base =
        (std::size_t{1} << 32) + std::size_t{900001};
    std::vector<ComputationalCell3D> cells;
    std::vector<double> values;
    std::vector<int> owners;
    std::vector<std::size_t> remote_ids;
    std::vector<std::vector<double> > matrix;
    std::vector<std::vector<std::size_t> > columns;

    if(rank == 0) {
        ComputationalCell3D first;
        first.ID = base;
        ComputationalCell3D duplicate = first;
        cells = {first, duplicate};
        values = {1, 2};
        matrix = {{1}, {1}};
        columns = {{0}, {1}};
    }
    auto duplicate_probe =
        RadiationDriverTestHooks::ProbeDistributedActiveMatVec(
            cells, values, owners, remote_ids, matrix, columns, 1);
    require(!duplicate_probe.exchange_initialized,
            "distributed request setup accepted duplicate owned cell IDs");

    cells.clear();
    values.clear();
    owners.clear();
    remote_ids.clear();
    matrix.clear();
    columns.clear();
    if(rank < 2) {
        ComputationalCell3D cell;
        cell.ID = base + static_cast<std::size_t>(rank);
        cells.push_back(cell);
        values.push_back(1 + rank);
        matrix.push_back({1});
        columns.push_back({0});
    }
    if(rank == 0) {
        owners.push_back(1);
        remote_ids.push_back(base + 99);
        matrix[0].push_back(1);
        columns[0].push_back(1);
    }
    auto missing_probe =
        RadiationDriverTestHooks::ProbeDistributedActiveMatVec(
            cells, values, owners, remote_ids, matrix, columns, 1);
    require(!missing_probe.exchange_initialized,
            "distributed request setup accepted a missing remote cell ID");

    cells.clear();
    values.clear();
    owners.clear();
    remote_ids.clear();
    matrix.clear();
    columns.clear();
    if(rank == 0) {
        ComputationalCell3D cell;
        cell.ID = base;
        cells.push_back(cell);
        values.push_back(1);
        owners.push_back(rank_count);
        remote_ids.push_back(base + 1);
        matrix.push_back({1});
        columns.push_back({1});
    }
    auto invalid_owner_probe =
        RadiationDriverTestHooks::ProbeDistributedActiveMatVec(
            cells, values, owners, remote_ids, matrix, columns, 1);
    require(!invalid_owner_probe.exchange_initialized,
            "distributed request setup accepted an invalid owner rank");

    cells.clear();
    values.clear();
    owners.clear();
    remote_ids.clear();
    matrix.clear();
    columns.clear();
    if(rank < 2) {
        ComputationalCell3D cell;
        cell.ID = base + static_cast<std::size_t>(rank);
        cells.push_back(cell);
        values.push_back(rank == 0 ?
            std::numeric_limits<double>::quiet_NaN() : 2.0);
        owners.push_back(1 - rank);
        remote_ids.push_back(base + static_cast<std::size_t>(1 - rank));
        matrix.push_back({1});
        columns.push_back({1});
    }
    auto nonfinite_probe =
        RadiationDriverTestHooks::ProbeDistributedActiveMatVec(
            cells, values, owners, remote_ids, matrix, columns, 1);
    require(nonfinite_probe.exchange_initialized &&
            !nonfinite_probe.multiplied &&
            !nonfinite_probe.second_multiply_succeeded,
            "distributed matvec accepted an exchanged nonfinite value");

    if(rank < 2)
        values[0] = 1 + rank;
    int const inconsistent_tag = rank == 0 ? 21058 : 21059;
    auto inconsistent_tag_probe =
        RadiationDriverTestHooks::ProbeDistributedActiveMatVec(
            cells, values, owners, remote_ids, matrix, columns, 1,
            inconsistent_tag);
    require(!inconsistent_tag_probe.exchange_initialized,
            "distributed request setup accepted rank-inconsistent tags");
    auto invalid_tag_probe =
        RadiationDriverTestHooks::ProbeDistributedActiveMatVec(
            cells, values, owners, remote_ids, matrix, columns, 1, -2);
    require(!invalid_tag_probe.exchange_initialized,
            "distributed request setup accepted a negative tag");
}

void testDistributedActiveLargeCountHelpersMPI()
{
    std::uint64_t const wide_total64 =
        (std::uint64_t{1} << 32) + std::uint64_t{17};
    require(wide_total64 <=
                static_cast<std::uint64_t>(
                    std::numeric_limits<std::size_t>::max()),
            "distributed-active large-count support requires 64-bit size_t");
    std::size_t const wide_total = static_cast<std::size_t>(wide_total64);
    auto const chunks =
        RadiationDriverTestHooks::ProbeDistributedActiveTransferChunks(
            wide_total);
    std::size_t const maximum_chunk =
        static_cast<std::size_t>(std::numeric_limits<int>::max());
    require(chunks.valid &&
                chunks.offsets == std::vector<std::size_t>(
                    {0, maximum_chunk, 2 * maximum_chunk}) &&
                chunks.counts == std::vector<std::size_t>(
                    {maximum_chunk, maximum_chunk, 19}),
            "2^32+17 distributed values were not segmented exactly");

    auto const wide_layout =
        RadiationDriverTestHooks::ProbeDistributedActiveSizeArithmetic(
            wide_total, 16);
    require(wide_layout.multiply_valid &&
                wide_layout.product == wide_total * 16 &&
                wide_layout.product > wide_total,
            "wide distributed layout multiplication was truncated");
    auto const overflow =
        RadiationDriverTestHooks::ProbeDistributedActiveSizeArithmetic(
            std::numeric_limits<std::size_t>::max() - 3, 4);
    require(!overflow.add_valid && !overflow.multiply_valid,
            "distributed layout arithmetic accepted size_t overflow");

    auto const enabled =
        RadiationDriverTestHooks::ProbeDistributedActiveToggle("YES", false);
    auto const disabled =
        RadiationDriverTestHooks::ProbeDistributedActiveToggle("off", true);
    auto const invalid =
        RadiationDriverTestHooks::ProbeDistributedActiveToggle(
            "sometimes", true);
    require(enabled.valid && enabled.value && disabled.valid &&
                !disabled.value && !invalid.valid && invalid.value,
            "distributed runtime toggle parser accepted an ambiguous value");
    require(!RadiationDriverTestHooks::
                ProbeDistributedActiveOverlapDefault(),
            "distributed active overlap did not default to the measured "
            "simpler path");
}
#endif

void testHistoricalFinalCorrectionGate()
{
    CG::HistoricalMGCorrectionAssessment finite;
    finite.finite = true;
    finite.negative_group_count = 2;
    finite.negative_extent = 3.5;
    require(CG::ClassifyHistoricalMGCorrection(finite) ==
                CG::HistoricalMGCorrectionDisposition::Commit,
            "finite correction did not commit immediately");

    CG::HistoricalMGCorrectionAssessment nonfinite;
    nonfinite.finite = false;
    require(CG::ClassifyHistoricalMGCorrection(nonfinite) ==
                CG::HistoricalMGCorrectionDisposition::RejectNonFinite,
            "non-finite correction was not rejected");

    std::vector<double> const pre{100, 1};
    std::vector<double> const unscaled_post{20, -10};
    auto const limits =
        CG::DetermineHistoricalMGResidualCorrectionLimits(
            pre, unscaled_post, 17);
    require(limits.size() == 2 &&
                CG::HistoricalMGResidualCorrectionLimited(limits[0]) &&
                CG::HistoricalMGResidualCorrectionLimited(limits[1]),
            "local residual-correction limits were not created");
    double const applied0 =
        CG::AppliedHistoricalMGResidualCorrectionExtent(limits[0]);
    double const applied1 =
        CG::AppliedHistoricalMGResidualCorrectionExtent(limits[1]);
    require(std::abs(limits[0].Scale - 0.625) < 1e-14 &&
                std::abs(limits[1].Scale - 0.5 / 11.0) < 1e-14 &&
                std::abs(limits[0].Scale - limits[1].Scale) > 0.5,
            "residual correction did not use an independent scale per group");
    require(applied0 >= 0.5 * pre[0] &&
                applied1 >= 0.5 * pre[1],
            "residual correction reduced a positive group by more than half");

    auto const unchanged_neighbor =
        CG::DetermineHistoricalMGResidualCorrectionLimit(
            8.0, 7.0, 17, 2);
    require(!CG::HistoricalMGResidualCorrectionLimited(unchanged_neighbor) &&
                unchanged_neighbor.Scale == 1.0,
            "one limited group scaled an unrelated group");

    auto const tiny_tail =
        CG::DetermineHistoricalMGResidualCorrectionLimit(
            1e-200, -1.0, 18, 3);
    require(CG::HistoricalMGResidualCorrectionLimited(tiny_tail) &&
                CG::AppliedHistoricalMGResidualCorrectionExtent(tiny_tail) >=
                    0.5e-200,
            "strictly positive tiny tail did not retain half its extent");

    auto const zero_start =
        CG::DetermineHistoricalMGResidualCorrectionLimit(
            0.0, -10.0, 19, 4);
    auto const negative_start =
        CG::DetermineHistoricalMGResidualCorrectionLimit(
            -1.0, -10.0, 19, 5);
    require(zero_start.Scale == 1.0 && negative_start.Scale == 1.0,
            "zero or already-negative group incorrectly constrained its correction");

    std::vector<double> const no_total_pre{100, -90};
    std::vector<double> const no_total_post{50, -90};
    auto const no_total_limits =
        CG::DetermineHistoricalMGResidualCorrectionLimits(
            no_total_pre, no_total_post, 20);
    require(no_total_limits.size() == 2 &&
                no_total_limits[0].Scale == 1.0 &&
                no_total_limits[1].Scale == 1.0 &&
                no_total_post[0] + no_total_post[1] < 0,
            "a cell-total cap was applied despite the per-group-only policy");

    auto const invalid_limit =
        CG::DetermineHistoricalMGResidualCorrectionLimit(
            std::numeric_limits<double>::quiet_NaN(), 1.0, 21, 0);
    require(!invalid_limit.Finite,
            "non-finite residual-correction input was accepted");

    CG::HistoricalMGResidualCorrectionDiagnostics diagnostics;
    CG::ResetHistoricalMGResidualCorrectionDiagnostics(diagnostics, 2, 2);
    CG::RecordHistoricalMGResidualCorrectionLimit(limits[0], diagnostics);
    CG::RecordHistoricalMGResidualCorrectionLimit(limits[1], diagnostics);
    require(diagnostics.available && diagnostics.finite &&
                diagnostics.limited_group_count == 2 &&
                diagnostics.signed_energy_bias_by_group.size() == 2 &&
                diagnostics.absolute_energy_bias_by_group.size() == 2,
            "residual-correction accounting shape is invalid");
    double const expected_bias0 = applied0 - unscaled_post[0];
    double const expected_bias1 = applied1 - unscaled_post[1];
    require(std::abs(diagnostics.signed_energy_bias_by_group[0] -
                     expected_bias0) < 1e-12 &&
                std::abs(diagnostics.signed_energy_bias_by_group[1] -
                         expected_bias1) < 1e-12 &&
                std::abs(diagnostics.signed_energy_bias -
                         (expected_bias0 + expected_bias1)) < 1e-12 &&
                std::abs(diagnostics.absolute_energy_bias -
                         (std::abs(expected_bias0) +
                          std::abs(expected_bias1))) < 1e-12 &&
                diagnostics.minimum_scale == limits[1].Scale &&
                diagnostics.limiting_cell_id == 17 &&
                diagnostics.limiting_group == 1,
            "residual-correction signed/absolute bias accounting is wrong");

    CG::HistoricalMGCorrectionSpectralFailure failure;
    failure.CausedRejection = true;
    failure.Failure = RadiationPositivity::SpectralRepairFailure::
        NegativeExtentExceedsTolerance;
    failure.CellId = 42;
    failure.Group = 7;
    failure.SignedGroupExtent = -3;
    failure.NegativeExtent = 3;
    failure.PositiveExtent = 10;
    failure.RelativeDeficit = 0.3;
    failure.GlobalMaximumCellExtent = 100;
    failure.Rank = 2;
    failure.PreCorrectionEg = -1;
    failure.ResidualCorrection = -2;
    failure.PostCorrectionEg = -3;
    failure.GlobalMaximumAbsoluteEg = 40;
    failure.NegativeToGlobalMaximumAbsoluteRatio = 0.075;
    failure.GlobalNegativeExtent = 4;
    failure.GlobalPositiveExtent = 2e8;
    failure.GlobalNegativeToPositiveRatio = 2e-8;
    failure.CorrectionScale = 0.25;
    failure.Causality = CG::HistoricalMGResidualCorrectionCausality::
        WorsenedExistingNegativity;
    CG::HistoricalMGPositivityContinuation continuation;
    continuation.Active = true;
    continuation.InitialIteration = 5;
    continuation.AdditionalIterationsUsed = 30;
    continuation.BlocksStarted = 3;
    continuation.BlocksCompleted = 3;
    CG::RecordHistoricalMGResidualCorrectionFailure(
        failure, diagnostics, 35, &continuation);
    require(diagnostics.failure_class == failure.Failure &&
                diagnostics.failure_cell_id == 42 &&
                diagnostics.failure_group == 7 &&
                diagnostics.failure_relative_deficit == 0.3 &&
                diagnostics.failure_rank == 2 &&
                diagnostics.failure_pre_correction_Eg == -1 &&
                diagnostics.failure_residual_correction == -2 &&
                diagnostics.failure_post_correction_Eg == -3 &&
                diagnostics.failure_global_maximum_absolute_Eg == 40 &&
                diagnostics.failure_negative_to_global_maximum_ratio ==
                    0.075 &&
                diagnostics.failure_global_negative_extent == 4 &&
                diagnostics.failure_global_positive_extent == 2e8 &&
                diagnostics.failure_global_negative_to_positive_ratio ==
                    2e-8 &&
                diagnostics.failure_correction_scale == 0.25 &&
                diagnostics.failure_causality == failure.Causality &&
                diagnostics.failure_solver_iterations == 35 &&
                diagnostics.positivity_initial_iteration == 5 &&
                diagnostics.positivity_additional_iterations == 30 &&
                diagnostics.positivity_blocks_started == 3 &&
                diagnostics.positivity_blocks_completed == 3,
            "residual-correction halving diagnostics lost the exact cause");
    std::ostringstream diagnostic_text;
    CG::AppendHistoricalMGResidualCorrectionFailureDiagnostics(
        diagnostic_text, diagnostics, 41);
    std::string const diagnostic_log = diagnostic_text.str();
    require(diagnostic_log.find("Eg_before_residual_correction=-1") !=
                std::string::npos &&
            diagnostic_log.find("residual_correction_added=-2") !=
                std::string::npos &&
            diagnostic_log.find("Eg_after_residual_correction=-3") !=
                std::string::npos &&
            diagnostic_log.find("global_maximum_absolute_Eg=40") !=
                std::string::npos &&
            diagnostic_log.find("positivity_restart_trigger_fraction=1e-10") !=
                std::string::npos &&
            diagnostic_log.find("global_negative_extent=4") !=
                std::string::npos &&
            diagnostic_log.find("global_positive_extent=2e+08") !=
                std::string::npos &&
            diagnostic_log.find("global_negative_to_positive_ratio=2e-08") !=
                std::string::npos &&
            diagnostic_log.find("global_negative_fraction_threshold=1e-08") !=
                std::string::npos &&
            diagnostic_log.find("correction_lambda=0.25") !=
                std::string::npos &&
            diagnostic_log.find(
                "residual_correction_causality=worsened_existing_negativity") !=
                std::string::npos &&
            diagnostic_log.find(
                "positivity_failure_class=negative_extent_exceeds_tolerance") !=
                std::string::npos &&
            diagnostic_log.find(
                "timestep_halving_reason=historical_residual_correction_post_cap_nonphysical") !=
                std::string::npos &&
            diagnostic_log.find("solver_iterations=41") !=
                std::string::npos &&
            diagnostic_log.find("failing_solve_iterations=35") !=
                std::string::npos,
            "residual-correction exceptional log omitted required fields");

    require(!CG::ShouldDeferComptonForResidualCorrectionCausality(false, false) &&
                CG::ShouldDeferComptonForResidualCorrectionCausality(false, true) &&
                !CG::ShouldDeferComptonForResidualCorrectionCausality(true, false) &&
                CG::ShouldDeferComptonForResidualCorrectionCausality(true, true),
            "Compton residual-correction causality table is wrong");
}

void testHistoricalMGPositivityContinuation()
{
    std::vector<std::size_t> const cell_ids{73};
    std::vector<double> const scales{1, 0.25};
    std::vector<double> const volumes{1, 1};

    std::vector<double> const positive_pre{1, 2};
    std::vector<double> positive_post{1.5, 2.5};
    auto const positive = CG::AssessHistoricalMGCorrectedNegativity(
        positive_pre, positive_post, scales, 2, cell_ids, 0, false);
    auto const positive_floor = CG::AssessAndApplyHistoricalMGPositiveFloor(
        positive_pre, positive_post, scales, volumes, 2, cell_ids, 0, 0,
        false);
    CG::HistoricalMGPositivityContinuation positive_continuation;
    require(positive.Finite && !positive.HasNegative &&
            positive_floor.Finite && !positive_floor.HasNegative &&
            CG::EvaluateHistoricalMGPositivityContinuation(
                positive, positive_floor, 5, positive_continuation) ==
                CG::HistoricalMGPositivityContinuationDecision::NoExtension &&
            !positive_continuation.Active,
            "positive converged candidate opened a positivity extension");

    std::vector<double> const threshold_pre{10, 0};
    std::vector<double> threshold_post{10, -1e-9};
    auto const at_threshold = CG::AssessHistoricalMGCorrectedNegativity(
        threshold_pre, threshold_post, scales, 2, cell_ids, 0, false);
    auto const threshold_floor =
        CG::AssessAndApplyHistoricalMGPositiveFloor(
            threshold_pre, threshold_post, scales, volumes, 2, cell_ids,
            0, 0, false);
    CG::HistoricalMGPositivityContinuation threshold_continuation;
    require(at_threshold.HasNegative &&
            at_threshold.NegativeToGlobalMaximumRatio <=
                CG::historical_mg_positivity_continuation_trigger_fraction &&
            threshold_floor.Applied && threshold_floor.Eligible &&
            CG::EvaluateHistoricalMGPositivityContinuation(
                at_threshold, threshold_floor, 5,
                threshold_continuation) ==
                CG::HistoricalMGPositivityContinuationDecision::NoExtension,
            "negative candidate at the 1e-10 trigger boundary opened rescue");

    std::vector<double> below_post{10, -5e-10};
    auto const below_threshold = CG::AssessHistoricalMGCorrectedNegativity(
        threshold_pre, below_post, scales, 2, cell_ids, 0, false);
    auto const below_floor = CG::AssessAndApplyHistoricalMGPositiveFloor(
        threshold_pre, below_post, scales, volumes, 2, cell_ids, 0, 0,
        false);
    CG::HistoricalMGPositivityContinuation below_continuation;
    require(CG::EvaluateHistoricalMGPositivityContinuation(
                below_threshold, below_floor, 5, below_continuation) ==
                CG::HistoricalMGPositivityContinuationDecision::NoExtension,
            "negative candidate below the 1e-10 trigger opened rescue");

    std::vector<double> above_post{10, -1.0001e-9};
    auto const above_threshold = CG::AssessHistoricalMGCorrectedNegativity(
        threshold_pre, above_post, scales, 2, cell_ids, 0, false);
    auto const above_floor = CG::AssessAndApplyHistoricalMGPositiveFloor(
        threshold_pre, above_post, scales, volumes, 2, cell_ids, 0, 0,
        false);
    CG::HistoricalMGPositivityContinuation clears_continuation;
    require(above_threshold.Causality ==
                CG::HistoricalMGResidualCorrectionCausality::
                    CreatedNegativity &&
            CG::EvaluateHistoricalMGPositivityContinuation(
                above_threshold, above_floor, 5,
                clears_continuation) ==
                CG::HistoricalMGPositivityContinuationDecision::Restart &&
            clears_continuation.InitialIteration == 5,
            "negative candidate above 1e-10 did not request a restart");
    require(CG::EvaluateHistoricalMGPositivityContinuation(
                positive, positive_floor, 12, clears_continuation) ==
                CG::HistoricalMGPositivityContinuationDecision::Cleared &&
            clears_continuation.AdditionalIterationsUsed == 7,
            "cleared negativity did not permit early acceptance");

    CG::HistoricalMGPositivityContinuation persistent_continuation;
    require(CG::EvaluateHistoricalMGPositivityContinuation(
                above_threshold, above_floor, 5,
                persistent_continuation) ==
                CG::HistoricalMGPositivityContinuationDecision::Restart,
            "persistent-negativity test did not open at iteration five");
    for(std::size_t iteration = 6; iteration < 35; ++iteration) {
        CG::HistoricalMGPositivityContinuationDecision const expected =
            (iteration - 5) % 10 == 0 ?
            CG::HistoricalMGPositivityContinuationDecision::Restart :
            CG::HistoricalMGPositivityContinuationDecision::Continue;
        require(CG::EvaluateHistoricalMGPositivityContinuation(
                    above_threshold, above_floor, iteration,
                    persistent_continuation) ==
                    expected,
                "persistent rescue did not advance in ten-iteration blocks");
    }
    require(CG::EvaluateHistoricalMGPositivityContinuation(
                above_threshold, above_floor, 35,
                persistent_continuation) ==
                CG::HistoricalMGPositivityContinuationDecision::Exhausted &&
            persistent_continuation.AdditionalIterationsUsed == 30 &&
            persistent_continuation.BlocksStarted == 3 &&
            persistent_continuation.BlocksCompleted == 3 &&
            CG::HistoricalMGPositivityContinuationBudgetReached(
                persistent_continuation, 35),
            "persistent negativity did not exhaust at exactly 30 iterations");
    require(CG::ShouldRestartHistoricalMGPositivityContinuation(
                false, true, false) &&
            !CG::ShouldAttemptHistoricalMGPositivityFinalization(
                false, true, false) &&
            !CG::ShouldRestartHistoricalMGPositivityContinuation(
                false, true, true) &&
            CG::ShouldAttemptHistoricalMGPositivityFinalization(
                false, true, true) &&
            CG::ShouldAttemptHistoricalMGPositivityFinalization(
                true, false, false),
            "positivity continuation did not finalize at budget exhaustion");
    std::ostringstream continuation_log;
    std::streambuf* const previous_clog =
        std::clog.rdbuf(continuation_log.rdbuf());
    CG::ReportHistoricalMGPositivityContinuationOpen(
        "test", persistent_continuation);
    CG::ReportHistoricalMGPositivityContinuationOpen(
        "test", persistent_continuation);
    CG::ReportHistoricalMGPositivityContinuationClose(
        "test", persistent_continuation, above_threshold,
        "budget_exhausted");
    CG::ReportHistoricalMGPositivityContinuationClose(
        "test", persistent_continuation, above_threshold,
        "duplicate_must_not_emit");
    std::clog.rdbuf(previous_clog);
    std::string const continuation_text = continuation_log.str();
    require(continuation_text.find(
                "MG_BICGSTAB_POSITIVITY_CONTINUATION_OPEN") !=
                std::string::npos &&
            continuation_text.find(
                "MG_BICGSTAB_POSITIVITY_CONTINUATION_OPEN") ==
                continuation_text.rfind(
                    "MG_BICGSTAB_POSITIVITY_CONTINUATION_OPEN") &&
            continuation_text.find(
                "MG_BICGSTAB_POSITIVITY_CONTINUATION_CLOSE") !=
                std::string::npos &&
            continuation_text.find(
                "MG_BICGSTAB_POSITIVITY_CONTINUATION_CLOSE") ==
                continuation_text.rfind(
                    "MG_BICGSTAB_POSITIVITY_CONTINUATION_CLOSE") &&
            continuation_text.find("initial_iteration=5") !=
                std::string::npos &&
            continuation_text.find("additional_iterations_used=30") !=
                std::string::npos &&
            continuation_text.find("worst_cell_id=73") !=
                std::string::npos &&
            continuation_text.find("worst_group=1") !=
                std::string::npos &&
            continuation_text.find("outcome=budget_exhausted") !=
                std::string::npos,
            "positivity continuation diagnostics were missing or duplicated");
    require(CG::ShouldCommitHistoricalMGComptonFallback(
                above_threshold,
                CG::HistoricalMGPositivityContinuationDecision::Exhausted,
                true) &&
            !CG::ShouldCommitHistoricalMGComptonFallback(
                above_threshold,
                CG::HistoricalMGPositivityContinuationDecision::Exhausted,
                false) &&
            !CG::ShouldCommitHistoricalMGComptonFallback(
                above_threshold,
                CG::HistoricalMGPositivityContinuationDecision::Continue,
                true) &&
            !CG::ShouldCommitHistoricalMGComptonFallback(
                at_threshold,
                CG::HistoricalMGPositivityContinuationDecision::Exhausted,
                true),
            "Compton fallback did not require exhausted significant negativity");

    double const fallback_scale = 10;
    double const fallback_boundary =
        CG::historical_mg_positivity_continuation_trigger_fraction *
        fallback_scale;
    require(!CG::HistoricalMGNegativeValueRequiresComptonFallback(
                -fallback_boundary, fallback_scale) &&
            !CG::HistoricalMGNegativeValueRequiresComptonFallback(
                -0.5 * fallback_boundary, fallback_scale) &&
            CG::HistoricalMGNegativeValueRequiresComptonFallback(
                -1.0001 * fallback_boundary, fallback_scale) &&
            CG::HistoricalMGCellRequiresComptonFallback(
                std::vector<double>{1, -1.0001 * fallback_boundary},
                fallback_scale) &&
            !CG::HistoricalMGCellRequiresComptonFallback(
                std::vector<double>{1, -fallback_boundary},
                fallback_scale),
            "Compton fallback threshold is not strict at 1e-10 of global max");

    double const local_limit =
        CG::historical_mg_positive_floor_single_cell_fraction;
    double const global_limit =
        CG::historical_mg_positive_floor_global_fraction;
    double const floor_extent =
        RadiationPositivity::spectral_repair_floor_fraction;
    std::vector<double> boundary_post{
        1, -(local_limit - floor_extent)};
    auto const boundary_floor = CG::AssessAndApplyHistoricalMGPositiveFloor(
        std::vector<double>{1, 0}, boundary_post, scales, volumes, 2,
        cell_ids, 1, 9, false);
    require(boundary_floor.Finite && boundary_floor.Eligible &&
            boundary_floor.Applied &&
            boundary_floor.MaximumCellInjectionRatio <= local_limit &&
            boundary_floor.GlobalInjectionRatio <= global_limit,
            "inclusive local/global positive-floor boundaries were rejected");

    std::vector<double> local_failure_post{
        1, -(1.0001 * local_limit - floor_extent)};
    auto const local_failure_floor =
        CG::AssessAndApplyHistoricalMGPositiveFloor(
            std::vector<double>{1, 0}, local_failure_post, scales, volumes,
            2, cell_ids, 1, 9, false);
    auto const local_failure_negativity =
        CG::AssessHistoricalMGCorrectedNegativity(
            std::vector<double>{1, 0},
            std::vector<double>{1,
                -(1.0001 * local_limit - floor_extent)},
            scales, 2, cell_ids, 1e9, false);
    CG::HistoricalMGPositivityContinuation local_failure_continuation;
    require(!local_failure_floor.Eligible &&
            local_failure_floor.Failure ==
                RadiationPositivity::SpectralRepairFailure::
                    SingleCellInjectedEnergyLimit &&
            !CG::HistoricalMGCorrectedNegativityExceedsContinuationThreshold(
                local_failure_negativity) &&
            CG::EvaluateHistoricalMGPositivityContinuation(
                local_failure_negativity, local_failure_floor, 8,
                local_failure_continuation) ==
                CG::HistoricalMGPositivityContinuationDecision::Restart,
            "finite local floor failure did not receive rescue first");

    std::vector<double> global_failure_post{
        1, -(1.0001 * global_limit - floor_extent)};
    auto const global_failure_floor =
        CG::AssessAndApplyHistoricalMGPositiveFloor(
            std::vector<double>{1, 0}, global_failure_post, scales, volumes,
            2, cell_ids, 1, 0, false);
    require(!global_failure_floor.Eligible &&
            global_failure_floor.MaximumCellInjectionRatio < local_limit &&
            global_failure_floor.Failure ==
                RadiationPositivity::SpectralRepairFailure::
                    GlobalInjectedEnergyLimit,
            "global injected-energy floor limit was not enforced");

    std::vector<double> no_positive_post{-1e-12, -2e-12};
    auto const no_positive_floor =
        CG::AssessAndApplyHistoricalMGPositiveFloor(
            std::vector<double>{0, 0}, no_positive_post, scales, volumes,
            2, cell_ids, 100, 100, false);
    require(no_positive_floor.Applied &&
            std::abs(no_positive_post[0] - 5e-10) < 1e-24 &&
            std::abs(no_positive_post[1] - 5e-10) < 1e-24,
            "no-positive-cell fallback did not use global max per group");

    CG::HistoricalMGResidualCorrectionDiagnostics floor_diagnostics;
    CG::ResetHistoricalMGResidualCorrectionDiagnostics(
        floor_diagnostics, 2, 2);
    CG::RecordHistoricalMGPositiveFloor(
        local_failure_floor, floor_diagnostics);
    auto const floor_failure = CG::HistoricalMGPositiveFloorFailure(
        local_failure_floor, local_failure_negativity);
    CG::RecordHistoricalMGResidualCorrectionFailure(
        floor_failure, floor_diagnostics, 38,
        &local_failure_continuation);
    require(floor_diagnostics.failure_cell_id == 73 &&
            floor_diagnostics.failure_group == 1 &&
            floor_diagnostics.failure_pre_correction_Eg == 0 &&
            floor_diagnostics.failure_post_correction_Eg < 0 &&
            floor_diagnostics.failure_residual_correction ==
                floor_diagnostics.failure_post_correction_Eg &&
            floor_diagnostics.failure_correction_scale == 0.25 &&
            floor_diagnostics.failure_causality ==
                CG::HistoricalMGResidualCorrectionCausality::
                    CreatedNegativity &&
            floor_diagnostics.failure_solver_iterations == 38,
            "floor rejection lost pre/post/correction/lambda causality");

#ifdef RICH_MPI
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    std::vector<double> mpi_post = rank == 0 ?
        std::vector<double>{100, -1e-9} :
        (rank == 1 ? std::vector<double>{1000, -1e-7} :
                     std::vector<double>{10, 1});
    std::vector<double> mpi_pre = mpi_post;
    if(mpi_post[1] < 0)
        mpi_pre[1] = 0;
    std::size_t const mpi_cell_id_base =
        (std::size_t{1} << 32) + std::size_t{100};
    std::vector<std::size_t> const mpi_cell_ids{
        mpi_cell_id_base + static_cast<std::size_t>(rank)};
    auto mpi_negativity =
        CG::AssessHistoricalMGCorrectedNegativity(
            mpi_pre, mpi_post, scales, 2, mpi_cell_ids, 0, true);
    std::vector<double> const mpi_volumes{1, 1};
    auto const mpi_floor = CG::AssessAndApplyHistoricalMGPositiveFloor(
        mpi_pre, mpi_post, scales, mpi_volumes, 2, mpi_cell_ids, 0, 0,
        true);
    require(mpi_negativity.Finite && mpi_floor.Finite &&
            mpi_negativity.GlobalMaximumAbsoluteEg == 1000 &&
            mpi_floor.GlobalMaximumCellEnergy == 1000 &&
            mpi_floor.CollectedGlobally && mpi_floor.Applied,
            "MPI floor did not use global radiation comparison scales");
    if(size >= 2)
        require(mpi_negativity.HasNegative && mpi_negativity.Rank == 1 &&
                mpi_negativity.CellId == mpi_cell_id_base + 1 &&
                mpi_negativity.Group == 1 &&
                mpi_negativity.PostCorrectionEg == -1e-7 &&
                mpi_floor.RepresentativeRank == 1 &&
                mpi_floor.RepresentativeCellId == mpi_cell_id_base + 1 &&
                mpi_floor.RepresentativeGroup == 1 &&
                mpi_floor.FlooredCells == 2,
                "MPI floor diagnostic lost worst-cell ownership");
#endif
}

void testHistoricalDiagnosticUnderflow()
{
    std::vector<double> const solution{1, 0};
    std::vector<double> const previous = solution;
    std::vector<double> const residual{0, 0};
    std::vector<double> const rhs{1, 0};
    std::vector<double> const diagonal{
        1, std::numeric_limits<double>::denorm_min()};
    CG::HistoricalMGMetrics const metrics = CG::MeasureHistoricalMG(
        solution, previous, residual, rhs, diagonal, 1);
    require(metrics.finite && metrics.historical_error == 0 &&
            metrics.max0 == 0 && metrics.max1 == 0,
            "finite zero residual was rejected after a diagnostic denominator underflow");
    CG::HistoricalMGDecision const decision = CG::ClassifyHistoricalMG(
        metrics, 1, CG::historical_mg_squared_tolerance);
    require(decision.accept && !decision.reject,
            "finite converged historical state was not accepted");
}

void testHistoricalFiniteBreakdownRestart()
{
    CG::HistoricalMGMetrics finite;
    finite.finite = true;
    require(CG::ShouldRestartHistoricalMGFiniteBreakdown(
                CG::HistoricalMGBreakdown::TinyAlphaOmega, finite, 9),
            "finite omega breakdown did not restart before iteration 10");
    require(CG::ShouldRestartHistoricalMGFiniteBreakdown(
                CG::HistoricalMGBreakdown::TinyRho, finite, 1),
            "finite rho breakdown did not restart before iteration 10");
    require(!CG::ShouldRestartHistoricalMGFiniteBreakdown(
                CG::HistoricalMGBreakdown::TinyAlphaOmega, finite, 10),
            "finite omega breakdown restarted after iteration 10");
    require(!CG::ShouldRestartHistoricalMGFiniteBreakdown(
                CG::HistoricalMGBreakdown::NonFinite, finite, 9),
            "genuine nonfinite breakdown was restarted");
    finite.finite = false;
    require(!CG::ShouldRestartHistoricalMGFiniteBreakdown(
                CG::HistoricalMGBreakdown::TinyAlphaOmega, finite, 9),
            "nonfinite true residual was restarted");
}

void testGloballyNegligibleNegativeSpectralFloor()
{
    double constexpr representative_negative_group =
        -6.3769648849599452e27;
    double constexpr total_negative_extent = 6.3799634091935332e27;
    double constexpr positive_extent = 2.5792383680236759e29;
    double constexpr global_maximum_cell_radiation_extent =
        2.3671911107700317e41;
    double constexpr second_negative_group =
        -(total_negative_extent + representative_negative_group);
    std::vector<double> groups{
        positive_extent, representative_negative_group,
        second_negative_group};
    double total_extent =
        std::accumulate(groups.begin(), groups.end(), 0.0);
    std::vector<double> ordinary_groups = groups;
    auto const ordinary =
        RadiationPositivity::RepairSmallNegativeGroupExtents(
            ordinary_groups, total_extent,
            RadiationPositivity::spectral_repair_relative_limit);
    require(!ordinary.valid && ordinary_groups == groups,
            "reference deficit unexpectedly passed the local 1e-6 policy");

    auto const controlled =
        RadiationPositivity::RepairControlledNegativeGroupExtents(
            groups, total_extent,
            RadiationPositivity::spectral_repair_relative_limit,
            global_maximum_cell_radiation_extent);
    double const expected_ratio =
        total_negative_extent /
        global_maximum_cell_radiation_extent;
    require(controlled.repair.valid && controlled.repair.repaired &&
            controlled.used_global_negative_exception,
            "exact reference spectrum did not use the global-negative exception");
    require(std::abs(
                controlled.global_negative.
                    negative_extent_to_global_max_ratio /
                    expected_ratio - 1) < 1e-14 &&
            expected_ratio <
                RadiationPositivity::
                    spectral_globally_negligible_negative_fraction,
            "reference decision did not use total-negative/global-maximum ratio");
    require(controlled.diagnostic_total_to_global_max_ratio >
                controlled.global_negative.
                    negative_extent_to_global_max_ratio,
            "test does not distinguish the diagnostic cell-total ratio");
    require(total_extent ==
                std::accumulate(groups.begin(), groups.end(), 0.0),
            "controlled repair did not synchronize Erad with sum(Eg)");
    require(groups[0] == positive_extent && groups[1] > 0 && groups[2] > 0,
            "controlled repair changed a positive group or left a negative group");

    double constexpr threshold_global_maximum = 1e40;
    double const threshold_negative =
        RadiationPositivity::
            spectral_globally_negligible_negative_fraction *
        threshold_global_maximum;
    std::vector<double> equality_groups{1e34, -threshold_negative};
    double equality_total =
        std::accumulate(equality_groups.begin(), equality_groups.end(), 0.0);
    auto const equality =
        RadiationPositivity::RepairControlledNegativeGroupExtents(
            equality_groups, equality_total,
            RadiationPositivity::spectral_repair_relative_limit,
            threshold_global_maximum);
    require(equality.repair.valid && equality.repair.repaired &&
            equality.used_global_negative_exception,
            "inclusive global-negative threshold rejected equality");

    double const above_negative = std::nextafter(
        threshold_negative, std::numeric_limits<double>::infinity());
    std::vector<double> above_groups{1e34, -above_negative};
    std::vector<double> const above_original = above_groups;
    double above_total =
        std::accumulate(above_groups.begin(), above_groups.end(), 0.0);
    double const above_total_original = above_total;
    auto const above =
        RadiationPositivity::RepairControlledNegativeGroupExtents(
            above_groups, above_total,
            RadiationPositivity::spectral_repair_relative_limit,
            threshold_global_maximum);
    require(!above.repair.valid &&
            above.repair.failure ==
                RadiationPositivity::SpectralRepairFailure::
                    NegativeExtentExceedsTolerance &&
            above_groups == above_original &&
            above_total == above_total_original,
            "negative/global ratio above the configured threshold did not reject atomically");

    std::vector<double> invalid_scale_groups{4, -1, 2};
    double invalid_scale_total = 5;
    auto const invalid_scale =
        RadiationPositivity::RepairControlledNegativeGroupExtents(
            invalid_scale_groups, invalid_scale_total,
            RadiationPositivity::spectral_repair_relative_limit, 0.0);
    require(!invalid_scale.repair.valid &&
            !invalid_scale.global_negative.valid,
            "invalid global maximum weakened a locally large failure");

    std::vector<double> mismatched_groups{4, 1, 2};
    std::vector<double> const mismatched_original = mismatched_groups;
    double mismatched_total = 6;
    auto const mismatch =
        RadiationPositivity::RepairControlledNegativeGroupExtents(
            mismatched_groups, mismatched_total, 0.2, 1e30);
    require(mismatch.repair.valid && !mismatch.repair.repaired &&
            mismatched_groups == mismatched_original &&
            mismatched_total == 7 &&
            mismatch.aggregate_sync_correction == 1,
            "finite Erad/group-sum drift was not synchronized exactly");

    std::vector<double> nonfinite_groups{
        1, std::numeric_limits<double>::quiet_NaN()};
    std::vector<double> const nonfinite_original = nonfinite_groups;
    double nonfinite_total = 1;
    auto const nonfinite =
        RadiationPositivity::RepairControlledNegativeGroupExtents(
            nonfinite_groups, nonfinite_total,
            RadiationPositivity::spectral_repair_relative_limit, 1e30);
    require(!nonfinite.repair.valid &&
            nonfinite.repair.failure ==
                RadiationPositivity::SpectralRepairFailure::
                    NonfiniteGroupExtent &&
            std::string(RadiationPositivity::SpectralRepairFailureLabel(
                nonfinite.repair.failure)) != "none" &&
            nonfinite_groups[0] == nonfinite_original[0] &&
            std::isnan(nonfinite_groups[1]) &&
            nonfinite_total == 1,
            "non-finite spectrum lacked an atomic classified rejection");
}

#ifdef RICH_MPI
void testCanonicalGlobalMaximumNegativePolicyMPI()
{
    int rank = 0;
    int ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    std::vector<double> const canonical_owned_radiation_extents{
        rank == ranks - 1 ? 2.3671911107700317e41 : 1e20,
        rank == 0 ? 3e30 : 2e30};
    double global_maximum = *std::max_element(
        canonical_owned_radiation_extents.begin(),
        canonical_owned_radiation_extents.end());
    MPI_Allreduce(MPI_IN_PLACE, &global_maximum, 1, MPI_DOUBLE, MPI_MAX,
                  MPI_COMM_WORLD);
    require(global_maximum == 2.3671911107700317e41,
            "MPI maximum did not use all canonical owned-cell Erad values");
    double const fallback_boundary =
        CG::historical_mg_positivity_continuation_trigger_fraction *
        global_maximum;
    int const local_fallback =
        CG::HistoricalMGNegativeValueRequiresComptonFallback(
            -1.0001 * fallback_boundary, global_maximum) &&
        !CG::HistoricalMGNegativeValueRequiresComptonFallback(
            -fallback_boundary, global_maximum) ? 1 : 0;
    int collective_fallback = local_fallback;
    MPI_Allreduce(MPI_IN_PLACE, &collective_fallback, 1, MPI_INT, MPI_MIN,
                  MPI_COMM_WORLD);
    require(collective_fallback == 1,
            "MPI ranks disagreed on the Compton fallback threshold");
    std::vector<double> groups{
        2.5792383680236759e29, -6.3799634091935332e27};
    double total_extent =
        std::accumulate(groups.begin(), groups.end(), 0.0);
    auto const controlled =
        RadiationPositivity::RepairControlledNegativeGroupExtents(
            groups, total_extent,
            RadiationPositivity::spectral_repair_relative_limit,
            global_maximum);
    int local_decision = controlled.repair.valid &&
        controlled.used_global_negative_exception ? 1 : 0;
    int minimum_decision = local_decision;
    int maximum_decision = local_decision;
    MPI_Allreduce(MPI_IN_PLACE, &minimum_decision, 1, MPI_INT, MPI_MIN,
                  MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &maximum_decision, 1, MPI_INT, MPI_MAX,
                  MPI_COMM_WORLD);
    require(minimum_decision == 1 && maximum_decision == 1,
            "MPI ranks disagreed on the canonical global-negative decision");
}
#endif

#ifndef RICH_MPI
void testRepairOrderingAroundComptonSubsteps()
{
    Voronoi3D tess(Vector3D(0, 0, 0), Vector3D(1, 1, 1));
    tess.Build(std::vector<Vector3D>{
        Vector3D(0.25, 0.5, 0.5), Vector3D(0.75, 0.5, 0.5)});
    std::vector<ComputationalCell3D> cells(2);
    std::vector<Conserved3D> extensives(2);
    for(std::size_t cell = 0; cell < cells.size(); ++cell) {
        cells[cell].ID = 8100 + cell;
        cells[cell].density = 1;
        cells[cell].temperature = 1;
        cells[cell].internal_energy = 1;
        extensives[cell].mass = 1;
        extensives[cell].energy = 1;
        extensives[cell].internal_energy = 1;
    }
    extensives[1].Erad = 2.3671911107700317e41;
    extensives[1].Eg[0] = extensives[1].Erad;
    cells[1].Erad = extensives[1].Erad;
    cells[1].Eg[0] = extensives[1].Eg[0];

    IndividualStepContext context;
    context.previous_event_tick = 0;
    context.event_tick = 1;
    context.previous_event_time = 0;
    context.event_time = 1;
    context.time_quantum = 1;
    context.mesh_build_policy = IndividualMeshBuildPolicy::FullReference;
    context.active_indices = {0};
    context.active_mask = {1, 0};
    context.cell_time_steps = {1, 1};
    context.primitive_ticks = {0, 0};

    std::vector<ComputationalCell3D> const canonical_cells = cells;
    std::vector<Conserved3D> canonical_extensives = extensives;
    ComputationalCell3D const inactive_cell_before = cells[1];
    Conserved3D const inactive_extensive_before = extensives[1];
    IdealGas eos(5.0 / 3.0);
    OrderedSpectralRepairDriver driver(eos);
    int iterations = 0;
    bool const accepted = driver.stepIndividual(
        1e-12, iterations, tess, cells, extensives, context, 1, 0,
        &canonical_cells, &canonical_extensives);
    require(accepted,
            "ordered spectral-repair candidate was unexpectedly rejected");
    require(driver.absorption_diffusion_commits == 1 &&
            driver.post_solve_calls == 1,
            "absorption/diffusion or Compton stage was applied twice");
    require(driver.dormant_global_storage_releases == 1,
            "distributed-active solve did not release dormant global storage");
    require(driver.absorption_repaired_before_compton &&
            driver.aggregate_consistent_before_compton,
            "Compton ran before absorption/diffusion repair and Erad sync");
    require(driver.compton_substep_validations == 2 &&
            driver.every_compton_substep_repaired,
            "controlled repair did not run after every Compton substep");
    require(cells[1].Erad == inactive_cell_before.Erad &&
            cells[1].Eg == inactive_cell_before.Eg &&
            cells[1].internal_energy == inactive_cell_before.internal_energy &&
            cells[1].temperature == inactive_cell_before.temperature &&
            extensives[1].Erad == inactive_extensive_before.Erad &&
            extensives[1].Eg == inactive_extensive_before.Eg &&
            extensives[1].mass == inactive_extensive_before.mass &&
            extensives[1].energy == inactive_extensive_before.energy &&
            extensives[1].internal_energy ==
                inactive_extensive_before.internal_energy,
            "production individual radiation path changed an inactive cell");
}
#endif

void testSimulationSynchronizedEventLifecycle()
{
    std::vector<Vector3D> const points = {
        Vector3D(0.2, 0.2, 0.2),
        Vector3D(0.8, 0.2, 0.2),
        Vector3D(0.2, 0.8, 0.8),
        Vector3D(0.8, 0.8, 0.8)};
    std::vector<ComputationalCell3D> cells(points.size());
    for(ComputationalCell3D& cell : cells)
    {
        cell.density = 1;
        cell.pressure = 1;
        cell.internal_energy = 1.5;
    }
    IdealGas eos(5.0 / 3.0);
    Voronoi3D tess(Vector3D(0, 0, 0), Vector3D(1, 1, 1));
    tess.Build(points);
    Simulation simulation(tess, cells, eos);
    simulation.SetTimeStepFunction(std::make_shared<ManualTimeStep>(0.125));

    IndividualTimeStepOptions options;
    options.initial_bin = 0;
    options.time_quantum = 0.125;
    simulation.EnableIndividualTimeSteps(options);

    auto make_sparse_states = [&]() {
        std::vector<CellTimeState> states(points.size());
        for(std::size_t index = 0; index < states.size(); ++index)
        {
            states[index].cell_id = simulation.getCells()[index].ID;
            states[index].begin_tick = 0;
            states[index].end_tick = index == 0 ? 1 : 4;
            states[index].last_primitive_tick = 0;
            states[index].time_bin = index == 0 ? 0 : 2;
        }
        return states;
    };
    IndividualTimeStepScheduler* scheduler =
        simulation.GetIndividualTimeStepScheduler();
    scheduler->restore(simulation.getCells(), simulation.GetTime(), 0.125, 0,
                       make_sparse_states());
    require(scheduler->prepareEvent(simulation.getCells()).active_indices.size() == 1,
            "one-shot synchronized-event test did not start from sparse bins");

    std::vector<std::size_t> callback_active_counts;
    bool all_callbacks_saw_in_flight_state = true;
    simulation.SetIndividualPostPhysics(
        [&](IndividualStepContext const& context) {
            callback_active_counts.push_back(context.active_indices.size());
            all_callbacks_saw_in_flight_state =
                all_callbacks_saw_in_flight_state &&
                !simulation.IndividualStateSynchronized();
        });
    simulation.RequestSynchronizedIndividualEvent();
    simulation.step();

    require(callback_active_counts.size() == 1,
            "individual post-physics callback did not run exactly once");
    require(callback_active_counts[0] == points.size(),
            "one-shot synchronized event did not activate every cell");
    require(all_callbacks_saw_in_flight_state,
            "in-flight individual event was reported as committed");
    require(simulation.IndividualStateSynchronized(),
            "committed one-shot individual event was not synchronized");
    require(!scheduler->forceAllActiveLatched(),
            "one-shot synchronized event set the permanent scheduler latch");

    *scheduler = IndividualTimeStepScheduler(options);
    scheduler->restore(simulation.getCells(), simulation.GetTime(), 0.125, 0,
                       make_sparse_states());
    simulation.step();
    require(callback_active_counts.size() == 2 &&
            callback_active_counts[1] == 1,
            "one-shot synchronized-event request did not clear after commit");
    require(all_callbacks_saw_in_flight_state,
            "later sparse event was reported as committed during its callback");
    require(!simulation.IndividualStateSynchronized(),
            "later sparse event was incorrectly reported as synchronized");
}

void testMonteCarloRejection()
{
    std::vector<Vector3D> const points = {
        Vector3D(0.2, 0.2, 0.2), Vector3D(0.8, 0.2, 0.2),
        Vector3D(0.2, 0.8, 0.2), Vector3D(0.8, 0.8, 0.2),
        Vector3D(0.2, 0.2, 0.8), Vector3D(0.8, 0.2, 0.8),
        Vector3D(0.2, 0.8, 0.8), Vector3D(0.8, 0.8, 0.8)};
    std::vector<ComputationalCell3D> cells(points.size());
    for(ComputationalCell3D& cell : cells)
    {
        cell.density = 1;
        cell.pressure = 1;
        cell.internal_energy = 1.5;
    }
    IdealGas eos(5.0 / 3.0);

    for(bool add_monte_carlo_first : {false, true})
    {
        Voronoi3D tess(Vector3D(0, 0, 0), Vector3D(1, 1, 1));
        tess.Build(points);
        Simulation simulation(tess, cells, eos);
        std::shared_ptr<PhysicsStep> const monte_carlo =
            std::make_shared<MonteCarloStepStub>();

        bool rejected = false;
        try
        {
            if(add_monte_carlo_first)
            {
                simulation.addPhysics(monte_carlo);
                simulation.EnableIndividualTimeSteps();
            }
            else
            {
                simulation.EnableIndividualTimeSteps();
                simulation.addPhysics(monte_carlo);
            }
        }
        catch(std::invalid_argument const& error)
        {
            std::string const message(error.what());
            rejected = message.find(RadiationMCStep::step_name) != std::string::npos &&
                message.find("Monte Carlo") != std::string::npos &&
                message.find("global timesteps") != std::string::npos;
        }
        require(rejected,
                "Monte Carlo radiation was not rejected during individual-timestep setup");
    }
}

void testGlobalRadiationFractionalRetry()
{
    Voronoi3D tess(Vector3D(0, 0, 0), Vector3D(1, 1, 1));
    tess.Build(std::vector<Vector3D>{Vector3D(0.5, 0.5, 0.5)});
    std::vector<ComputationalCell3D> cells(1);
    std::vector<Conserved3D> extensives(1);
    IdealGas eos(5.0 / 3.0);
    ProgressTracker tracker;
    RetryingRadiationDriver driver(eos);
    RadiationStep step(tess, cells, extensives, tracker,
#ifdef RICH_MPI
        nullptr,
#endif
        driver, false);

    step.step(1.0);
    require(driver.rejections == 11,
            "global radiation did not cross the former 0.1% retry cutoff");
    require(driver.acceptances == 65,
            "global radiation retry interval did not recover geometrically");
    require(close(driver.accepted_time, 1.0, 2e-15),
            "global radiation fractional retries did not cover the target interval");
    require(driver.time_consistent,
            "global radiation fractional retries used a stale candidate time");
}

void testGlobalRadiationPersistentFractionalRetryBackoff()
{
    Voronoi3D tess(Vector3D(0, 0, 0), Vector3D(1, 1, 1));
    tess.Build(std::vector<Vector3D>{Vector3D(0.5, 0.5, 0.5)});
    std::vector<ComputationalCell3D> cells(1);
    std::vector<Conserved3D> extensives(1);
    IdealGas eos(5.0 / 3.0);
    ProgressTracker tracker;
    RetryingRadiationDriver driver(eos, false);
    RadiationStep step(tess, cells, extensives, tracker,
#ifdef RICH_MPI
        nullptr,
#endif
        driver, false);

    step.step(1.0);
    require(driver.rejections == 23,
            "persistent retry restriction did not use exponential probe backoff");
    require(driver.acceptances == 2048,
            "persistent retry restriction did not cover the target at its safe interval");
    require(close(driver.accepted_time, 1.0, 2e-15),
            "persistent fractional retries did not cover the target interval");
    require(driver.time_consistent,
            "persistent fractional retries used a stale candidate time");
}

void testIndividualRadiationRetryLimiterScope()
{
    Voronoi3D tess(Vector3D(0, 0, 0), Vector3D(1, 1, 1));
    std::vector<Vector3D> points;
#ifdef RICH_MPI
    int rank = 0;
    int rank_count = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &rank_count);
    std::size_t global_index = 0;
    for(std::size_t i = 0; i < 4; ++i)
        for(std::size_t j = 0; j < 4; ++j)
            for(std::size_t k = 0; k < 4; ++k, ++global_index)
                if(global_index % static_cast<std::size_t>(rank_count) ==
                   static_cast<std::size_t>(rank))
                    points.push_back(Vector3D(
                        (static_cast<double>(i) + 0.5) / 4.0,
                        (static_cast<double>(j) + 0.5) / 4.0,
                        (static_cast<double>(k) + 0.5) / 4.0));
    tess.BuildParallel(points);
#else
    points = {Vector3D(0.25, 0.5, 0.5),
              Vector3D(0.75, 0.5, 0.5)};
    tess.Build(points);
    int const rank = 0;
#endif
    require(tess.GetPointNo() >= 2,
            "retry-scope test requires two owned cells per MPI rank");
    std::vector<ComputationalCell3D> cells(tess.GetPointNo());
    for(std::size_t cell = 0; cell < cells.size(); ++cell)
        cells[cell].ID = 701 + static_cast<std::size_t>(rank) * 100000 + cell;
    std::vector<Conserved3D> extensives(cells.size());
    IdealGas eos(5.0 / 3.0);
    ProgressTracker tracker;

    IndividualStepContext context;
    context.previous_event_tick = 0;
    context.event_tick = 1;
    context.previous_event_time = 0;
    context.event_time = 1;
    context.time_quantum = 1;
    context.mesh_build_policy = IndividualMeshBuildPolicy::FullReference;
    context.active_indices.resize(cells.size());
    std::iota(context.active_indices.begin(), context.active_indices.end(), 0);
    context.active_mask.assign(cells.size(), 1);
    context.cell_time_steps.assign(cells.size(), 1);
    context.primitive_ticks.assign(cells.size(), 0);

    for(int const failure_scope : {0, 1, 2}) {
        bool const cell_local_failure = failure_scope == 0;
        bool const attributed_collective_failure = failure_scope == 2;
        RetryingIndividualRadiationDriver driver(
            eos, cell_local_failure, attributed_collective_failure);
        RadiationStep step(tess, cells, extensives, tracker,
#ifdef RICH_MPI
            nullptr,
#endif
            driver, false);
        step.stepIndividual(context);
        require(driver.rejections == 1 && driver.acceptances == 2,
                "individual radiation retry did not cover the event");

        std::vector<double> limits(2,
            std::numeric_limits<double>::infinity());
        step.suggestIndividualTimeSteps(context, limits);
        require(close(limits[0], 0.5, 2e-15),
                "failed individual radiation cell did not retain its retry limit");
        double const expected_other = cell_local_failure ? 8.0 : 0.5;
        require(close(limits[1], expected_other, 2e-15),
                "individual radiation retry limiter used the wrong scope");
    }
}

void testGreyFractionalCandidateRefresh()
{
    Voronoi3D tess(Vector3D(0, 0, 0), Vector3D(1, 1, 1));
    std::vector<Vector3D> points;
    int rank = 0;
#ifdef RICH_MPI
    int rank_count = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &rank_count);
    std::size_t global_index = 0;
    for(std::size_t i = 0; i < 4; ++i)
        for(std::size_t j = 0; j < 4; ++j)
            for(std::size_t k = 0; k < 4; ++k, ++global_index)
                if(global_index % static_cast<std::size_t>(rank_count) ==
                   static_cast<std::size_t>(rank))
                    points.push_back(Vector3D(
                        (static_cast<double>(i) + 0.5) / 4.0,
                        (static_cast<double>(j) + 0.5) / 4.0,
                        (static_cast<double>(k) + 0.5) / 4.0));
    tess.BuildParallel(points);
#else
    points = {
        Vector3D(0.25, 0.25, 0.25), Vector3D(0.75, 0.25, 0.25),
        Vector3D(0.25, 0.75, 0.25), Vector3D(0.75, 0.75, 0.25),
        Vector3D(0.25, 0.25, 0.75), Vector3D(0.75, 0.25, 0.75),
        Vector3D(0.25, 0.75, 0.75), Vector3D(0.75, 0.75, 0.75)};
    tess.Build(points);
#endif

    IdealGas eos(5.0 / 3.0, 1.5, 1.0, 0.0);
    std::vector<ComputationalCell3D> initial_cells(tess.GetPointNo());
    std::vector<Conserved3D> initial_extensives(tess.GetPointNo());
    for(std::size_t i = 0; i < initial_cells.size(); ++i)
    {
        ComputationalCell3D& cell = initial_cells[i];
        cell.ID = (std::size_t{1} << 32) + 9001 +
            static_cast<std::size_t>(rank) * 100000 + i;
        cell.density = 1.0;
        cell.temperature = 900.0 + 200.0 * tess.GetMeshPoint(i).x;
        cell.internal_energy = eos.dT2e(
            cell.density, cell.temperature, cell.tracers,
            ComputationalCell3D::tracerNames);
        cell.pressure = eos.de2p(
            cell.density, cell.internal_energy, cell.tracers,
            ComputationalCell3D::tracerNames);
        double const equilibrium = CG::radiation_constant *
            std::pow(cell.temperature, 4);
        cell.Erad = (2.0 + 0.25 * tess.GetMeshPoint(i).x) *
            equilibrium / cell.density;
        PrimitiveToConserved(cell, tess.GetVolume(i), initial_extensives[i]);
    }

    auto make_context = [&](std::uint64_t begin_tick,
                            std::uint64_t end_tick,
                            double begin_time,
                            double end_time)
    {
        IndividualStepContext context;
        context.previous_event_tick = begin_tick;
        context.event_tick = end_tick;
        context.previous_event_time = begin_time;
        context.event_time = end_time;
        context.time_quantum = 0.5;
        context.mesh_build_policy = IndividualMeshBuildPolicy::FullReference;
        context.active_indices.resize(initial_cells.size());
        std::iota(context.active_indices.begin(), context.active_indices.end(), 0);
        context.active_mask.assign(initial_cells.size(), 1);
        context.cell_time_steps.assign(initial_cells.size(), end_time - begin_time);
        context.primitive_ticks.assign(initial_cells.size(), begin_tick);
        return context;
    };

    PowerLawOpacity opacity(1e-12, 0.0, 0.0, 1e-12, 0.0, 0.0);
    DiffusionClosedBox boundary;

    std::vector<ComputationalCell3D> retried_cells = initial_cells;
    std::vector<Conserved3D> retried_extensives = initial_extensives;
    ProgressTracker retried_tracker;
    RetryOnceGreyDiffusion retried_driver(opacity, boundary, eos, true);
    RadiationStep retried_step(tess, retried_cells, retried_extensives,
        retried_tracker,
#ifdef RICH_MPI
        nullptr,
#endif
        retried_driver, false);
    retried_step.stepIndividual(make_context(0, 2, 0.0, 1.0));

    std::vector<ComputationalCell3D> sequenced_cells = initial_cells;
    std::vector<Conserved3D> sequenced_extensives = initial_extensives;
    ProgressTracker sequenced_tracker;
    RetryOnceGreyDiffusion sequenced_driver(opacity, boundary, eos, false);
    RadiationStep sequenced_step(tess, sequenced_cells, sequenced_extensives,
        sequenced_tracker,
#ifdef RICH_MPI
        nullptr,
#endif
        sequenced_driver, false);
    sequenced_step.stepIndividual(make_context(0, 1, 0.0, 0.5));
    sequenced_step.stepIndividual(make_context(1, 2, 0.5, 1.0));

    if(retried_driver.rejected_candidates != 1 ||
       retried_driver.candidate_preparations != 3 ||
       retried_driver.accepted_intervals != 2)
    {
        std::ostringstream message;
        message << "grey retry counts differ: rejected="
                << retried_driver.rejected_candidates
                << " prepared=" << retried_driver.candidate_preparations
                << " accepted=" << retried_driver.accepted_intervals;
        require(false, message.str());
    }
    require(sequenced_driver.rejected_candidates == 0 &&
            sequenced_driver.candidate_preparations == 2 &&
            sequenced_driver.accepted_intervals == 2,
            "explicit grey half steps did not execute exactly twice");

    bool changed = false;
    for(std::size_t i = 0; i < retried_cells.size(); ++i)
    {
        changed = changed ||
            std::abs(retried_cells[i].Erad - initial_cells[i].Erad) > 1e-10;
        require(close(retried_cells[i].Erad, sequenced_cells[i].Erad, 2e-8) &&
                close(retried_cells[i].temperature,
                      sequenced_cells[i].temperature, 2e-8) &&
                close(retried_cells[i].internal_energy,
                      sequenced_cells[i].internal_energy, 2e-8) &&
                close(retried_extensives[i].Erad,
                      sequenced_extensives[i].Erad, 2e-8) &&
                close(retried_extensives[i].energy,
                      sequenced_extensives[i].energy, 2e-8),
                "grey fractional retry differs from two explicit half steps");
    }
#ifdef RICH_MPI
    int changed_anywhere = changed ? 1 : 0;
    MPI_Allreduce(MPI_IN_PLACE, &changed_anywhere, 1, MPI_INT, MPI_MAX,
                  MPI_COMM_WORLD);
    changed = changed_anywhere != 0;
#endif
    require(changed, "grey fractional retry test produced no radiation update");
}

} // namespace

int main(int argc, char** argv)
{
    int rank = 0;
#ifdef RICH_MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    char const* const fail_stop_test =
        std::getenv("RICH_TEST_DISTRIBUTED_ACTIVE_FAILSTOP");
    if(fail_stop_test != nullptr) {
        std::string const fail_stop_mode(fail_stop_test);
        if(fail_stop_mode == "1") {
            if(rank == 0)
                RadiationDriverTestHooks::
                    TriggerDistributedActiveFailStopForTest();
            MPI_Barrier(MPI_COMM_WORLD);
            MPI_Finalize();
            return 98;
        }
        if(fail_stop_mode == "returned_after_post")
            RadiationDriverTestHooks::
                TriggerDistributedActiveReturnedMpiFailureForTest();
        if(fail_stop_mode == "returned_collective")
            RadiationDriverTestHooks::
                TriggerDistributedActiveReturnedCollectiveFailureForTest();
    }
#endif
    if(rank == 0) {
        std::remove("test_passed.res");
        std::remove("test_failed.res");
        std::remove("individual_time_steps_failure.txt");
    }
#ifdef RICH_MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif

    int status = 0;
    std::string failure;
    try
    {
        testHistoricalFinalCorrectionGate();
        testBiCGSTABWorkspaceRelease();
        testFixed16BlockStencilCore();
        testPrecomputedComptonLinearTables();
        testCompactWideRadiationMatrixRows();
        testCellBlockCSRProductionShapes();
        testCellBlockForwardGaussSeidel();
        testRankLocalILU0();
        testHistoricalMGPositivityContinuation();
        testHistoricalDiagnosticUnderflow();
        testHistoricalFiniteBreakdownRestart();
        testGloballyNegligibleNegativeSpectralFloor();
        testSpectralRoundoffPositivityRepair();
#ifdef RICH_MPI
        testPermutationAwareAllActiveMapping();
        testDistributedActiveWideIdExchangeMPI();
        testDistributedActiveExchangeFailuresMPI();
        testDistributedActiveLargeCountHelpersMPI();
        testCanonicalGlobalMaximumNegativePolicyMPI();
        testIndividualRadiationRetryLimiterScope();
        testGreyFractionalCandidateRefresh();
#else
        testSchedulerAndAMR();
        testSynchronizedScheduler();
        testInactiveWakeAlignment();
        testForcedAllActiveEventOverlay();
        testDefaultCellUpdaterAllActiveCommit();
        testHydroRadiationPositivityLimiter();
        testRepairOrderingAroundComptonSubsteps();
        testPartialGeometry();
        testSimulationSynchronizedEventLifecycle();
        testMonteCarloRejection();
        testGlobalRadiationFractionalRetry();
        testGlobalRadiationPersistentFractionalRetryBackoff();
        testIndividualRadiationRetryLimiterScope();
        testGreyFractionalCandidateRefresh();
#endif
    }
    catch(std::exception const& error)
    {
        status = 1;
        failure = error.what();
    }

#ifdef RICH_MPI
    int global_status = status;
    MPI_Allreduce(MPI_IN_PLACE, &global_status, 1, MPI_INT, MPI_MAX,
                  MPI_COMM_WORLD);
    status = global_status;
#endif
    if(rank == 0) {
        if(status == 0)
            std::ofstream("test_passed.res").close();
        else {
            std::ofstream output("individual_time_steps_failure.txt");
            output << (failure.empty() ?
                "individual timestep test failed on another MPI rank" :
                failure) << '\n';
            std::ofstream("test_failed.res").close();
        }
    }
#ifdef RICH_MPI
    MPI_Finalize();
#endif
    return status;
}
