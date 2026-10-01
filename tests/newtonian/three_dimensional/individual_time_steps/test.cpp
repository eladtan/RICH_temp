#include <algorithm>
#include <array>
#include <cerrno>
#include <cfenv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <exception>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>
#include <unistd.h>

#include "3D/tessellation/Voronoi3D.hpp"
#include "newtonian/common/ideal_gas.hpp"
#include "newtonian/three_dimensional/ConditionActionFlux1.hpp"
#include "newtonian/three_dimensional/ConditionExtensiveUpdater3D.hpp"
#include "newtonian/three_dimensional/Ghost3D.hpp"
#include "newtonian/three_dimensional/LinearGauss3D.hpp"
#include "newtonian/three_dimensional/ManualTimeStep.hpp"
#include "newtonian/three_dimensional/computational_cell.hpp"
#include "newtonian/three_dimensional/default_cell_updater.hpp"
#include "newtonian/three_dimensional/default_extensive_updater.hpp"
#include "newtonian/three_dimensional/eulerian_3d.hpp"
#include "newtonian/three_dimensional/hdsim_3d.hpp"
#include "newtonian/three_dimensional/Hllc3D.hpp"
#include "newtonian/three_dimensional/SourceTerm3D.hpp"
#include "newtonian/three_dimensional/simulation/ActiveMeshView.hpp"
#include "newtonian/three_dimensional/simulation/IndividualTimeStep.hpp"
#include "newtonian/three_dimensional/IndividualChangeWakeAccounting.hpp"
#include "newtonian/three_dimensional/simulation/RuntimeLog.hpp"
#include "newtonian/three_dimensional/simulation/Simulation.hpp"
#include "newtonian/three_dimensional/simulation/steps/RemeshStep.hpp"
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

void duplicateDescriptor(int old_descriptor, int new_descriptor,
	const char* failure_message)
{
	int result = -1;
	do
	{
		result = ::dup2(old_descriptor, new_descriptor);
	} while(result < 0 && errno == EINTR);
	if(result < 0)
		throw std::runtime_error(failure_message);
}

template<class Function>
std::string captureStandardError(Function&& function)
{
	std::cerr.flush();
	std::fflush(stderr);
	FILE* const captured = std::tmpfile();
	if(captured == nullptr)
		throw std::runtime_error("failed to create stderr capture file");
	const int saved_stderr = ::dup(STDERR_FILENO);
	if(saved_stderr < 0)
	{
		std::fclose(captured);
		throw std::runtime_error("failed to duplicate stderr");
	}
	try
	{
		duplicateDescriptor(::fileno(captured), STDERR_FILENO,
			"failed to redirect stderr");
	}
	catch(...)
	{
		::close(saved_stderr);
		std::fclose(captured);
		throw;
	}

	try
	{
		function();
	}
	catch(...)
	{
		const std::exception_ptr original_error = std::current_exception();
		std::cerr.flush();
		std::fflush(stderr);
		try
		{
			duplicateDescriptor(saved_stderr, STDERR_FILENO,
				"failed to restore stderr after exception");
			::close(saved_stderr);
			std::fclose(captured);
		}
		catch(...)
		{
			::close(saved_stderr);
			std::fclose(captured);
			throw;
		}
		std::rethrow_exception(original_error);
	}

	std::cerr.flush();
	std::fflush(stderr);
	try
	{
		duplicateDescriptor(saved_stderr, STDERR_FILENO,
			"failed to restore stderr");
	}
	catch(...)
	{
		::close(saved_stderr);
		std::fclose(captured);
		throw;
	}
	::close(saved_stderr);
	if(std::fseek(captured, 0, SEEK_END) != 0)
	{
		std::fclose(captured);
		throw std::runtime_error("failed to size stderr capture");
	}
	const long captured_size = std::ftell(captured);
	if(captured_size < 0 || std::fseek(captured, 0, SEEK_SET) != 0)
	{
		std::fclose(captured);
		throw std::runtime_error("failed to rewind stderr capture");
	}
	std::string output(static_cast<std::size_t>(captured_size), '\0');
	if(!output.empty() &&
		std::fread(&output[0], 1, output.size(), captured) != output.size())
	{
		std::fclose(captured);
		throw std::runtime_error("failed to read stderr capture");
	}
	std::fclose(captured);
	return output;
}

struct DiagnosticFile
{
	std::string path;
	std::string contents;
};

DiagnosticFile readDiagnosticFile(const std::string& notice)
{
	const std::string path_key = " path=";
	const std::size_t path_begin = notice.find(path_key);
	if(path_begin == std::string::npos)
		throw std::runtime_error("diagnostic notice has no path");
	const std::size_t value_begin = path_begin + path_key.size();
	const std::size_t value_end = notice.find(' ', value_begin);
	if(value_end == std::string::npos)
		throw std::runtime_error("diagnostic notice has no path terminator");
	DiagnosticFile result;
	result.path = notice.substr(value_begin, value_end - value_begin);
	std::ifstream input(result.path.c_str());
	if(!input)
		throw std::runtime_error("failed to open diagnostic record");
	std::ostringstream contents;
	contents << input.rdbuf();
	if(input.bad())
		throw std::runtime_error("failed to read diagnostic record");
	result.contents = contents.str();
	input.close();
	if(std::remove(result.path.c_str()) != 0)
		throw std::runtime_error("failed to remove diagnostic test record");
	return result;
}

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

class NonRebalancingIndividualStep final : public PhysicsStep
{
public:
    void step(double) override
    {
        emitRuntimeDiagnostics();
    }
    double suggestTimeStep(void) const override {return 0.125;}
    bool supportsIndividualTimeSteps(void) const override {return true;}
    void stepIndividual(IndividualStepContext const&) override
    {
        ++individual_steps_;
        emitRuntimeDiagnostics();
    }
    std::string getName(void) const override
    {
        return "non-rebalancing individual test step";
    }
    std::size_t individualSteps(void) const {return individual_steps_;}
    void setRuntimeDiagnostics(
        SourceStepTiming const& source_timing,
        std::vector<StepRetryRecord> retries)
    {
        source_timing_ = source_timing;
        retries_ = std::move(retries);
    }
    void throwAfterRuntimeDiagnostics(bool const enabled)
    {
        throw_after_runtime_diagnostics_ = enabled;
    }
    SourceStepTiming getSourceStepTiming(void) const override
    {return source_timing_;}
    MeshBuildTiming getMeshBuildTiming(void) const override
    {
        MeshBuildTiming timing;
        timing.seconds = 0.04;
        timing.builds = 3;
        return timing;
    }
    void setIndividualTimeStepLimits(std::vector<double> limits)
    {individual_time_step_limits_ = std::move(limits);}
    void suggestIndividualTimeSteps(
        IndividualStepContext const& context,
        std::vector<double>& time_step_limits) const override
    {
        if(individual_time_step_limits_.empty())
        {
            PhysicsStep::suggestIndividualTimeSteps(context, time_step_limits);
            return;
        }
        for(std::size_t const index : context.active_indices)
            time_step_limits.at(index) = std::min(
                time_step_limits.at(index),
                individual_time_step_limits_.at(index));
    }

#ifdef RICH_MPI
    bool allowRebalance(void) override {return false;}
    std::string getRequiredLB(void) const override {return std::string();}
    std::vector<double> getLoadBalanceWeights(void) override
    {
        return std::vector<double>();
    }
#endif

private:
    void emitRuntimeDiagnostics() const
    {
        for(StepRetryRecord const& retry : retries_)
            reportStepRetry(retry);
        if(throw_after_runtime_diagnostics_)
            throw std::runtime_error("test failure after retry");
    }

    std::size_t individual_steps_ = 0;
    SourceStepTiming source_timing_;
    std::vector<StepRetryRecord> retries_;
    std::vector<double> individual_time_step_limits_;
    bool throw_after_runtime_diagnostics_ = false;
};

#ifdef RICH_MPI
class RebalanceLoggingStep final : public PhysicsStep
{
public:
    RebalanceLoggingStep(std::size_t const owned_point_count,
                         std::string required_load_balance)
        : owned_point_count_(owned_point_count),
          required_load_balance_(std::move(required_load_balance)) {}

    void step(double) override {}
    double suggestTimeStep(void) const override {return 0.125;}
    std::string getName(void) const override
    {return "rebalance logging test step";}
    bool allowRebalance(void) override {return false;}
    std::string getRequiredLB(void) const override
    {return required_load_balance_;}
    std::vector<double> getLoadBalanceWeights(void) override
    {return std::vector<double>(owned_point_count_, 1);}
    void setRequiredLoadBalance(std::string required_load_balance)
    {required_load_balance_ = std::move(required_load_balance);}

private:
    std::size_t owned_point_count_;
    std::string required_load_balance_;
};
#endif

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

    bool validateDefectForTest(double const maximum_local_fraction,
                               double const maximum_tolerance_ratio,
                               double const event_absolute_fraction) const
    {
        clearStepFailure();
        IndividualRadiationDefectEvent event;
        event.signed_extent = event_absolute_fraction;
        event.absolute_extent = event_absolute_fraction;
        event.passive_withdrawal_extent = event_absolute_fraction;
        event.maximum_local_fraction = maximum_local_fraction;
        event.maximum_local_tolerance_ratio = maximum_tolerance_ratio;
        event.candidate_start_positive_global_extent = 1;
        event.normalization_scale = 1;
        event.face_group_terms = 1;
        event.representative_active_id = 7;
        event.representative_passive_id = 8;
        event.representative_group = 0;
        event.representative_active_rank = 0;
        return validateIndividualRadiationDefect(event);
    }

    std::size_t defectRejectionsForTest() const
    {
        return standalone_defect_accounting_.defect_rejections;
    }

    void measureDefectForTest(double const withdrawal,
                              double const passive_extent,
                              double const roundoff_floor,
                              double const normalization_scale,
                              double& relative_fraction,
                              double& allowed_withdrawal,
                              double& tolerance_ratio) const
    {
        IndividualRadiationLocalDefectMeasure const measure =
            measureIndividualRadiationLocalDefect(
                withdrawal, passive_extent, roundoff_floor,
                normalization_scale);
        relative_fraction = measure.relative_fraction;
        allowed_withdrawal = measure.allowed_withdrawal;
        tolerance_ratio = measure.tolerance_ratio;
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
        // Preparation runs for both the all-active and reduced solve paths.
        if(reject_next_)
        {
            reject_next_ = false;
            ++rejected_candidates;
            throw std::runtime_error("forced grey candidate retry");
        }
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

void requireCollectively(bool condition, std::string const& message)
{
#ifdef RICH_MPI
    int passed = condition ? 1 : 0;
    MPI_Allreduce(MPI_IN_PLACE, &passed, 1, MPI_INT, MPI_MIN,
                  MPI_COMM_WORLD);
    condition = passed != 0;
#endif
    require(condition, message);
}

std::uint64_t runtimeUnsignedField(
    std::string const& output, std::string const& field)
{
    std::string const marker = " " + field + "=";
    std::size_t position = output.find(marker);
    if(position == std::string::npos)
        return std::numeric_limits<std::uint64_t>::max();
    position += marker.size();
    if(position == output.size() || output[position] < '0' ||
        output[position] > '9')
        return std::numeric_limits<std::uint64_t>::max();

    std::uint64_t value = 0;
    while(position < output.size() && output[position] >= '0' &&
        output[position] <= '9')
    {
        std::uint64_t const digit =
            static_cast<std::uint64_t>(output[position] - '0');
        if(value > (std::numeric_limits<std::uint64_t>::max() - digit) / 10)
            return std::numeric_limits<std::uint64_t>::max();
        value = 10 * value + digit;
        ++position;
    }
    return value;
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

void checkIndividualThermodynamicSlopeCoupling(bool shocked)
{
    std::vector<Vector3D> const points = makePoints(true);
    Voronoi3D tess(Vector3D(0, 0, 0), Vector3D(1, 1, 1));
    tess.Build(points);

    IdealGas eos(5.0 / 3.0);
    RigidWallGenerator3D ghost;
    std::vector<ComputationalCell3D> cells(points.size());
    double const shocked_compression_rate = 0.45;
    for(std::size_t index = 0; index < cells.size(); ++index)
    {
        Vector3D const& point = points[index];
        double const radius_squared =
            (point.x - 0.47) * (point.x - 0.47) +
            (point.y - 0.52) * (point.y - 0.52) +
            (point.z - 0.49) * (point.z - 0.49);
        cells[index].density = 1 + 0.02 * point.x + 0.01 * point.y;
        cells[index].pressure = 1 + 0.03 * point.x + 0.02 * point.z +
            0.08 * std::exp(-80 * radius_squared);
        cells[index].internal_energy = cells[index].pressure /
            (cells[index].density * (5.0 / 3.0 - 1.0));
        cells[index].velocity = shocked ?
            Vector3D(-shocked_compression_rate * (point.x - 0.5),
                     -shocked_compression_rate * (point.y - 0.5),
                     -shocked_compression_rate * (point.z - 0.5)) :
            Vector3D(0.02 * point.y, -0.015 * point.x, 0.01 * point.z);
    }

    LinearGauss3D global_reconstruction(eos, ghost);
    std::vector<std::pair<ComputationalCell3D, ComputationalCell3D> >
        global_face_values;
    global_reconstruction(tess, cells, 0, global_face_values);
    std::vector<Slope3D> const global_slopes =
        global_reconstruction.GetSlopes();
    std::vector<Slope3D> const unlimited_slopes =
        global_reconstruction.GetSlopesUnlimited();

    LinearGauss3D individual_reconstruction(eos, ghost);
    std::vector<unsigned char> const all_active(points.size(), 1);
    std::vector<std::pair<ComputationalCell3D, ComputationalCell3D> >
        individual_face_values;
    individual_reconstruction.InterpolateIndividual(
        tess, cells, 0, all_active, individual_face_values);
    std::vector<Slope3D> const individual_slopes =
        individual_reconstruction.GetSlopes();

    auto gradient_factor = [](double limited_x, double limited_y,
        double limited_z, double unlimited_x, double unlimited_y,
        double unlimited_z)
    {
        double const norm_squared = unlimited_x * unlimited_x +
            unlimited_y * unlimited_y + unlimited_z * unlimited_z;
        return (limited_x * unlimited_x + limited_y * unlimited_y +
            limited_z * unlimited_z) / norm_squared;
    };

    bool found_discriminating_cell = false;
    std::size_t interior_cells = 0;
    std::size_t nonzero_gradient_cells = 0;
    std::size_t owned_neighbor_cells = 0;
    std::size_t shock_window_cells = 0;
    double minimum_shock_weight = std::numeric_limits<double>::infinity();
    double maximum_shock_weight = -std::numeric_limits<double>::infinity();
    double maximum_limiter_difference = 0;
    for(std::size_t index = 0; index < cells.size(); ++index)
    {
        Vector3D const& point = points[index];
        if(point.x <= 0.25 || point.x >= 0.75 ||
           point.y <= 0.25 || point.y >= 0.75 ||
           point.z <= 0.25 || point.z >= 0.75)
            continue;
        ++interior_cells;
        Slope3D const& naive = unlimited_slopes[index];
        double const density_norm =
            naive.xderivative.density * naive.xderivative.density +
            naive.yderivative.density * naive.yderivative.density +
            naive.zderivative.density * naive.zderivative.density;
        double const pressure_norm =
            naive.xderivative.pressure * naive.xderivative.pressure +
            naive.yderivative.pressure * naive.yderivative.pressure +
            naive.zderivative.pressure * naive.zderivative.pressure;
        double const energy_norm =
            naive.xderivative.internal_energy *
                naive.xderivative.internal_energy +
            naive.yderivative.internal_energy *
                naive.yderivative.internal_energy +
            naive.zderivative.internal_energy *
                naive.zderivative.internal_energy;
        if(density_norm <= 1e-20 || pressure_norm <= 1e-20 ||
            energy_norm <= 1e-20)
            continue;
        ++nonzero_gradient_cells;

        double pressure_ratio = 1;
        bool all_neighbors_owned = true;
        for(std::size_t const face : tess.GetCellFaces(index))
        {
            std::pair<std::size_t, std::size_t> const neighbors =
                tess.GetFaceNeighbors(face);
            std::size_t const neighbor = neighbors.first == index ?
                neighbors.second : neighbors.first;
            if(neighbor >= cells.size())
            {
                all_neighbors_owned = false;
                break;
            }
            pressure_ratio = std::min(pressure_ratio,
                cells[index].pressure / cells[neighbor].pressure);
            pressure_ratio = std::min(pressure_ratio,
                cells[neighbor].pressure / cells[index].pressure);
        }
        if(!all_neighbors_owned)
            continue;
        ++owned_neighbor_cells;
        double const compression_weight = std::max(0.0, std::min(1.0,
            -(naive.xderivative.velocity.x +
              naive.yderivative.velocity.y +
              naive.zderivative.velocity.z) * tess.GetWidth(index) /
            (0.2 * eos.de2c(cells[index].density,
                            cells[index].internal_energy,
                            cells[index].tracers,
                            ComputationalCell3D::tracerNames))));
        double const pressure_weight = std::max(0.0, std::min(1.0,
            (0.7 - 0.8 * pressure_ratio) / (0.4 * 0.7)));
        double const shock_weight = std::max(
            compression_weight, pressure_weight);
        minimum_shock_weight = std::min(minimum_shock_weight, shock_weight);
        maximum_shock_weight = std::max(maximum_shock_weight, shock_weight);
        if((shocked && !(shock_weight > 0.55 && shock_weight < 0.80)) ||
           (!shocked && !(shock_weight < 0.45)))
            continue;
        ++shock_window_cells;

        Slope3D const& global = global_slopes[index];
        double const global_density_factor = gradient_factor(
            global.xderivative.density, global.yderivative.density,
            global.zderivative.density, naive.xderivative.density,
            naive.yderivative.density, naive.zderivative.density);
        double const global_pressure_factor = gradient_factor(
            global.xderivative.pressure, global.yderivative.pressure,
            global.zderivative.pressure, naive.xderivative.pressure,
            naive.yderivative.pressure, naive.zderivative.pressure);
        double const global_energy_factor = gradient_factor(
            global.xderivative.internal_energy,
            global.yderivative.internal_energy,
            global.zderivative.internal_energy,
            naive.xderivative.internal_energy,
            naive.yderivative.internal_energy,
            naive.zderivative.internal_energy);
        double const limiter_difference =
            std::abs(global_density_factor - global_pressure_factor);
        maximum_limiter_difference = std::max(
            maximum_limiter_difference, limiter_difference);
        if(limiter_difference <= 1e-5)
            continue;

        Slope3D const& individual = individual_slopes[index];
        double const individual_density_factor = gradient_factor(
            individual.xderivative.density, individual.yderivative.density,
            individual.zderivative.density, naive.xderivative.density,
            naive.yderivative.density, naive.zderivative.density);
        double const individual_pressure_factor = gradient_factor(
            individual.xderivative.pressure, individual.yderivative.pressure,
            individual.zderivative.pressure, naive.xderivative.pressure,
            naive.yderivative.pressure, naive.zderivative.pressure);
        double const individual_energy_factor = gradient_factor(
            individual.xderivative.internal_energy,
            individual.yderivative.internal_energy,
            individual.zderivative.internal_energy,
            naive.xderivative.internal_energy,
            naive.yderivative.internal_energy,
            naive.zderivative.internal_energy);
        double const common_factor = std::min(global_density_factor,
            std::min(global_pressure_factor, global_energy_factor));

        require(close(global_pressure_factor, global_energy_factor, 1e-10),
                "global pressure and internal-energy limiters diverged");
        if(shocked)
            require(close(individual_density_factor, common_factor, 1e-10) &&
                    close(individual_pressure_factor, common_factor, 1e-10) &&
                    close(individual_energy_factor, common_factor, 1e-10),
                    "shocked individual rho/P/e slopes did not use their common limiter");
        else
            require(close(individual_density_factor,
                          global_density_factor, 1e-10) &&
                    close(individual_pressure_factor,
                          global_pressure_factor, 1e-10) &&
                    close(individual_energy_factor,
                          global_energy_factor, 1e-10),
                    "smooth individual rho/P/e slopes were unnecessarily coupled");
        require(close(global.xderivative.velocity,
                      individual.xderivative.velocity, 1e-10) &&
                close(global.yderivative.velocity,
                      individual.yderivative.velocity, 1e-10) &&
                close(global.zderivative.velocity,
                      individual.zderivative.velocity, 1e-10),
                "thermodynamic coupling changed velocity slopes");
        found_discriminating_cell = true;
        break;
    }
    if(!found_discriminating_cell)
    {
        std::ostringstream diagnostic;
        diagnostic << (shocked ? "shocked" : "smooth")
                   << " slope-coupling test produced no qualifying cell"
                   << "; interior=" << interior_cells
                   << " nonzero_gradients=" << nonzero_gradient_cells
                   << " owned_neighbors=" << owned_neighbor_cells
                   << " shock_window=" << shock_window_cells
                   << " shock_weight_min=" << minimum_shock_weight
                   << " shock_weight_max=" << maximum_shock_weight
                   << " max_limiter_difference="
                   << maximum_limiter_difference;
        require(false, diagnostic.str());
    }
}

void testIndividualThermodynamicSlopeCoupling()
{
    checkIndividualThermodynamicSlopeCoupling(false);
    checkIndividualThermodynamicSlopeCoupling(true);
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
    IndividualRadiationDefectAccounting& defect =
        scheduler.radiationDefectAccounting();
    defect.cumulative_signed_extent = 0.25;
    defect.cumulative_absolute_extent = 0.75;
    defect.initial_positive_global_extent = 4096;
    defect.last_normalization_scale = 4096;
    defect.maximum_event_absolute_fraction = 8e-7;
    defect.maximum_local_fraction = 43.84254061743485;
    defect.maximum_local_tolerance_ratio = 0.02;
    defect.accepted_dirichlet_candidates = 17;
    defect.defect_rejections = 3;
    defect.defect_retry_substeps = 9;
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
    IndividualRadiationDefectAccounting const& restored_defect =
        restored.radiationDefectAccounting();
    require(restored_defect.cumulative_signed_extent ==
                defect.cumulative_signed_extent &&
            restored_defect.cumulative_absolute_extent ==
                defect.cumulative_absolute_extent &&
            restored_defect.maximum_local_fraction ==
                defect.maximum_local_fraction &&
            restored_defect.maximum_local_tolerance_ratio ==
                defect.maximum_local_tolerance_ratio &&
            restored_defect.config_version == 3 &&
            restored_defect.local_withdrawal_limit == 1e-2 &&
            restored_defect.local_absolute_limit == 1e-9 &&
            restored_defect.event_absolute_target == 1e-6,
            "scheduler restart changed radiation-defect policy bits");

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

void testMixedDirichletDefectAcceptance()
{
    IdealGas eos(5.0 / 3.0);
    RetryingIndividualRadiationDriver driver(eos, true, false);

    double relative_fraction = 0;
    double allowed_withdrawal = 0;
    double tolerance_ratio = 0;
    driver.measureDefectForTest(6.05e-10, 2e-8, 1e-9, 1,
                                relative_fraction, allowed_withdrawal,
                                tolerance_ratio);
    require(close(relative_fraction, 6.05e-10 / 2.1e-8, 2e-15),
            "mixed defect measurement changed its relative denominator");
    require(close(allowed_withdrawal, 1.21e-9, 2e-15),
            "mixed defect measurement did not apply the 1e-9 absolute term");
    require(close(tolerance_ratio, 0.5, 2e-15),
            "mixed defect measurement changed withdrawal/tolerance ratio");

    require(driver.validateDefectForTest(43.84254061743485, 0.02,
                                        8.238564529135027e-7),
            "mixed defect tolerance rejected a bounded floor-scale event");
    require(driver.validateDefectForTest(43.84254061743485, 1.0, 1e-6),
            "mixed defect tolerance rejected its exact acceptance boundary");
    std::size_t const rejection_count = driver.defectRejectionsForTest();
    require(driver.validateDefectForTest(43.84254061743485, 1.0001, 5e-7),
            "soft local defect target rejected a finite candidate");
    require(driver.getLastStepFailureReason().empty(),
            "soft local defect target created a hard failure");
    require(driver.validateDefectForTest(0.001, 0.5, 1.0001e-6),
            "soft event defect target rejected a finite candidate");
    require(driver.getLastStepFailureReason().empty(),
            "soft event defect target created a hard failure");
    require(driver.defectRejectionsForTest() == rejection_count,
            "soft defect targets incremented the hard rejection counter");
}

void testActiveTimeBinMask()
{
    std::vector<CellTimeState> states(4);
    states[0].time_bin = 1;
    states[1].time_bin = 3;
    states[2].time_bin = 3;
    states[3].time_bin = 62;

    IndividualStepContext context;
    context.active_mask = {1, 0, 1, 1};
    context.active_indices = {0, 2, 3};
    std::uint64_t const expected = (std::uint64_t(1) << 1) |
        (std::uint64_t(1) << 3) | (std::uint64_t(1) << 62);
    require(IndividualActiveTimeBinMask(context, states) == expected,
            "active-bin cache key omitted or added a scheduler bin");

    context.active_indices[1] = 1;
    bool rejected_inactive_index = false;
    try
    {
        (void)IndividualActiveTimeBinMask(context, states);
    }
    catch(std::out_of_range const&)
    {
        rejected_inactive_index = true;
    }
    require(rejected_inactive_index,
            "active-bin cache key accepted an inactive index");

    context.active_indices = {0};
    context.active_mask.pop_back();
    bool rejected_misaligned_state = false;
    try
    {
        (void)IndividualActiveTimeBinMask(context, states);
    }
    catch(std::invalid_argument const&)
    {
        rejected_misaligned_state = true;
    }
    require(rejected_misaligned_state,
            "active-bin cache key accepted misaligned scheduler state");
}

void testHydroFaceIntervalContract()
{
    IndividualStepContext context;
    context.event_tick = 5;
    context.time_quantum = 0.25;
    context.cell_time_steps = {1, 1.25};
    context.primitive_ticks = {4, 0};
    require(close(context.hydroFaceTimeStep(0, 1), 0.25) &&
            close(context.hydroFaceTimeStep(1, 0), 0.25),
            "hydro face interval repeated time before the last endpoint activation");
    require(close(context.faceTimeStep(0, 1), 1),
            "hydro elapsed-time correction changed radiation scheduled intervals");

    auto rejects_invalid_context = [](IndividualStepContext const& invalid)
    {
        try
        {
            (void)invalid.hydroFaceTimeStep(0, 1);
        }
        catch(std::invalid_argument const&)
        {
            return true;
        }
        return false;
    };
    IndividualStepContext invalid = context;
    invalid.time_quantum = 0;
    require(rejects_invalid_context(invalid),
            "hydro face interval accepted an absent time quantum");
    invalid = context;
    invalid.primitive_ticks.pop_back();
    require(rejects_invalid_context(invalid),
            "hydro face interval accepted missing endpoint activation ticks");
    invalid = context;
    invalid.primitive_ticks[0] = context.event_tick;
    require(rejects_invalid_context(invalid),
            "hydro face interval accepted a non-advancing activation tick");
    invalid.primitive_ticks[0] = context.event_tick + 1;
    require(rejects_invalid_context(invalid),
            "hydro face interval accepted a future activation tick");
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

// restore() must rebuild the finest occupied bin.  Left at its default of 0,
// the closure that follows a restart sent every overdue neighbour to the tick
// after the restart instead of the next tick of the finest bin in use.
void testRestoreMinimumOccupiedBin()
{
    std::vector<Vector3D> points = {
        Vector3D(0.2, 0.2, 0.2), Vector3D(0.8, 0.2, 0.2),
        Vector3D(0.2, 0.8, 0.2), Vector3D(0.8, 0.8, 0.2),
        Vector3D(0.2, 0.2, 0.8), Vector3D(0.8, 0.2, 0.8),
        Vector3D(0.2, 0.8, 0.8), Vector3D(0.8, 0.8, 0.8)};
    Voronoi3D tess(Vector3D(0, 0, 0), Vector3D(1, 1, 1));
#ifdef RICH_MPI
    int rank = 0;
    int rank_count = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &rank_count);
    for(Vector3D& point : points)
        point.x = (rank + point.x) / rank_count;
    tess.BuildParallel(points);
    points = tess.getAllPoints();
    tess.BuildPartiallyParallel(points, std::vector<double>(points.size(), 1.0),
        [&]() {std::vector<std::size_t> all(points.size()); std::iota(all.begin(), all.end(), 0); return all;}(),
        true, true);
#else
    int const rank = 0;
    tess.Build(points);
#endif
    std::vector<ComputationalCell3D> cells(points.size());
    for(std::size_t i = 0; i < cells.size(); ++i)
        cells[i].ID = 2100 + 100 * static_cast<std::size_t>(rank) + i;

    IndividualTimeStepOptions options;
    options.time_quantum = 1;
    std::uint64_t const B = std::uint64_t(1) << 30;
    std::vector<CellTimeState> states(cells.size());
    for(std::size_t i = 0; i < cells.size(); ++i)
    {
        states[i].cell_id = cells[i].ID;
        states[i].begin_tick = 0;
        states[i].last_primitive_tick = 0;
        states[i].end_tick = 4 * B;
        states[i].time_bin = 32;
    }
    // Only rank 0 holds the bin-30 cell, so the cache must be collective.
    if(rank == 0)
    {
        states[0].begin_tick = 2 * B;
        states[0].last_primitive_tick = 2 * B;
        states[0].end_tick = 3 * B;
        states[0].time_bin = 30;
    }
    IndividualTimeStepScheduler scheduler(options);
    scheduler.restore(cells, 0, 1, 2 * B, states);
    requireCollectively(scheduler.minimumOccupiedBin() == 30,
        "restore did not rebuild the finest occupied bin");

    scheduler.enforceNeighborBinClosure(tess, cells);
    int bad = 0;
    int lowered = 0;
    for(CellTimeState const& state : scheduler.states())
    {
        if(state.time_bin == 31)
        {
            ++lowered;
            // Overdue at bin 31 (allowance 2B ended at this tick): the cell
            // joins the next bin-30 tick, 3B, not the tick after this one.
            bad = bad || state.end_tick != 3 * B;
        }
        else if(state.time_bin == 30)
            bad = bad || state.end_tick != 3 * B;
        else
            bad = bad || state.time_bin != 32 || state.end_tick != 4 * B;
    }
#ifdef RICH_MPI
    MPI_Allreduce(MPI_IN_PLACE, &bad, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &lowered, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
#endif
    require(lowered > 0, "restore closure test lowered no neighbour");
    require(bad == 0, "closure after restore sent an overdue cell off the finest-bin grid");
}

// A conserved-change wake ends the cell at the next scheduled event, found
// after every other endpoint of the event is set, so it never creates an
// event and never depends on the spacing of past events.  Variants: a
// shifted origin with a non-dyadic quantum; an endpoint lowered after the
// commit (as AMR or the closure do) that the wakes must follow; a passive
// AMR merge recipient, which must be woken too.
void runChangeWakeFinalization(double const origin, double const quantum, bool const lower_after_commit,
    bool const merge, bool const signal = false)
{
    std::vector<Vector3D> points = {
        Vector3D(0.2, 0.2, 0.2), Vector3D(0.8, 0.2, 0.2),
        Vector3D(0.2, 0.8, 0.2), Vector3D(0.8, 0.8, 0.2),
        Vector3D(0.2, 0.2, 0.8), Vector3D(0.8, 0.2, 0.8),
        Vector3D(0.2, 0.8, 0.8), Vector3D(0.8, 0.8, 0.8)};
    Voronoi3D tess(Vector3D(0, 0, 0), Vector3D(1, 1, 1));
    int rank = 0;
#ifdef RICH_MPI
    int rank_count = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &rank_count);
    for(Vector3D& point : points)
        point.x = (rank + point.x) / rank_count;
    tess.BuildParallel(points);
    points = tess.getAllPoints();
    std::vector<std::size_t> all(points.size());
    std::iota(all.begin(), all.end(), 0);
    tess.BuildPartiallyParallel(points, std::vector<double>(points.size(), 1.0), all, true, true);
#else
    tess.Build(points);
#endif
    std::vector<ComputationalCell3D> cells(points.size());
    for(std::size_t i = 0; i < cells.size(); ++i)
        cells[i].ID = 2300 + 100 * static_cast<std::size_t>(rank) + i;

    IndividualTimeStepOptions options;
    options.time_quantum = quantum;
    options.initial_bin = 3;
    options.maximum_bin = 6;
    IndividualTimeStepScheduler scheduler(options);
    scheduler.initialize(cells, origin, 8 * quantum);
    auto const collective_event = [&]()
    {
        std::uint64_t tick = scheduler.nextEventTick();
#ifdef RICH_MPI
        MPI_Allreduce(MPI_IN_PLACE, &tick, 1, MPI_UINT64_T, MPI_MIN, MPI_COMM_WORLD);
#endif
        return scheduler.prepareEvent(cells, tick);
    };
    // Event at tick 8, every cell active: rank 0's first cell drops to bin 1.
    IndividualStepContext event = collective_event();
    std::vector<double> limits(cells.size(), std::numeric_limits<double>::infinity());
    if(rank == 0 && !limits.empty())
        limits[0] = 2 * quantum;
    scheduler.commitEvent(event, tess, cells, limits);
    requireCollectively(scheduler.finalizeChangeWakes(true).applied == 0,
        "change-wake finalization applied a wake nobody requested");

    // Event at tick 10: only the bin-1 cell is active.  Passive cells that
    // still end at 16 ask for a change wake, except (merge variant) the last
    // two on rank 0, which merge instead: the last is removed into the other.
    event = collective_event();
    requireCollectively(event.event_tick == 10, "change-wake test event is not at tick 10");
    std::vector<std::size_t> far;
    for(std::size_t i = 0; i < cells.size(); ++i)
        if(!event.isActive(i) && scheduler.states()[i].end_tick == 16)
            far.push_back(i);
    std::size_t recipient = cells.size();
    std::size_t removed = cells.size();
    if(merge && rank == 0 && far.size() >= 3)
    {
        removed = far.back();
        far.pop_back();
        recipient = far.back();
        far.pop_back();
    }
    std::vector<double> ratios(cells.size(), 0.0);
    for(std::size_t const i : far)
        ratios[i] = 0.3;
    unsigned long requested = 0;
    limits.assign(cells.size(), std::numeric_limits<double>::infinity());
    // Signal variant: a physical wake 1.5 quanta ahead (quantized down to one
    // tick) creates the event at 11, and the change wakes join it.
    std::vector<double> signal_wakes;
    if(signal)
    {
        signal_wakes.assign(cells.size(), std::numeric_limits<double>::infinity());
        if(rank == 0 && !far.empty())
        {
            signal_wakes[far.front()] = 1.5 * quantum;
            ratios[far.front()] = 0;
            far.erase(far.begin());
        }
    }
    requested = static_cast<unsigned long>(far.size());
    scheduler.commitEvent(event, tess, cells, limits, signal_wakes, ratios);
    std::uint64_t expected = signal ? 11 : 12;
    if(lower_after_commit)
    {
        // As AMR or the closure do after the commit: one cell now ends at 11.
        if(rank == 0 && !far.empty())
            scheduler.states()[far.front()].end_tick = 11;
        expected = 11;
    }
    std::size_t recipient_id = 0;
    if(merge)
    {
        IndividualAMRChangeSet changes;
        std::vector<ComputationalCell3D> remaining;
        for(std::size_t i = 0; i < cells.size(); ++i)
            if(i != removed)
                remaining.push_back(cells[i]);
        if(removed < cells.size())
        {
            changes.removed_cell_ids.push_back(cells[removed].ID);
            changes.merge_targets.push_back({cells[removed].ID, cells[recipient].ID, 0});
            recipient_id = cells[recipient].ID;
        }
        scheduler.applyAMRChangeSet(remaining, changes);
        cells = remaining;
    }
#ifdef RICH_MPI
    MPI_Allreduce(MPI_IN_PLACE, &requested, 1, MPI_UNSIGNED_LONG, MPI_SUM, MPI_COMM_WORLD);
    unsigned long long merged = recipient_id != 0 ? 1 : 0;
    MPI_Allreduce(MPI_IN_PLACE, &merged, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
#else
    unsigned long long const merged = recipient_id != 0 ? 1 : 0;
#endif
    requireCollectively(!merge || merged == 1, "change-wake merge variant found no passive pair");
    IndividualTimeStepScheduler::ChangeWakeFinalization const wakes =
        scheduler.finalizeChangeWakes(true);
    std::uint64_t const woken = requested + merged;
    // Under the reference rule (RICH_INDIVIDUAL_CHANGE_WAKE_RULE=spacing_ticks, run in its own
    // invocation) a wake ends at the event plus the largest power of two within the last spacing
    // (10 - 8 = 2), i.e. 12, whatever the next event is; the next event itself is unchanged here.
    bool const join = IndividualTimeStepScheduler::changeWakesJoinNextEvent();
    std::uint64_t const woken_end = join ? expected : 12;
    requireCollectively(requested > 0 && wakes.applied == woken &&
        wakes.shortened == woken - (lower_after_commit ? 1 : 0) &&
        wakes.next_tick_before == expected && wakes.next_tick_after == expected &&
        wakes.merge_wakes == merged && close(wakes.largest_ratio, 0.3),
        "change-wake finalization did not end the woken cells at the next event");
    int bad = 0;
    for(CellTimeState const& state : scheduler.states())
    {
        bad = bad || state.change_wake_pending != 0;
        if(state.cell_id == recipient_id)
            bad = bad || state.end_tick != woken_end;
    }
    for(std::size_t const i : far)
    {
        // Indices are unchanged: the removed cell was the last passive one.
        // The lowered cell (variant) keeps its earlier endpoint 11 under both rules.
        CellTimeState const& state = scheduler.states()[i];
        bool const lowered = lower_after_commit && rank == 0 && i == far.front();
        bad = bad || state.end_tick != (lowered ? 11 : woken_end) || state.time_bin != 3;
    }
    requireCollectively(bad == 0, "a woken cell kept its request, missed the next event or changed bin");
    requireCollectively(scheduler.finalizeChangeWakes(true).applied == 0,
        "change-wake finalization is not idempotent");
}

void testChangeWakeFinalization()
{
    runChangeWakeFinalization(0, 1, false, false);
    // A TDE-like clock: origin 20.8 and a quantum that is not a power of two.
    runChangeWakeFinalization(20.8214548773, 0.000931322574615 * 1.37 / 1024, false, false);
    runChangeWakeFinalization(20.8214548773, 0.000931322574615 * 1.37 / 1024, true, false);
    runChangeWakeFinalization(0, 1, false, true);
    runChangeWakeFinalization(20.8214548773, 0.000931322574615 * 1.37 / 1024, false, false, true);

    // Hydro-side accuracy accounting: issued once, then sampled at activation
    // or censored with the ratio it had reached, exactly once.
    IndividualChangeWakeAccounting accounting;
    std::vector<IndividualConservedChange> changes(3);
    for(IndividualConservedChange& change : changes)
    {
        change.mass_at_activation = 1;
        change.energy_at_activation = -2;
    }
    changes[0].mass_abs_change = 0.3;
    changes[1].energy_abs_change = 1.6;
    changes[2].mass_abs_change = 0.9;
    accounting.Issue(changes[0]);
    accounting.Issue(changes[0]);
    accounting.Issue(changes[1]);
    accounting.SampleAtActivation(changes[2], 12, 0.5);
    accounting.SampleAtActivation(changes[0], 10, 0.5);
    accounting.SampleAtActivation(changes[0], 10, 0.5);
    IndividualChangeWakeAccounting::Tally tally = accounting.GetTally();
    require(tally.issued == 2 && tally.sampled == 1 && tally.sampled_above == 0 && close(tally.sampled_largest, 0.3) &&
        tally.sampled_largest_id == 10 && !changes[0].change_woken,
        "change-wake accounting did not sample an issued cell exactly once");
    accounting.Drain(changes, 0.5);
    accounting.Drain(changes, 0.5);
    tally = accounting.GetTally();
    require(tally.censored == 1 && tally.censored_above == 1 && close(tally.censored_largest, 0.8) &&
        !changes[1].change_woken, "change-wake accounting lost a censored violation or censored twice");
    accounting.ClearTally();
    require(accounting.GetTally().issued == 0 && accounting.GetTally().censored == 0,
        "change-wake accounting tally did not clear");

    // Change after issue: the snapshot is taken at the first issue only and
    // subtracted per component, with the activation denominators.  Issued at
    // mass 0.3 (energy 0.1 of 2); a repeated issue after more change must not
    // move the snapshot; at activation mass has grown by 0.1 and energy by
    // 0.9, so the after-issue ratio is energy's 0.45, not mass's 0.1.
    IndividualConservedChange late;
    late.mass_at_activation = 1;
    late.energy_at_activation = 2;
    late.mass_abs_change = 0.3;
    late.energy_abs_change = 0.1;
    accounting.Issue(late);
    late.mass_abs_change = 0.35;
    late.energy_abs_change = 0.5;
    accounting.Issue(late);
    require(late.mass_abs_change_at_issue == 0.3 && late.energy_abs_change_at_issue == 0.1,
        "a repeated wake request moved the first-issue snapshot");
    late.mass_abs_change = 0.4;
    late.energy_abs_change = 1.0;
    accounting.SampleAtActivation(late, 20, 0.4);
    tally = accounting.GetTally();
    require(tally.issued == 1 && close(tally.issued_largest, 0.3) && tally.sampled == 1 &&
        close(tally.sampled_largest, 0.5) && tally.sampled_above == 1 &&
        close(tally.sampled_after_issue_largest, 0.45) && tally.sampled_after_issue_above == 1,
        "change-wake accounting mixed the at-issue and after-issue change");
    // Outstanding counts marks until each is sampled or censored.
    std::vector<IndividualConservedChange> pending(3);
    for(IndividualConservedChange& change : pending)
        change.mass_at_activation = 1;
    accounting.Issue(pending[0]);
    accounting.Issue(pending[2]);
    require(IndividualChangeWakeAccounting::Outstanding(pending) == 2, "outstanding did not count two marks");
    accounting.SampleAtActivation(pending[0], 30, 0.5);
    require(IndividualChangeWakeAccounting::Outstanding(pending) == 1, "a sampled mark stayed outstanding");
    accounting.Drain(pending, 0.5);
    require(IndividualChangeWakeAccounting::Outstanding(pending) == 0, "a censored mark stayed outstanding");
    accounting.ClearTally();
#ifdef RICH_MPI
    // The migration serializer carries the request.
    CellTimeState state;
    state.cell_id = 7;
    state.begin_tick = 3;
    state.end_tick = 9;
    state.time_bin = 2;
    state.change_wake_pending = 1;
    state.change_wake_ratio = 0.625;
    state.gravity_half_kick_pending = true;
    Serializer serializer;
    state.dump(&serializer);
    CellTimeState copy;
    copy.load(&serializer, 0);
    require(copy.cell_id == 7 && copy.begin_tick == 3 && copy.end_tick == 9 && copy.time_bin == 2 &&
        copy.change_wake_pending == 1 && copy.change_wake_ratio == 0.625 && copy.gravity_half_kick_pending,
        "CellTimeState serializer lost the change-wake request");
#endif
}

// An anchor grid sets the quantum; the first interval is the coarsest bin not
// longer than the initial step, whether the anchor is longer or shorter.
void testAnchoredInitialization()
{
    std::vector<ComputationalCell3D> cells(4);
    for(std::size_t i = 0; i < cells.size(); ++i)
        cells[i].ID = 2500 + i;
    IndividualTimeStepOptions options;
    options.initial_bin = 3;
    options.maximum_bin = 10;
    {
        // Anchor 1.5: quantum 1.5/8; 1.0 holds 5.33 quanta -> bin 2 (0.75).
        IndividualTimeStepScheduler scheduler(options);
        scheduler.initialize(cells, 0, 1.0, 1.5);
        require(close(scheduler.timeQuantum(), 0.1875) && scheduler.minimumOccupiedBin() == 2,
            "anchored initialization chose the wrong quantum or first bin");
        for(CellTimeState const& state : scheduler.states())
            require(state.time_bin == 2 && state.end_tick == 4, "anchored first interval exceeds the initial step");
    }
    {
        // Anchor 0.25 (shorter): quantum 0.25/8; 1.0 is exactly bin 5.
        IndividualTimeStepScheduler scheduler(options);
        scheduler.initialize(cells, 0, 1.0, 0.25);
        for(CellTimeState const& state : scheduler.states())
            require(state.time_bin == 5 && state.end_tick == 32,
                "anchored initialization did not take the coarsest bin within the initial step");
    }
    {
        // No anchor: unchanged.
        IndividualTimeStepScheduler scheduler(options);
        scheduler.initialize(cells, 0, 1.0);
        require(close(scheduler.timeQuantum(), 0.125), "unanchored quantum changed");
        for(CellTimeState const& state : scheduler.states())
            require(state.time_bin == 3 && state.end_tick == 8, "unanchored first interval changed");
    }
    bool threw = false;
    try
    {
        IndividualTimeStepScheduler scheduler(options);
        scheduler.initialize(cells, 0, 0.001, 1.0);
    }
    catch(std::runtime_error const&)
    {
        threw = true;
    }
    require(threw, "an initial step below one anchored quantum was accepted");
}

// The reference wake rule (RICH_INDIVIDUAL_CHANGE_WAKE_RULE=spacing_ticks) ends a
// woken cell at the event plus the largest power of two within the spacing of
// the event just committed: spacings 3, 4, 5 give 2, 4, 4.  Only rank 0 holds
// the active and the woken cell; with spacing 3 the wake (at 15) creates the
// next event before the active cell's (16), so the global minimum moves on
// ranks with no local wake.  Under the default rule the wake joins 16.
void testChangeWakeSpacingReference()
{
    std::vector<Vector3D> points = {
        Vector3D(0.2, 0.2, 0.2), Vector3D(0.8, 0.2, 0.2),
        Vector3D(0.2, 0.8, 0.2), Vector3D(0.8, 0.8, 0.2),
        Vector3D(0.2, 0.2, 0.8), Vector3D(0.8, 0.2, 0.8),
        Vector3D(0.2, 0.8, 0.8), Vector3D(0.8, 0.8, 0.8)};
    Voronoi3D tess(Vector3D(0, 0, 0), Vector3D(1, 1, 1));
    int rank = 0;
#ifdef RICH_MPI
    int rank_count = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &rank_count);
    for(Vector3D& point : points)
        point.x = (rank + point.x) / rank_count;
    tess.BuildParallel(points);
    points = tess.getAllPoints();
    std::vector<std::size_t> all(points.size());
    std::iota(all.begin(), all.end(), 0);
    tess.BuildPartiallyParallel(points, std::vector<double>(points.size(), 1.0), all, true, true);
#else
    tess.Build(points);
#endif
    std::vector<ComputationalCell3D> cells(points.size());
    for(std::size_t i = 0; i < cells.size(); ++i)
        cells[i].ID = 2700 + 100 * static_cast<std::size_t>(rank) + i;
    // The active cell (smallest x, y, z) and the woken cell (largest), on rank 0.
    auto const extreme = [&](bool largest)
    {
        std::size_t best = 0;
        for(std::size_t i = 1; i < points.size(); ++i)
        {
            double const a = points[i].x + points[i].y + points[i].z;
            double const b = points[best].x + points[best].y + points[best].z;
            if(largest ? a > b : a < b)
                best = i;
        }
        return best;
    };
    std::size_t const active = extreme(false);
    std::size_t const woken = extreme(true);
    bool const join = IndividualTimeStepScheduler::changeWakesJoinNextEvent();
    IndividualTimeStepOptions options;
    options.time_quantum = 1;
    for(std::uint64_t const event : {std::uint64_t(13), std::uint64_t(14), std::uint64_t(15)})
    {
        std::uint64_t const spacing = event - 10;
        std::uint64_t const power = spacing >= 4 ? 4 : 2;
        std::uint64_t const expected_end = join ? 16 : event + power;
        std::uint64_t const expected_next = std::min<std::uint64_t>(16, expected_end);
        std::vector<CellTimeState> states(cells.size());
        for(std::size_t i = 0; i < cells.size(); ++i)
        {
            bool const is_active = rank == 0 && i == active;
            states[i].cell_id = cells[i].ID;
            states[i].begin_tick = is_active ? 2 : 0;
            states[i].last_primitive_tick = states[i].begin_tick;
            states[i].time_bin = is_active ? 3 : 5;
            states[i].end_tick = is_active ? event : 64;
        }
        IndividualTimeStepScheduler scheduler(options);
        scheduler.restore(cells, 0, 1, 10, states);
        IndividualStepContext const context = scheduler.prepareEvent(cells, event);
        std::vector<double> limits(cells.size(), std::numeric_limits<double>::infinity());
        std::vector<double> ratios(cells.size(), 0.0);
        if(rank == 0)
            ratios[woken] = 0.3;
        scheduler.commitEvent(context, tess, cells, limits, std::vector<double>(), ratios);
        IndividualTimeStepScheduler::ChangeWakeFinalization const wakes = scheduler.finalizeChangeWakes(true);
        bool const ok_rank = rank != 0 || scheduler.states()[woken].end_tick == expected_end;
        requireCollectively(ok_rank && wakes.applied == 1 && wakes.next_tick_before == 16 &&
            wakes.next_tick_after == expected_next,
            "change-wake reference rule gave the wrong deadline or next event");
    }
}

// Under the bin-spread cap K (default 2; RICH_INDIVIDUAL_MAX_BIN_SPREAD sets it
// per invocation, -1 turns it off and the test does nothing) no cell's bin
// exceeds the fixed ceiling, anchor bin (initial_bin) + K, after a commit,
// including its neighbour propagation, after synchronized limits, uniform
// aligned growth, anchored initialization or forced-synchronized growth; a
// capped cell keeps an end within its new allowance or, if already overdue, at
// the next finest-bin tick.
void testBinSpreadCap()
{
    int const configured = IndividualTimeStepScheduler::maximumBinSpread();
    if(configured < 0)
        return;
    unsigned const spread = static_cast<unsigned>(configured);
    std::vector<Vector3D> points;
    for(double const x : {0.1, 0.3, 0.5, 0.7, 0.9})
        for(double const y : {0.3, 0.7})
            for(double const z : {0.3, 0.7})
                points.emplace_back(x, y, z);
    Voronoi3D tess(Vector3D(0, 0, 0), Vector3D(1, 1, 1));
    int rank = 0;
#ifdef RICH_MPI
    int rank_count = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &rank_count);
    for(Vector3D& point : points)
        point.x = (rank + point.x) / rank_count;
    tess.BuildParallel(points);
    points = tess.getAllPoints();
    std::vector<std::size_t> all(points.size());
    std::iota(all.begin(), all.end(), 0);
    tess.BuildPartiallyParallel(points, std::vector<double>(points.size(), 1.0), all, true, true);
#else
    tess.Build(points);
#endif
    std::vector<ComputationalCell3D> cells(points.size());
    for(std::size_t i = 0; i < cells.size(); ++i)
        cells[i].ID = 2900 + 100 * static_cast<std::size_t>(rank) + i;
    IndividualTimeStepOptions options;
    options.time_quantum = 1;
    options.initial_bin = 6;
    options.maximum_bin = 8;
    auto const check = [&](IndividualTimeStepScheduler const& scheduler, std::uint64_t event, char const* where)
    {
        IndividualTimeStepOptions const& options = scheduler.options();
        unsigned lowest = options.maximum_bin;
        unsigned highest = 0;
        int bad = 0;
        for(CellTimeState const& state : scheduler.states())
        {
            lowest = std::min<unsigned>(lowest, state.time_bin);
            highest = std::max<unsigned>(highest, state.time_bin);
            std::uint64_t const allowance_end = state.begin_tick + (std::uint64_t(1) << state.time_bin);
            std::uint64_t const finest_tick = event + (std::uint64_t(1) << scheduler.minimumOccupiedBin()) -
                event % (std::uint64_t(1) << scheduler.minimumOccupiedBin());
            bad = bad || state.end_tick <= event ||
                (state.end_tick > allowance_end && state.end_tick > finest_tick);
        }
#ifdef RICH_MPI
        MPI_Allreduce(MPI_IN_PLACE, &lowest, 1, MPI_UNSIGNED, MPI_MIN, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, &highest, 1, MPI_UNSIGNED, MPI_MAX, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, &bad, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
#endif
        // The cap is a fixed ceiling, anchor bin + K.
        require(highest <= std::min<unsigned>(options.maximum_bin, options.initial_bin + spread) &&
            scheduler.minimumOccupiedBin() == lowest,
            std::string("bin spread above the cap after ") + where);
        require(bad == 0, std::string("a capped cell ended outside its allowance after ") + where);
    };
    {
        // Every cell active at 64 (bin 6); rank 0's first cell drops to bin 1,
        // its neighbours follow by propagation, the rest must be capped.
        IndividualTimeStepScheduler scheduler(options);
        scheduler.initialize(cells, 0, 64);
        std::uint64_t tick = scheduler.nextEventTick();
#ifdef RICH_MPI
        MPI_Allreduce(MPI_IN_PLACE, &tick, 1, MPI_UINT64_T, MPI_MIN, MPI_COMM_WORLD);
#endif
        IndividualStepContext const event = scheduler.prepareEvent(cells, tick);
        std::vector<double> limits(cells.size(), std::numeric_limits<double>::infinity());
        if(rank == 0)
            limits[0] = 2;
        scheduler.commitEvent(event, tess, cells, limits);
        check(scheduler, event.event_tick, "a commit");
        // A cell pinned below the anchor (bin 6) must not drag the others: cells
        // away from it keep bins at or above the anchor.
        unsigned highest = 0;
        for(CellTimeState const& state : scheduler.states())
            highest = std::max<unsigned>(highest, state.time_bin);
#ifdef RICH_MPI
        MPI_Allreduce(MPI_IN_PLACE, &highest, 1, MPI_UNSIGNED, MPI_MAX, MPI_COMM_WORLD);
#endif
        require(highest >= options.initial_bin, "a cell below the anchor dragged every other cell down");
    }
    {
        // Synchronized limits at the start: one cell to bin 1 on rank 0.
        IndividualTimeStepScheduler scheduler(options);
        scheduler.initialize(cells, 0, 64);
        std::vector<std::uint8_t> maximum_bins(cells.size(), options.maximum_bin);
        if(rank == 0)
            maximum_bins[0] = 1;
        scheduler.limitSynchronizedBins(maximum_bins);
        check(scheduler, 0, "synchronized limits");
    }
    if(spread == 2)
    {
        // K = 2 only (the expected bins below assume it).
        // Propagation lowers the minimum after the first refresh.  Restored at
        // tick 24 with anchor bin 3 (ceiling 5): rank 0's first cell (begin 24,
        // end 32, bin 3, pending neighbour bin 1) completes its nominal
        // interval at 32 and grows to bin 4, so the refresh before propagation
        // sees minimum 4; every other cell sits in bin 5 (begin 16, end 48, at
        // the ceiling).  Propagation then lowers the first cell's neighbours to
        // bin 2; the fixed ceiling does not follow that minimum, so the distant
        // bin-5 cells keep bin 5 and end 48 (a finest-bin + K cap would have
        // cut them to bin 4, ending at 36).
        std::vector<CellTimeState> states(cells.size());
        for(std::size_t i = 0; i < cells.size(); ++i)
        {
            bool const source = rank == 0 && i == 0;
            states[i].cell_id = cells[i].ID;
            states[i].begin_tick = source ? 24 : 16;
            states[i].last_primitive_tick = states[i].begin_tick;
            states[i].end_tick = source ? 32 : 48;
            states[i].time_bin = source ? 3 : 5;
            states[i].pending_neighbor_bin = source ? 1 : std::numeric_limits<std::uint8_t>::max();
        }
        // Anchor bin 3, so the ceiling is bin 5 and the bin-5 cells are not capped.
        IndividualTimeStepOptions low_anchor = options;
        low_anchor.initial_bin = 3;
        IndividualTimeStepScheduler scheduler(low_anchor);
        scheduler.restore(cells, 0, 1, 24, states);
        IndividualStepContext const event = scheduler.prepareEvent(cells, 32);
        std::vector<double> limits(cells.size(), std::numeric_limits<double>::infinity());
        scheduler.commitEvent(event, tess, cells, limits);
        check(scheduler, 32, "propagation after the first refresh");
        // Propagation lowers the minimum to bin 2, but a fixed ceiling does not
        // follow it: distant bin-5 cells keep their bin and their end, 48.
        int kept = 0;
        int bad = 0;
        for(CellTimeState const& state : scheduler.states())
            if(state.begin_tick == 16)
            {
                kept += state.time_bin == 5 ? 1 : 0;
                bad = bad || (state.time_bin == 5 && state.end_tick != 48) || state.time_bin > 5;
            }
        int lowest = scheduler.minimumOccupiedBin();
#ifdef RICH_MPI
        MPI_Allreduce(MPI_IN_PLACE, &kept, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, &bad, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
#endif
        require(lowest == 2, "propagation did not lower the minimum to bin 2");
        require(kept > 0 && bad == 0, "a fine minimum dragged distant cells below the fixed ceiling");
    }
    // Anchor bin 2: the ceiling min(8, 2 + K) sits below the maximum bin for K < 6.
    IndividualTimeStepOptions low = options;
    low.initial_bin = 2;
    unsigned const ceiling = std::min<unsigned>(low.maximum_bin, low.initial_bin + spread);
    // Every cell grows at its aligned ends until the ceiling stops it, both
    // with individual bins and with one forced-synchronized shared bin.
    for(bool const synchronized : {false, true})
    {
        IndividualTimeStepOptions grow = low;
        grow.force_synchronized = synchronized;
        IndividualTimeStepScheduler scheduler(grow);
        scheduler.initialize(cells, 0, 4);
        unsigned reached = 0;
        for(int step = 0; step < 24; ++step)
        {
            std::uint64_t tick = scheduler.nextEventTick();
#ifdef RICH_MPI
            MPI_Allreduce(MPI_IN_PLACE, &tick, 1, MPI_UINT64_T, MPI_MIN, MPI_COMM_WORLD);
#endif
            IndividualStepContext const event = scheduler.prepareEvent(cells, tick);
            std::vector<double> limits(cells.size(), std::numeric_limits<double>::infinity());
            scheduler.commitEvent(event, tess, cells, limits);
            check(scheduler, event.event_tick, synchronized ? "forced-synchronized growth" : "uniform aligned growth");
            reached = scheduler.minimumOccupiedBin();
        }
        require(reached == ceiling, synchronized ? "forced-synchronized growth did not reach the ceiling" :
            "uniform aligned growth did not reach the ceiling");
    }
    {
        // Anchored initialization with an initial step far above the ceiling:
        // quantum 1/4 (anchor 1, bin 2), initial step 1024 is bin 12 uncapped.
        IndividualTimeStepOptions anchored = low;
        anchored.time_quantum = 0;
        IndividualTimeStepScheduler scheduler(anchored);
        scheduler.initialize(cells, 0, 1024, 1);
        int bad = scheduler.minimumOccupiedBin() == ceiling ? 0 : 1;
        for(CellTimeState const& state : scheduler.states())
            bad = bad || state.time_bin != ceiling;
        requireCollectively(bad == 0, "anchored initialization started above the bin-spread ceiling");
    }
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
    states[0].last_primitive_tick = 8;
    states[0].end_tick = 9;
    states[0].time_bin = 0;
    states[1].begin_tick = 8;
    states[1].last_primitive_tick = 8;
    states[1].end_tick = 16;
    states[1].time_bin = 3;
    std::vector<CellTimeState> inconsistent_states = states;
    inconsistent_states[1].last_primitive_tick = 7;
    bool inconsistent_restart_rejected = false;
    try
    {
        IndividualTimeStepScheduler invalid_scheduler(options);
        invalid_scheduler.restore(cells, 0, 1, 8, inconsistent_states);
    }
    catch(std::invalid_argument const&)
    {
        inconsistent_restart_rejected = true;
    }
    require(inconsistent_restart_rejected,
            "scheduler accepted a restart with mismatched hydro activation ticks");
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
    require(close(event.hydroFaceTimeStep(0, 1), 1),
            "aligned wake repeated an earlier face interval");
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

void testHydroWakeFaceIntervals()
{
    std::vector<Vector3D> const points = {
        Vector3D(0.25, 0.5, 0.5), Vector3D(0.75, 0.5, 0.5)};
    Voronoi3D tess(Vector3D(0, 0, 0), Vector3D(1, 1, 1));
    tess.Build(points);
    std::vector<ComputationalCell3D> cells(points.size());
    for(std::size_t cell = 0; cell < cells.size(); ++cell)
    {
        cells[cell].ID = 3050 + cell;
        cells[cell].density = 1;
        cells[cell].pressure = 1;
        cells[cell].internal_energy = 1;
    }
    std::size_t shared_face = tess.GetTotalFacesNumber();
    for(std::size_t face = 0; face < tess.GetTotalFacesNumber(); ++face)
        if(!tess.BoundaryFace(face))
        {
            auto const neighbors = tess.GetFaceNeighbors(face);
            if(neighbors.first < cells.size() && neighbors.second < cells.size())
            {
                shared_face = face;
                break;
            }
        }
    require(shared_face < tess.GetTotalFacesNumber(),
            "hydro wake test found no shared face");
    auto const neighbors = tess.GetFaceNeighbors(shared_face);

    for(bool const signal_wake : {true, false})
    {
        IndividualTimeStepOptions options;
        options.time_quantum = 1;
        options.initial_bin = 3;
        options.maximum_bin = 3;
        IndividualTimeStepScheduler initial(options);
        initial.initialize(cells, 0, 8);
        std::vector<CellTimeState> states = initial.states();
        states[0].end_tick = 4;
        states[0].time_bin = 2;
        IndividualTimeStepScheduler scheduler(options);
        scheduler.restore(cells, 0, 1, 0, states);
        scheduler.enforceNeighborBinClosure(tess, cells);

        std::vector<Conserved3D> extensives(cells.size());
        for(Conserved3D& extensive : extensives)
        {
            extensive.mass = 1;
            extensive.energy = 1;
            extensive.internal_energy = 1;
            extensive.Erad = 1;
            extensive.Eg[0] = 1;
        }
        std::vector<Conserved3D> fluxes(tess.GetTotalFacesNumber());
        fluxes[shared_face].mass = 0.1 / tess.GetArea(shared_face);
        DefaultExtensiveUpdater updater;
        std::uint64_t last_face_tick = 0;
        std::vector<std::uint64_t> const event_ticks = signal_wake ?
            std::vector<std::uint64_t>{3, 4, 5, 8} :
            std::vector<std::uint64_t>{4, 5, 6, 8};
        for(std::uint64_t const tick : event_ticks)
        {
            IndividualStepContext const event = scheduler.prepareEvent(cells, tick);
            if(!event.active_indices.empty())
            {
                require(close(event.hydroFaceTimeStep(0, 1),
                              static_cast<double>(tick - last_face_tick)),
                        "woken face did not use its elapsed integration interval");
                updater.UpdateIndividual(fluxes, tess, event, cells, extensives,
                    event.event_time, {}, {}, {});
                require(close(extensives[neighbors.first].mass, 1 - 0.1 * tick) &&
                        close(extensives[neighbors.second].mass, 1 + 0.1 * tick),
                        "woken face transferred the wrong time-integrated mass");
                require(close(extensives[0].mass + extensives[1].mass, 2),
                        "woken face failed to conserve mass");
                last_face_tick = tick;
            }

            std::vector<double> limits(cells.size(),
                std::numeric_limits<double>::infinity());
            for(std::size_t const active : event.active_indices)
                limits[active] = event.nominalCellTimeStep(active);
            std::vector<double> wake_deadlines(cells.size(),
                std::numeric_limits<double>::infinity());
            // A collective event on another rank can arrive before either
            // local cell is active. Shorten only cell 1 at that event.
            if(tick == (signal_wake ? 3u : 5u))
            {
                require(event.active_indices.empty(),
                        "hydro wake test unexpectedly activated a local cell");
                if(signal_wake)
                    wake_deadlines[1] = 2;
                else
                    limits[1] = 2;
            }
            scheduler.commitEvent(event, tess, cells, limits, wake_deadlines);
        }
        require(last_face_tick == 8,
                "hydro wake regression did not finish the shared interval");
    }
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
    // This fixture isolates forced-event timing with adjacent bins 1 and 3.
    options.maximum_neighbor_bin_difference = 2;
    IndividualTimeStepScheduler initial_scheduler(options);
    initial_scheduler.initialize(cells, 0, 4);

    std::vector<CellTimeState> states = initial_scheduler.states();
    states[0].begin_tick = 8;
    states[0].last_primitive_tick = 8;
    states[0].end_tick = 10;
    states[0].time_bin = 1;
    states[1].begin_tick = 8;
    states[1].last_primitive_tick = 8;
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
    require(close(forced.hydroFaceTimeStep(0, 1), 1),
            "forced all-active event used an incorrect face interval");
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
            scheduler.states()[1].time_bin == 3 &&
            scheduler.states()[1].end_tick == 16,
        "forced event replaced a cell's physical bin with its catch-up interval");
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
	context.time_quantum = 1;
	context.event_tick = 1;
	context.event_time = 1;
	context.primitive_ticks.assign(points.size(), 0);

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

	// Mass and energy can point toward opposite endpoints on a moving face.
	// A valid transfer must preserve both its direction and its full magnitude.
	std::vector<Conserved3D> opposite_extensives(points.size());
	for(Conserved3D& extensive : opposite_extensives)
	{
		extensive.mass = 1;
		extensive.energy = 1;
		extensive.internal_energy = 1;
		extensive.Erad = 1;
		extensive.Eg[0] = 1;
	}
	std::vector<Conserved3D> opposite_fluxes(tess.GetTotalFacesNumber());
	opposite_fluxes[target_face].mass = 0.25 / tess.GetArea(target_face);
	opposite_fluxes[target_face].energy = -0.125 / tess.GetArea(target_face);
	updater.UpdateIndividual(opposite_fluxes, tess, context, cells,
		opposite_extensives, 0, std::vector<Vector3D>(),
		std::vector<Vector3D>(),
		std::vector<std::pair<ComputationalCell3D, ComputationalCell3D> >());
	require(close(opposite_extensives[target_neighbors.first].mass, 0.75, 1e-12) &&
		close(opposite_extensives[target_neighbors.second].mass, 1.25, 1e-12) &&
		close(opposite_extensives[target_neighbors.first].energy, 1.125, 1e-12) &&
		close(opposite_extensives[target_neighbors.second].energy, 0.875, 1e-12),
		"material update changed an opposite-component transfer");
	double opposite_mass = 0;
	double opposite_energy = 0;
	for(const Conserved3D& extensive : opposite_extensives)
	{
		require(extensive.mass > 0 && extensive.energy > 0 &&
			extensive.internal_energy > 0,
			"opposite-component material transport left an invalid extent");
		opposite_mass += extensive.mass;
		opposite_energy += extensive.energy;
	}
	require(close(opposite_mass, static_cast<double>(points.size()), 1e-12) &&
		close(opposite_energy, static_cast<double>(points.size()), 1e-12),
		"opposite-component material transport is not conservative");

	// Keep the positive tracked internal energy when total-energy closure would
	// produce a negative value through kinetic-energy cancellation.
	std::vector<ComputationalCell3D> dual_cells = cells;
	dual_cells[target_neighbors.first].velocity = Vector3D(0.48, 0, 0);
	std::vector<Conserved3D> dual_extensives(points.size());
	for(Conserved3D& extensive : dual_extensives)
	{
		extensive.mass = 1;
		extensive.energy = 1;
		extensive.internal_energy = 1;
		extensive.Erad = 1;
		extensive.Eg[0] = 1;
	}
	dual_extensives[target_neighbors.first].momentum = Vector3D(0.48, 0, 0);
	dual_extensives[target_neighbors.first].energy = 1.1152;
	std::vector<Conserved3D> dual_fluxes(tess.GetTotalFacesNumber());
	dual_fluxes[target_face].momentum =
		Vector3D(-0.2 / tess.GetArea(target_face), 0, 0);
	dual_fluxes[target_face].energy = 0.894 / tess.GetArea(target_face);
	updater.UpdateIndividual(dual_fluxes, tess, context, dual_cells,
		dual_extensives, 0, std::vector<Vector3D>(), std::vector<Vector3D>(),
		std::vector<std::pair<ComputationalCell3D, ComputationalCell3D> >());
	require(close(dual_extensives[target_neighbors.first].internal_energy,
		0.01, 1e-12),
		"dual-energy synchronization replaced a positive tracked extent with "
		"a non-positive total-energy value");
}

void testIndividualNegativeMassDiagnostic()
{
	class TargetCondition final : public ConditionExtensiveUpdater3D::Condition3D
	{
	public:
		explicit TargetCondition(std::size_t target) : target_(target) {}
		bool operator()(std::size_t index, const Tessellation3D&,
			const std::vector<ComputationalCell3D>&, double) const override
		{
			return index == target_;
		}
	private:
		std::size_t target_;
	};
	class NegativeMassAction final : public ConditionExtensiveUpdater3D::Action3D
	{
	public:
		void operator()(const std::vector<Conserved3D>&,
			const Tessellation3D&, double,
			const std::vector<ComputationalCell3D>&,
			std::vector<Conserved3D>& extensives,
			std::size_t index, double) const override
		{
			extensives[index].mass = -1;
		}
	};

	std::vector<Vector3D> points{
		Vector3D(0.2, 0.2, 0.2), Vector3D(0.8, 0.2, 0.2),
		Vector3D(0.2, 0.8, 0.2), Vector3D(0.8, 0.8, 0.2),
		Vector3D(0.2, 0.2, 0.8), Vector3D(0.8, 0.2, 0.8),
		Vector3D(0.2, 0.8, 0.8), Vector3D(0.8, 0.8, 0.8)};
	Voronoi3D tess(Vector3D(0, 0, 0), Vector3D(1, 1, 1));
	tess.Build(points);

	std::vector<ComputationalCell3D> cells(points.size());
	std::vector<Conserved3D> extensives(points.size());
	for(std::size_t index = 0; index < points.size(); ++index)
	{
		cells[index].ID = 9000 + index;
		cells[index].density = 1 / tess.GetVolume(index);
		cells[index].pressure = 1;
		cells[index].internal_energy = 1;
		extensives[index].mass = 1;
		extensives[index].energy = 1;
		extensives[index].internal_energy = 1;
		extensives[index].Erad = ENERGY_GROUPS_NUM;
		for(std::size_t group = 0; group < ENERGY_GROUPS_NUM; ++group)
			extensives[index].Eg[group] = 1;
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
		"negative-mass diagnostic test found no internal face");

	IndividualStepContext context;
	context.previous_event_tick = 40;
	context.event_tick = 48;
	context.time_quantum = 0.03125;
	context.previous_event_time = 1.25;
	context.event_time = 1.5;
	context.active_indices.push_back(target_neighbors.first);
	context.active_mask.assign(points.size(), 0);
	context.active_mask[target_neighbors.first] = 1;
	context.cell_time_steps.assign(points.size(), 0.25);
	context.cell_time_bins.assign(points.size(), 7);
	context.primitive_ticks.assign(points.size(), 40);

	std::vector<Conserved3D> fluxes(tess.GetTotalFacesNumber());
	fluxes[target_face].mass = 0.5 / tess.GetArea(target_face) /
		context.cellTimeStep(target_neighbors.first);
	fluxes[target_face].energy = fluxes[target_face].mass;
	int diagnostic_rank = 0;
#ifdef RICH_MPI
	MPI_Comm_rank(MPI_COMM_WORLD, &diagnostic_rank);
#endif

	IndividualStepContext applied_context = context;
	applied_context.previous_event_tick = 48;
	applied_context.event_tick = 56;
	applied_context.previous_event_time = 1.5;
	applied_context.event_time = 1.75;
	applied_context.primitive_ticks.assign(points.size(), 48);
	std::vector<Conserved3D> applied_fluxes = fluxes;
	applied_fluxes[target_face].mass = 2 / tess.GetArea(target_face) /
		applied_context.cellTimeStep(target_neighbors.first);
	applied_fluxes[target_face].energy = applied_fluxes[target_face].mass;
	std::vector<Conserved3D> applied_extensives = extensives;
	std::vector<Vector3D> diagnostic_face_velocities(tess.GetTotalFacesNumber(), Vector3D(0.125, 0.25, 0.5));
	std::vector<Vector3D> diagnostic_point_velocities(points.size(), Vector3D(0.25, 0, 0));
	std::vector<std::pair<ComputationalCell3D, ComputationalCell3D> > diagnostic_face_states(
		tess.GetTotalFacesNumber(), std::make_pair(cells[target_neighbors.first], cells[target_neighbors.second]));
	DefaultExtensiveUpdater applied_updater;
	bool applied_threw = false;
	std::string applied_error;
	const std::string applied_notice = captureStandardError([&]()
	{
		try
		{
			applied_updater.UpdateIndividual(applied_fluxes, tess,
				applied_context, cells, applied_extensives,
				applied_context.event_time, diagnostic_face_velocities,
				diagnostic_point_velocities, diagnostic_face_states);
		}
		catch(std::runtime_error const& error)
		{
			applied_threw = true;
			applied_error = error.what();
		}
	});
	if(!applied_threw ||
		applied_error.find("produced an invalid state") == std::string::npos)
	{
		std::ostringstream diagnostic;
		diagnostic << "applied individual correction did not reject invalid mass"
			<< "; threw=" << (applied_threw ? 1 : 0)
			<< " error=" << applied_error
			<< " target_face=" << target_face
			<< " first_index=" << target_neighbors.first
			<< " first_mass="
			<< applied_extensives[target_neighbors.first].mass
			<< " second_index=" << target_neighbors.second
			<< " second_mass="
			<< applied_extensives[target_neighbors.second].mass
			<< " captured_stderr=" << applied_notice;
		require(false, diagnostic.str());
	}
	const DiagnosticFile applied_diagnostic =
		readDiagnosticFile(applied_notice);
	const std::string& applied_output = applied_diagnostic.contents;
	const std::string applied_record_id = "record_id=" +
		std::to_string(diagnostic_rank) + ":56:" +
		std::to_string(cells[target_neighbors.first].ID);
	require(std::count(applied_notice.begin(), applied_notice.end(), '\n') == 1 &&
		applied_notice.find("INDIVIDUAL_HYDRO_DIAGNOSTIC_FILE") == 0 &&
		applied_notice.find(" status=written") != std::string::npos,
		"applied-correction diagnostic notice is not one complete line");
	require(applied_output.find("INDIVIDUAL_HYDRO_INVALID_MASS_APPLIED") !=
		std::string::npos &&
		applied_output.find("INDIVIDUAL_HYDRO_INVALID_MASS_DELTA") !=
		std::string::npos &&
		applied_output.find("INDIVIDUAL_HYDRO_INVALID_MASS_APPLIED_SUMMARY") !=
		std::string::npos &&
		applied_output.find(applied_record_id) != std::string::npos,
		"applied-correction diagnostic lost its structured records");
	require(applied_output.find("reconstructed_pre_mass=1") !=
		std::string::npos &&
		applied_output.find(" post_mass=-1") != std::string::npos &&
		applied_output.find(" applied_mass_sum=-2") != std::string::npos &&
		applied_output.find(" origin=local") != std::string::npos &&
		applied_output.find(" delta_mass=-2") != std::string::npos &&
		applied_output.find(" flux_available=1") != std::string::npos &&
		applied_output.find(" face=" + std::to_string(target_face)) !=
			std::string::npos &&
		applied_output.find(" flux=(mass=") != std::string::npos,
		"applied-correction diagnostic lost exact delta or source flux data");
	require(applied_output.find(" cell_bin=7 ") != std::string::npos &&
		applied_output.find(" face_dt=0.25 ") != std::string::npos &&
		applied_output.find(" face_velocity_available=1") != std::string::npos &&
		applied_output.find(" face_velocity=(0.125, 0.25, 0.5)") != std::string::npos &&
		applied_output.find(" reconstructed_states_available=1") != std::string::npos &&
		applied_output.find(" left_density=") != std::string::npos &&
		applied_output.find(" right_velocity=") != std::string::npos,
		"applied-correction diagnostic lost numeric bin or face kinematics/state");

	TargetCondition target_condition(target_neighbors.first);
	NegativeMassAction negative_mass_action;
	std::vector<std::pair<const ConditionExtensiveUpdater3D::Condition3D*,
		const ConditionExtensiveUpdater3D::Action3D*> > sequence;
	sequence.emplace_back(&target_condition, &negative_mass_action);
	ConditionExtensiveUpdater3D updater(sequence);
	bool threw = false;
	bool eos_recovery_reached = false;
	std::string error_message;
	const std::string notice = captureStandardError([&]()
	{
		try
		{
			updater.UpdateIndividual(fluxes, tess, context, cells, extensives,
				context.event_time, std::vector<Vector3D>(),
				std::vector<Vector3D>(),
				std::vector<std::pair<ComputationalCell3D,
					ComputationalCell3D> >());
			eos_recovery_reached = true;
		}
		catch(UniversalError const& error)
		{
			threw = true;
			error_message = error.getErrorMessage();
		}
	});

	require(threw, "individual negative mass did not fail before EOS recovery");
	require(!eos_recovery_reached,
		"individual negative-mass failure reached the EOS recovery boundary");
	require(error_message.find("non-positive or non-finite mass") !=
		std::string::npos,
		"individual negative-mass error lost its failure reason");
	const DiagnosticFile diagnostic = readDiagnosticFile(notice);
	const std::string& output = diagnostic.contents;
	require(std::count(notice.begin(), notice.end(), '\n') == 1 &&
		notice.find("INDIVIDUAL_HYDRO_DIAGNOSTIC_FILE") == 0 &&
		notice.find(" status=written") != std::string::npos,
		"individual negative-mass diagnostic notice is not one complete line");
	require(output.find("INDIVIDUAL_HYDRO_INVALID_MASS") != std::string::npos,
		"individual negative-mass diagnostic header is missing");
	const std::string record_id = "record_id=" +
		std::to_string(diagnostic_rank) + ":48:" +
		std::to_string(cells[target_neighbors.first].ID);
	require(output.find(record_id) != std::string::npos,
		"individual negative-mass diagnostic lost its record ID");
	std::istringstream diagnostic_lines(output);
	std::string diagnostic_line;
	std::size_t diagnostic_line_count = 0;
	while(std::getline(diagnostic_lines, diagnostic_line))
	{
		if(diagnostic_line.empty())
			continue;
		require(diagnostic_line.find("INDIVIDUAL_HYDRO_") == 0,
			"individual negative-mass diagnostic emitted a partial line");
		require(diagnostic_line.find(record_id) != std::string::npos,
			"individual negative-mass diagnostic line lost its record ID");
		++diagnostic_line_count;
	}
	require(diagnostic_line_count >= 3,
		"individual negative-mass diagnostic lost structured records");
	require(output.find("event_tick=48") != std::string::npos &&
		output.find("cell_dt=0.25") != std::string::npos &&
		output.find("orientation=-1") != std::string::npos &&
		output.find("face_dt=0.25") != std::string::npos,
		"individual negative-mass diagnostic lost timestep data");
	require(output.find("face=" + std::to_string(target_face)) !=
		std::string::npos &&
		output.find("flux=") != std::string::npos &&
		output.find("applied_change=(mass=-0.5") != std::string::npos &&
		output.find("mass_change_from_primitive=-2") != std::string::npos &&
		output.find("incident_flux_change=(mass=-0.5") != std::string::npos,
		"individual negative-mass diagnostic lost face flux data");

	std::vector<ComputationalCell3D> canonical_cells = cells;
	canonical_cells.push_back(cells.front());
	canonical_cells.back().ID = 99999;
	std::vector<Conserved3D> canonical_extensives(points.size() + 1);
	for(Conserved3D& extensive : canonical_extensives)
	{
		extensive.mass = 1;
		extensive.energy = 1;
		extensive.internal_energy = 1;
		extensive.Erad = ENERGY_GROUPS_NUM;
		for(std::size_t group = 0; group < ENERGY_GROUPS_NUM; ++group)
			extensive.Eg[group] = 1;
	}
	canonical_extensives.back().mass = -1;
	std::vector<Conserved3D> local_extensives(
		canonical_extensives.begin(),
		canonical_extensives.begin() + points.size());
	std::vector<std::pair<const ConditionExtensiveUpdater3D::Condition3D*,
		const ConditionExtensiveUpdater3D::Action3D*> > empty_sequence;
	ConditionExtensiveUpdater3D canonical_updater(empty_sequence);
	bool canonical_threw = false;
	const std::string canonical_notice = captureStandardError([&]()
	{
		try
		{
			canonical_updater.UpdateIndividual(
				std::vector<Conserved3D>(tess.GetTotalFacesNumber()), tess,
				context, cells, local_extensives, context.event_time,
				std::vector<Vector3D>(), std::vector<Vector3D>(),
				std::vector<std::pair<ComputationalCell3D,
					ComputationalCell3D> >(),
				&canonical_cells, &canonical_extensives);
		}
		catch(std::runtime_error const& error)
		{
			canonical_threw = std::string(error.what()).find(
				"invalid input before_flux_application") != std::string::npos;
		}
	});
	require(canonical_threw,
		"individual update accepted invalid canonical passive mass");
	const DiagnosticFile canonical_diagnostic =
		readDiagnosticFile(canonical_notice);
	const std::string& canonical_output = canonical_diagnostic.contents;
	require(std::count(canonical_notice.begin(), canonical_notice.end(), '\n') == 1 &&
		canonical_notice.find("INDIVIDUAL_HYDRO_DIAGNOSTIC_FILE") == 0 &&
		canonical_notice.find(" status=written") != std::string::npos &&
		canonical_diagnostic.path != diagnostic.path,
		"canonical negative-mass diagnostic notice is not a distinct complete line");
	require(canonical_output.find(
		"INDIVIDUAL_HYDRO_INVALID_INPUT") != std::string::npos &&
		canonical_output.find("record_id=" +
			std::to_string(diagnostic_rank) + ":48:99999") !=
			std::string::npos &&
		canonical_output.find("phase=before_flux_application") != std::string::npos &&
		canonical_output.find("component_mask=16") != std::string::npos &&
		canonical_output.find("state=(mass=-1") != std::string::npos,
		"canonical negative-mass preflight lost stable-ID input data");
}

void testHydroMaterialStateValidation()
{
	int rank = 0;
#ifdef RICH_MPI
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif
	std::vector<Vector3D> points{
		Vector3D(0.2, 0.2, 0.2), Vector3D(0.8, 0.2, 0.2),
		Vector3D(0.2, 0.8, 0.2), Vector3D(0.8, 0.8, 0.2),
		Vector3D(0.2, 0.2, 0.8), Vector3D(0.8, 0.2, 0.8),
		Vector3D(0.2, 0.8, 0.8), Vector3D(0.8, 0.8, 0.8)};
	Voronoi3D tess(Vector3D(0, 0, 0), Vector3D(1, 1, 1));
	tess.Build(points);
	std::vector<ComputationalCell3D> cells(points.size());
	for(std::size_t cell = 0; cell < cells.size(); ++cell)
	{
		cells[cell].ID = 990000 + static_cast<std::size_t>(rank) * 10000 + cell;
		cells[cell].density = 1 / tess.GetVolume(cell);
		cells[cell].pressure = cells[cell].internal_energy = 1;
	}
	std::size_t target_face = tess.GetTotalFacesNumber();
	std::pair<std::size_t, std::size_t> endpoints;
	for(std::size_t face = 0; face < tess.GetTotalFacesNumber(); ++face)
	{
		const auto neighbors = tess.GetFaceNeighbors(face);
		if(neighbors.first < points.size() && neighbors.second < points.size())
		{
			target_face = face;
			endpoints = neighbors;
			break;
		}
	}
	requireCollectively(target_face < tess.GetTotalFacesNumber(),
		"material-state test found no internal face");
	IndividualStepContext context;
	context.active_indices.push_back(endpoints.first);
	context.active_mask.assign(points.size(), 0);
	context.active_mask[endpoints.first] = 1;
	context.cell_time_steps.assign(points.size(), 0.25);
	context.time_quantum = 0.25;
	auto set_event_tick = [&](std::uint64_t tick)
	{
		context.event_tick = tick;
		context.previous_event_tick = tick - 1;
		context.event_time = static_cast<double>(tick) * context.time_quantum;
		context.previous_event_time = static_cast<double>(tick - 1) * context.time_quantum;
		context.primitive_ticks.assign(points.size(), tick - 1);
	};
	set_event_tick(1999);
	Conserved3D initial;
	initial.mass = initial.energy = initial.internal_energy = 1;
	std::vector<Conserved3D> extensives(points.size(), initial);
	std::vector<Conserved3D> canonical_extensives = extensives;
	const std::vector<ComputationalCell3D> canonical_cells = cells;
	std::vector<Conserved3D> fluxes(tess.GetTotalFacesNumber());
	DefaultExtensiveUpdater updater;
	std::string failure;
	auto update = [&]()
	{
		failure.clear();
		return captureStandardError([&]()
		{
			try
			{
				updater.UpdateIndividual(fluxes, tess, context, cells, extensives, 0,
					std::vector<Vector3D>(), std::vector<Vector3D>(),
					std::vector<std::pair<ComputationalCell3D, ComputationalCell3D> >(),
					&canonical_cells, &canonical_extensives);
			}
			catch(std::runtime_error const& error)
			{
				failure = error.what();
			}
		});
	};
	auto canonical_unchanged = [&]()
	{
		for(const Conserved3D& canonical : canonical_extensives)
			if(canonical.mass != 1 || canonical.energy != 1 ||
				canonical.internal_energy != 1 || canonical.momentum.x != 0)
				return false;
		return true;
	};
	auto has_record = [&](const std::string& notice, const std::string& expected)
	{
		try
		{
			if(rank != 0)
				return notice.empty();
			const DiagnosticFile diagnostic = readDiagnosticFile(notice);
			return diagnostic.contents.find(expected) != std::string::npos;
		}
		catch(...)
		{
			return false;
		}
	};
	fluxes[target_face].mass = 0.25 /
		(tess.GetArea(target_face) * context.cellTimeStep(endpoints.first));
	std::string notice = update();
	bool valid = failure.empty() && notice.empty() &&
		close(extensives[endpoints.first].mass, 0.75, 1e-12) &&
		close(extensives[endpoints.second].mass, 1.25, 1e-12);
	for(std::size_t cell = 0; cell < extensives.size(); ++cell)
	{
		valid = valid && extensives[cell].Erad == 0 &&
			canonical_extensives[cell].mass == extensives[cell].mass;
		for(double group_energy : extensives[cell].Eg)
			valid = valid && group_energy == 0;
	}
	requireCollectively(valid, "zero-radiation hydro changed the prescribed material transfer");

	enum class InvalidInput { zero_mass, nan_mass, infinite_energy, nan_internal, infinite_momentum };
	for(InvalidInput input : {InvalidInput::zero_mass, InvalidInput::nan_mass,
		InvalidInput::infinite_energy, InvalidInput::nan_internal, InvalidInput::infinite_momentum})
	{
		set_event_tick(2000 + static_cast<std::uint64_t>(input));
		extensives.assign(points.size(), initial);
		canonical_extensives = extensives;
		fluxes.assign(tess.GetTotalFacesNumber(), Conserved3D());
		std::uint64_t expected_mask = 0;
		Conserved3D invalid = initial;
		switch(input)
		{
		case InvalidInput::zero_mass:
			invalid.mass = 0;
			expected_mask = 16;
			break;
		case InvalidInput::nan_mass:
			invalid.mass = std::numeric_limits<double>::quiet_NaN();
			expected_mask = 16;
			break;
		case InvalidInput::infinite_energy:
			invalid.energy = std::numeric_limits<double>::infinity();
			expected_mask = 32;
			break;
		case InvalidInput::nan_internal:
			invalid.internal_energy = std::numeric_limits<double>::quiet_NaN();
			expected_mask = 64;
			break;
		case InvalidInput::infinite_momentum:
			invalid.momentum.x = std::numeric_limits<double>::infinity();
			expected_mask = 128;
			break;
		}
		if(rank == 0)
			extensives[endpoints.first] = invalid;
		std::feclearexcept(FE_ALL_EXCEPT);
		notice = update();
		const int arithmetic_flags = std::fetestexcept(FE_DIVBYZERO | FE_INVALID);
		valid = failure.find("invalid input before_flux_application") != std::string::npos &&
			failure.find("component mask " + std::to_string(expected_mask)) != std::string::npos &&
			canonical_unchanged() && (input != InvalidInput::zero_mass || arithmetic_flags == 0);
		const bool recorded = has_record(notice, "phase=before_flux_application component_mask=" +
			std::to_string(expected_mask));
		requireCollectively(valid && recorded,
			"invalid material input escaped the collective pre-arithmetic guard");
	}

	set_event_tick(2010);
	extensives.assign(points.size(), initial);
	canonical_extensives = extensives;
	if(rank == 0)
		fluxes[target_face].energy = std::numeric_limits<double>::infinity();
	notice = update();
	valid = failure.find("before_dual_energy_synchronization") != std::string::npos &&
		canonical_unchanged();
	if(rank == 0)
		valid = valid && (IndividualHydroInvalidComponentMask(extensives[endpoints.first]) & 32) != 0;
	const bool recorded = has_record(notice, "phase=before_dual_energy_synchronization");
	requireCollectively(valid && recorded,
		"nonfinite material correction escaped the collective pre-commit guard");

	set_event_tick(2011);
	extensives.assign(points.size(), initial);
	for(Conserved3D& extensive : extensives)
		extensive.internal_energy = -1;
	canonical_extensives = extensives;
	fluxes.assign(tess.GetTotalFacesNumber(), Conserved3D());
	notice = update();
	valid = failure.empty() && notice.empty();
	for(const Conserved3D& extensive : extensives)
		valid = valid && extensive.internal_energy == -1;
	requireCollectively(valid, "finite tracked internal energy was rejected before entropy recovery");
}

#ifdef RICH_MPI
void testConditionExtensiveCollectiveFailureMPI()
{
	enum class FailureSource
	{
		valid,
		negative_mass,
		nonfinite_energy,
		standard_exception,
		universal_exception,
		canonical_mass,
		canonical_energy
	};
	class TargetCondition final : public ConditionExtensiveUpdater3D::Condition3D
	{
	public:
		explicit TargetCondition(bool enabled) : enabled_(enabled) {}
		bool operator()(std::size_t index, const Tessellation3D&,
			const std::vector<ComputationalCell3D>&, double) const override
		{
			return enabled_ && index == 0;
		}
	private:
		bool enabled_;
	};
	class TestAction final : public ConditionExtensiveUpdater3D::Action3D
	{
	public:
		TestAction(FailureSource source, std::vector<Conserved3D>& canonical) :
			source_(source), canonical_(canonical) {}
		void operator()(const std::vector<Conserved3D>&, const Tessellation3D&,
			double, const std::vector<ComputationalCell3D>&,
			std::vector<Conserved3D>& extensives, std::size_t index, double) const override
		{
			switch(source_)
			{
			case FailureSource::valid:
				extensives[index].mass = 1.25;
				break;
			case FailureSource::negative_mass:
				extensives[index].mass = -1;
				break;
			case FailureSource::nonfinite_energy:
				extensives[index].energy = std::numeric_limits<double>::quiet_NaN();
				break;
			case FailureSource::standard_exception:
				throw std::runtime_error("test standard condition action");
			case FailureSource::universal_exception:
			{
				extensives[index].mass = -1;
				UniversalError error("test universal condition action");
				error.addEntry("test failure value", 7);
				throw error;
			}
			case FailureSource::canonical_mass:
				canonical_.back().mass = -1;
				break;
			case FailureSource::canonical_energy:
				canonical_.back().energy = std::numeric_limits<double>::quiet_NaN();
				break;
			}
		}
	private:
		FailureSource source_;
		std::vector<Conserved3D>& canonical_;
	};

	int rank = 0;
	int rank_count = 1;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &rank_count);
	require(rank_count >= 2, "collective condition test requires at least two ranks");
	const int failing_rank = rank_count - 1;
	std::vector<Vector3D> points{
		Vector3D(0.2, 0.2, 0.2), Vector3D(0.8, 0.2, 0.2),
		Vector3D(0.2, 0.8, 0.2), Vector3D(0.8, 0.8, 0.2),
		Vector3D(0.2, 0.2, 0.8), Vector3D(0.8, 0.2, 0.8),
		Vector3D(0.2, 0.8, 0.8), Vector3D(0.8, 0.8, 0.8)};
	Voronoi3D tess(Vector3D(0, 0, 0), Vector3D(1, 1, 1));
	tess.Build(points);
	std::vector<ComputationalCell3D> cells(points.size());
	Conserved3D initial;
	initial.mass = initial.energy = initial.internal_energy = 1;
	initial.Erad = initial.Eg[0] = 1;
	for(std::size_t cell = 0; cell < cells.size(); ++cell)
	{
		cells[cell].ID = 970000 + static_cast<std::size_t>(rank) * 10000 + cell;
		cells[cell].density = 1 / tess.GetVolume(cell);
		cells[cell].pressure = cells[cell].internal_energy = 1;
	}
	std::vector<ComputationalCell3D> canonical_cells = cells;
	canonical_cells.push_back(cells.front());
	canonical_cells.back().ID += 5000;
	IndividualStepContext context;
	context.active_indices.push_back(0);
	context.active_mask.assign(points.size(), 0);
	context.active_mask[0] = 1;
	context.cell_time_steps.assign(points.size(), 0.25);
	context.time_quantum = 0.25;
	const std::vector<Conserved3D> fluxes(tess.GetTotalFacesNumber());
	TargetCondition target(rank == failing_rank);
	for(FailureSource source : {FailureSource::valid, FailureSource::negative_mass,
		FailureSource::nonfinite_energy, FailureSource::standard_exception,
		FailureSource::universal_exception, FailureSource::canonical_mass,
		FailureSource::canonical_energy})
	{
		context.event_tick = 100 + static_cast<std::uint64_t>(source);
		context.previous_event_tick = context.event_tick - 1;
		context.primitive_ticks.assign(points.size(), context.previous_event_tick);
		context.event_time = static_cast<double>(context.event_tick) * context.time_quantum;
		context.previous_event_time =
			static_cast<double>(context.previous_event_tick) * context.time_quantum;
		std::vector<Conserved3D> extensives(points.size(), initial);
		std::vector<Conserved3D> canonical_extensives(canonical_cells.size(), initial);
		TestAction action(source, canonical_extensives);
		ConditionExtensiveUpdater3D updater({{&target, &action}});
		bool rejected = false;
		bool unexpected_exception = false;
		bool callback_reason_kept = true;
		const std::string notice = captureStandardError([&]()
		{
			try
			{
				updater.UpdateIndividual(fluxes, tess, context, cells, extensives, 0,
					std::vector<Vector3D>(), std::vector<Vector3D>(),
					std::vector<std::pair<ComputationalCell3D, ComputationalCell3D> >(),
					&canonical_cells, &canonical_extensives);
			}
			catch(UniversalError const& error)
			{
				rejected = true;
				if(rank == failing_rank && source == FailureSource::standard_exception)
					callback_reason_kept = error.getErrorMessage().find(
						"test standard condition action") != std::string::npos;
				if(rank == failing_rank && source == FailureSource::universal_exception)
				{
					std::ostringstream details;
					reportError(error, details);
					callback_reason_kept = error.getErrorMessage().find(
						"test universal condition action") != std::string::npos &&
						details.str().find("test failure value") != std::string::npos;
				}
			}
			catch(...)
			{
				unexpected_exception = true;
			}
		});
		const bool canonical_failure = source == FailureSource::canonical_mass ||
			source == FailureSource::canonical_energy;
		const bool state_failure = canonical_failure ||
			source == FailureSource::negative_mass || source == FailureSource::nonfinite_energy ||
			source == FailureSource::universal_exception;
		int valid = !unexpected_exception && callback_reason_kept &&
			rejected == (source != FailureSource::valid);
		try
		{
			if(rank == failing_rank && state_failure)
			{
				const DiagnosticFile diagnostic = readDiagnosticFile(notice);
				const std::size_t cell_id = canonical_failure ?
					canonical_cells.back().ID : cells.front().ID;
				valid = valid && diagnostic.contents.find("record_id=" + std::to_string(rank) +
					":" + std::to_string(context.event_tick) + ":" + std::to_string(cell_id)) !=
					std::string::npos;
				if(source == FailureSource::nonfinite_energy || source == FailureSource::canonical_energy)
					valid = valid && diagnostic.contents.find("component_mask=32") != std::string::npos;
				else
					valid = valid && diagnostic.contents.find("mass=-1") != std::string::npos;
			}
			else
				valid = valid && notice.empty();
		}
		catch(...)
		{
			valid = 0;
		}
		// Condition actions stay local until the caller's existing scatter; a
		// rejected action must not overwrite the valid canonical flux result.
		for(std::size_t cell = 0; cell < cells.size(); ++cell)
			valid = valid && canonical_extensives[cell].mass == 1 &&
				canonical_extensives[cell].energy == 1;
		if(source == FailureSource::valid)
			valid = valid && extensives.front().mass == (rank == failing_rank ? 1.25 : 1);
		MPI_Allreduce(MPI_IN_PLACE, &valid, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
		require(valid == 1, "condition action did not preserve collective rejection and diagnostics");
	}
}
#endif

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
// A neighbour-bin request for a cell owned by another rank must be applied
// exactly as a local one.  The requester used to send nextAlignedTick(event,
// bin), which ignores the owner's begin tick: an overdue or unaligned cell kept
// an end beyond its allowance (the job-10196266 overrun).
// Re-expansion of a remote depth promotion on the existing mesh (A1,
// RICH_INDIVIDUAL_CLOSURE_REEXPAND), through the real closure loop of
// HDSim3D::timeAdvanceIndividual via IndividualClosureTestProbe.  Only A is
// active; B is A's remote neighbour and every other cell is prebuilt at
// depth two, so A's closure asks B's owner (which has no active cell) to
// promote B to depth one while no rank adds a target cell.  Off: one
// depth-only rebuild.  On: B is expanded at depth one on the first mesh.  Both
// must close with the same depths and target set, and the retained mesh must
// match a full build within the verifier tolerances.
void testClosureReexpandPromotionMPI()
{
    int rank = 0;
    int rank_count = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &rank_count);
    require(rank_count >= 2, "re-expansion test requires at least two ranks");
    std::vector<Vector3D> points;
    for(double const x : {0.25, 0.75})
        for(double const y : {0.25, 0.75})
            for(double const z : {0.25, 0.75})
                points.emplace_back((rank + x) / rank_count, y, z);
    Voronoi3D tess(Vector3D(0, 0, 0), Vector3D(1, 1, 1));
    tess.BuildParallel(points);
    std::vector<Vector3D> const canonical_points = tess.getAllPoints();
    std::size_t const n = canonical_points.size();
    std::vector<std::size_t> all(n);
    std::iota(all.begin(), all.end(), 0);
    std::vector<double> const weights(n, 1.0);
    auto full_mesh = [&]() { tess.BuildPartiallyParallel(canonical_points, weights, all, true, true); };
    full_mesh();
    IdealGas eos(5.0 / 3.0);
    std::vector<ComputationalCell3D> cells(n);
    for(std::size_t cell = 0; cell < n; ++cell)
    {
        cells[cell].density = 1;
        cells[cell].pressure = 1;
        cells[cell].internal_energy = eos.dp2e(1, 1, cells[cell].tracers, ComputationalCell3D::tracerNames);
        cells[cell].velocity = Vector3D(0, 0, 0);
        cells[cell].ID = 840000 + rank * 10000 + cell;
    }

    // A: the lowest ID with a face to another rank; B: A's lowest-ID remote neighbour.
    std::uint64_t const no_id = std::numeric_limits<std::uint64_t>::max();
    std::vector<ComputationalCell3D> mesh_cells = cells;
    tess.SyncPartialBuildData(mesh_cells, cells);
    std::uint64_t a_id = no_id;
    for(std::size_t face = 0; face < tess.GetTotalFacesNumber(); ++face)
    {
        auto const pair = tess.GetFaceNeighbors(face);
        if(!tess.BoundaryFace(face) && (pair.first < tess.GetPointNo()) != (pair.second < tess.GetPointNo()))
            a_id = std::min(a_id, static_cast<std::uint64_t>(
                mesh_cells[pair.first < tess.GetPointNo() ? pair.first : pair.second].ID));
    }
    MPI_Allreduce(MPI_IN_PLACE, &a_id, 1, MPI_UINT64_T, MPI_MIN, MPI_COMM_WORLD);
    require(a_id != no_id, "re-expansion test found no inter-rank face");
    int const a_rank = static_cast<int>((a_id - 840000) / 10000);
    std::size_t a_local = n;
    std::uint64_t b_id = no_id;
    if(rank == a_rank)
    {
        a_local = static_cast<std::size_t>(a_id - 840000 - static_cast<std::uint64_t>(rank) * 10000);
        std::vector<std::size_t> neighbours;
        tess.GetNeighbors(a_local, neighbours);
        for(std::size_t neighbour : neighbours)
            if(neighbour >= tess.GetPointNo() && neighbour < mesh_cells.size() && !tess.IsPointOutsideBox(neighbour) &&
               mesh_cells[neighbour].ID >= 840000 &&
               static_cast<int>((mesh_cells[neighbour].ID - 840000) / 10000) != rank)
                b_id = std::min(b_id, static_cast<std::uint64_t>(mesh_cells[neighbour].ID));
    }
    MPI_Allreduce(MPI_IN_PLACE, &b_id, 1, MPI_UINT64_T, MPI_MIN, MPI_COMM_WORLD);
    require(b_id != no_id, "re-expansion test found no remote neighbour of A");
    int const b_rank = static_cast<int>((b_id - 840000) / 10000);
    require(b_rank != a_rank, "re-expansion test: B must be remote from A");
    std::size_t const b_local = rank == b_rank ?
        static_cast<std::size_t>(b_id - 840000 - static_cast<std::uint64_t>(rank) * 10000) : n;

    std::vector<Conserved3D> extensives(n);
    RigidWallGenerator3D ghost;
    LinearGauss3D interp(eos, ghost);
    Hllc3D rs;
    IsBoundaryFace3D is_boundary;
    IsBulkFace3D is_bulk;
    RegularFlux3D regular(rs);
    RigidWallFlux3D rigid(rs);
    std::vector<std::pair<ConditionActionFlux1::Condition3D const*, ConditionActionFlux1::Action3D const*> >
        sequence{std::make_pair(&is_boundary, &rigid), std::make_pair(&is_bulk, &regular)};
    ConditionActionFlux1 flux(sequence, interp);
    DefaultCellUpdater cu;
    DefaultExtensiveUpdater eu;
    ZeroForce3D force;
    Eulerian3D pm;
    ManualTimeStep tsf(1e-3);
    ProgressTracker tracker;
    HDSim3D sim(tess, cells, extensives, eos, tracker, pm, tsf, flux, cu, eu, force,
        std::make_pair(ComputationalCell3D::tracerNames, ComputationalCell3D::stickerNames));

    IndividualStepContext context;
    context.previous_event_tick = 0;
    context.event_tick = 1;
    context.time_quantum = 1e-3;
    context.previous_event_time = 0;
    context.event_time = 1e-3;
    context.mesh_build_policy = IndividualMeshBuildPolicy::AutoPartial;
    // No closure-threshold bailout: the whole mesh may be a partial target.
    context.partial_build_fraction = 1.0;
    context.active_mask.assign(n, 0);
    if(rank == a_rank)
    {
        context.active_mask[a_local] = 1;
        context.active_indices.push_back(a_local);
    }
    context.cell_time_steps.assign(n, 1e-3);
    context.cell_time_bins.assign(n, 0);
    context.primitive_ticks.assign(n, 0);

    struct Run
    {
        IndividualClosureTestProbe probe;
        unsigned long long builds = 0;
        unsigned long long full_builds = 0;
    };
    auto run = [&](int reexpand, int verify) -> Run
    {
        full_mesh();
        Run result;
        result.probe.closure_reexpand = reexpand;
        result.probe.reexpand_verify = verify;
        for(std::size_t cell = 0; cell < n; ++cell)
            if(cell != a_local)
                result.probe.depth_two_seed.push_back(cell);
        IndividualClosureTestAccess::attach(sim, &result.probe);
        sim.timeAdvanceIndividual(context);
        IndividualClosureTestAccess::attach(sim, nullptr);
        result.builds = sim.GetLastMeshBuildTiming().builds;
        result.full_builds = sim.GetLastMeshBuildTiming().full_builds;
        return result;
    };
    auto sum = [](unsigned long long value)
    {
        MPI_Allreduce(MPI_IN_PLACE, &value, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
        return value;
    };
    auto expanded_at = [&](Run const& r, std::size_t attempt, std::size_t reexpansions, std::size_t depth)
    {
        if(rank != b_rank)
            return true;
        for(auto const& e : r.probe.expansions)
            if(e[0] == attempt && e[1] == reexpansions && e[2] == b_local && e[3] == depth)
                return true;
        return false;
    };

    Run const off = run(0, 0);
    Run const on = run(1, 0);
    // Geometry of the retained re-expanded mesh against a full build (tess still holds the "on" mesh).
    std::vector<std::size_t> checked;
    for(std::size_t cell = 0; cell < n; ++cell)
        if(cell < on.probe.final_depth.size() && on.probe.final_depth[cell] < 2)
            checked.push_back(cell);
    requireCollectively(sum(checked.size()) >= 2, "re-expansion test checks fewer than two cells");
    requireCollectively(rank != b_rank || std::find(checked.begin(), checked.end(), b_local) != checked.end(),
        "re-expansion test: B is not in the checked depth-one set");
    bool const geometry_agrees = IndividualClosureTestAccess::retainedMatchesFullBuild(sim, checked, 1);
    Run const verified = run(1, 1);

    for(Run const* r : {&off, &on, &verified})
    {
        requireCollectively(r->probe.result == "partial", "re-expansion test: build was not partial (" +
            r->probe.result + " " + r->probe.reason + ")");
        requireCollectively(r->full_builds == 0, "re-expansion test: a full build ran");
        requireCollectively(r->probe.final_target.size() == n, "re-expansion test: target does not cover every cell");
    }
    requireCollectively(rank != b_rank || on.probe.remote_promotions >= 1,
        "re-expansion test: B's owner saw no remote promotion");
    requireCollectively(sum(off.probe.remote_promotions) >= 1, "re-expansion test: no remote promotion (off)");
    // Off: one depth-only rebuild, B expanded at depth one on the second mesh.
    requireCollectively(off.probe.reexpansions == 0 && off.probe.attempts == 2,
        "re-expansion test: off run did not take one depth-only rebuild");
    requireCollectively(sum(off.probe.rebuild_local_additions.empty() ? 1 : off.probe.rebuild_local_additions[0]) == 0,
        "re-expansion test: off rebuild had additions");
    requireCollectively(expanded_at(off, 2, 0, 1), "re-expansion test: off run did not expand B at depth one");
    // On: B re-expanded at depth one on the first mesh, no additions anywhere.
    requireCollectively(on.probe.reexpansions == 1 && on.probe.attempts == 1,
        "re-expansion test: on run did not re-expand once on the first mesh");
    requireCollectively(off.builds == 2 && on.builds == 1, "re-expansion test: unexpected mesh build counts");
    requireCollectively(on.probe.reason == "closed", "re-expansion test: on run reason " + on.probe.reason);
    requireCollectively(!on.probe.promoted_at_continue.empty(), "re-expansion test: no continuation recorded");
    requireCollectively(rank != b_rank || std::find(on.probe.promoted_at_continue[0].begin(),
        on.probe.promoted_at_continue[0].end(), b_local) != on.probe.promoted_at_continue[0].end(),
        "re-expansion test: B not among the promotions at the continuation");
    requireCollectively(on.probe.rebuild_local_additions.empty(), "re-expansion test: on run reached a rebuild");
    requireCollectively(expanded_at(on, 1, 1, 1), "re-expansion test: on run did not expand B at depth one");
    // Same closure either way.
    std::vector<std::size_t> off_target = off.probe.final_target;
    std::vector<std::size_t> on_target = on.probe.final_target;
    std::sort(off_target.begin(), off_target.end());
    std::sort(on_target.begin(), on_target.end());
    requireCollectively(off.probe.final_depth == on.probe.final_depth && off_target == on_target,
        "re-expansion test: on and off closures differ");
    requireCollectively(verified.probe.reexpansions == 1 && verified.probe.reason == "closed after re-expansion check",
        "re-expansion test: verifier run reason " + verified.probe.reason);
    requireCollectively(geometry_agrees, "re-expansion test: retained mesh differs from a full build");
}

void testRemoteNeighborBinClosureMPI()
{
    int rank = 0;
    int rank_count = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &rank_count);
    require(rank_count >= 2, "MPI remote closure test requires at least two ranks");
    std::vector<Vector3D> points;
    for(double const x : {0.25, 0.75})
        for(double const y : {0.25, 0.75})
            for(double const z : {0.25, 0.75})
                points.emplace_back((rank + x) / rank_count, y, z);
    Voronoi3D tess(Vector3D(0, 0, 0), Vector3D(1, 1, 1));
    tess.BuildParallel(points);
    std::vector<Vector3D> const canonical_points = tess.getAllPoints();
    std::vector<std::size_t> all(canonical_points.size());
    std::iota(all.begin(), all.end(), 0);
    tess.BuildPartiallyParallel(canonical_points,
        std::vector<double>(canonical_points.size(), 1.0), all, true, true);
    std::vector<ComputationalCell3D> cells(canonical_points.size());
    for(std::size_t cell = 0; cell < cells.size(); ++cell)
        cells[cell].ID = 830000 + rank * 10000 + cell;

    // The bin-30 source: the lowest ID with a face to another rank.
    std::uint64_t const no_id = std::numeric_limits<std::uint64_t>::max();
    std::vector<ComputationalCell3D> mesh_cells = cells;
    tess.SyncPartialBuildData(mesh_cells, cells);
    std::uint64_t source_id = no_id;
    for(std::size_t face = 0; face < tess.GetTotalFacesNumber(); ++face)
    {
        auto const pair = tess.GetFaceNeighbors(face);
        if(!tess.BoundaryFace(face) &&
           (pair.first < tess.GetPointNo()) != (pair.second < tess.GetPointNo()))
        {
            std::size_t const owned = pair.first < tess.GetPointNo() ? pair.first : pair.second;
            source_id = std::min(source_id, static_cast<std::uint64_t>(mesh_cells[owned].ID));
        }
    }
    MPI_Allreduce(MPI_IN_PLACE, &source_id, 1, MPI_UINT64_T, MPI_MIN, MPI_COMM_WORLD);
    require(source_id != no_id, "MPI remote closure test found no inter-rank face");
    int const source_rank = static_cast<int>((source_id - 830000) / 10000);

    std::uint64_t const B = std::uint64_t(1) << 30;
    IndividualTimeStepOptions options;
    options.time_quantum = 1;
    // Every non-source cell starts at bin 32; its begin tick decides what a
    // bin-31 request must give (event tick 2B, finest occupied bin 30):
    // begin 0 is overdue -> next bin-30 tick 3B; begin B has allowance
    // B + 2^31 = 3B ahead -> 3B; begin 2B is aligned -> 4B.
    // On the full mesh every owner also sees the source as a ghost and lowers
    // its cell itself, so the remote path never decides the result.  The
    // partial mesh holds only the source: the other ranks build nothing and
    // their cells are lowered by remote requests alone.
    for(bool const partial : {false, true})
    for(std::uint64_t const begin : {std::uint64_t(0), B, 2 * B})
    {
        std::vector<std::size_t> target;
        if(!partial)
            target = all;
        else
            for(std::size_t cell = 0; cell < cells.size(); ++cell)
                if(cells[cell].ID == source_id)
                    target.push_back(cell);
        tess.BuildPartiallyParallel(canonical_points,
            std::vector<double>(canonical_points.size(), 1.0), target, true, true);
        std::uint64_t const expected_end = begin == 2 * B ? 4 * B : 3 * B;
        std::vector<CellTimeState> states(cells.size());
        for(std::size_t cell = 0; cell < cells.size(); ++cell)
        {
            CellTimeState& state = states[cell];
            state.cell_id = cells[cell].ID;
            bool const source = cells[cell].ID == source_id;
            state.begin_tick = source ? 2 * B : begin;
            state.last_primitive_tick = state.begin_tick;
            state.end_tick = source ? 3 * B : 4 * B;
            state.time_bin = source ? 30 : 32;
        }
        IndividualTimeStepScheduler scheduler(options);
        scheduler.restore(cells, 0, 1, 2 * B, states);
        scheduler.enforceNeighborBinClosure(tess, cells);
        int counts[3] = {0, 0, 0};
        for(CellTimeState const& state : scheduler.states())
        {
            if(state.cell_id == source_id)
                counts[2] |= state.time_bin != 30 || state.end_tick != 3 * B;
            else if(state.time_bin == 31)
            {
                counts[rank == source_rank ? 0 : 1] += 1;
                counts[2] |= state.end_tick != expected_end;
            }
            else
                counts[2] |= state.time_bin != 32 || state.end_tick != 4 * B;
        }
        MPI_Allreduce(MPI_IN_PLACE, counts, 2, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, counts + 2, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        require(counts[1] > 0, "MPI remote closure test lowered no cell on another rank");
        require(counts[2] == 0, "remote neighbour-bin request ignored the owner's allowance");
    }
}

// Two sources on rank 0 send competing requests (bins 3 and 4) for the same
// cell on another rank.  binnedEndTick is not monotonic in the bin: for
// begin 17, event 32, finest bin 2 it gives 33 for bin 4 (allowance ahead) and
// 36 for bin 3 (overdue, next bin-2 tick), so applying the requests one by one
// left 33 or 36 depending on arrival order.  The owner must apply the minimum.
void testCompetingRemoteBinRequestsMPI()
{
    int rank = 0;
    int rank_count = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &rank_count);
    require(rank_count >= 2, "MPI competing-request test requires at least two ranks");
    std::mt19937_64 generator(4242 + rank);
    std::uniform_real_distribution<double> unit(0.05, 0.95);
    std::vector<Vector3D> points;
    for(int i = 0; i < 40; ++i)
        points.emplace_back((rank + unit(generator)) / rank_count, unit(generator), unit(generator));
    Voronoi3D tess(Vector3D(0, 0, 0), Vector3D(1, 1, 1));
    tess.BuildParallel(points);
    std::vector<Vector3D> const canonical_points = tess.getAllPoints();
    std::vector<std::size_t> all(canonical_points.size());
    std::iota(all.begin(), all.end(), 0);
    tess.BuildPartiallyParallel(canonical_points,
        std::vector<double>(canonical_points.size(), 1.0), all, true, true);
    std::vector<ComputationalCell3D> cells(canonical_points.size());
    for(std::size_t cell = 0; cell < cells.size(); ++cell)
        cells[cell].ID = 850000 + rank * 10000 + cell;
    std::vector<ComputationalCell3D> mesh_cells = cells;
    tess.SyncPartialBuildData(mesh_cells, cells);

    // On rank 0: a ghost owned by another rank with at least two rank-0
    // neighbours; the smallest such ghost ID and its two smallest neighbours.
    std::uint64_t const no_id = std::numeric_limits<std::uint64_t>::max();
    std::uint64_t ids[3] = {no_id, no_id, no_id};
    if(rank == 0)
    {
        std::map<std::uint64_t, std::vector<std::uint64_t> > ghost_neighbors;
        std::vector<std::size_t> neighbors;
        for(std::size_t local = 0; local < tess.GetPointNo(); ++local)
        {
            neighbors.clear();
            tess.GetNeighbors(local, neighbors);
            for(std::size_t const neighbor : neighbors)
                if(neighbor >= tess.GetPointNo() && neighbor < mesh_cells.size() &&
                   !tess.IsPointOutsideBox(neighbor))
                    ghost_neighbors[mesh_cells[neighbor].ID].push_back(mesh_cells[local].ID);
        }
        for(auto& entry : ghost_neighbors)
            if(entry.second.size() >= 2)
            {
                std::sort(entry.second.begin(), entry.second.end());
                ids[0] = entry.first;
                ids[1] = entry.second[0];
                ids[2] = entry.second[1];
                break;
            }
    }
    MPI_Bcast(ids, 3, MPI_UINT64_T, 0, MPI_COMM_WORLD);
    require(ids[0] != no_id, "MPI competing-request test found no ghost with two rank-0 neighbours");
    std::uint64_t const target_id = ids[0];

    IndividualTimeStepOptions options;
    options.time_quantum = 1;
    // Both bin assignments, so either request can arrive first.
    for(int const order : {0, 1})
    {
        std::vector<std::size_t> target;
        for(std::size_t cell = 0; cell < cells.size(); ++cell)
            if(cells[cell].ID == ids[1] || cells[cell].ID == ids[2])
                target.push_back(cell);
        tess.BuildPartiallyParallel(canonical_points,
            std::vector<double>(canonical_points.size(), 1.0), target, true, true);
        std::vector<CellTimeState> states(cells.size());
        for(std::size_t cell = 0; cell < cells.size(); ++cell)
        {
            CellTimeState& state = states[cell];
            std::uint64_t const id = cells[cell].ID;
            state.cell_id = id;
            if(id == ids[1] || id == ids[2])
            {
                // Sources: bins 2 and 3, so the finest occupied bin is 2 and
                // the requests are bins 3 and 4.
                state.time_bin = ((id == ids[1]) == (order == 0)) ? 2 : 3;
                state.begin_tick = 32;
                state.end_tick = 32 + (std::uint64_t(1) << state.time_bin);
            }
            else
            {
                state.time_bin = 6;
                state.begin_tick = 17;
                state.end_tick = 64;
            }
            state.last_primitive_tick = state.begin_tick;
        }
        IndividualTimeStepScheduler scheduler(options);
        scheduler.restore(cells, 0, 1, 32, states);
        std::map<std::size_t, std::uint64_t> restored_end;
        for(CellTimeState const& state : scheduler.states())
            restored_end[state.cell_id] = state.end_tick;
        scheduler.enforceNeighborBinClosure(tess, cells);
        int bad = 0;
        int seen = 0;
        for(CellTimeState const& state : scheduler.states())
        {
            if(state.cell_id == ids[1] || state.cell_id == ids[2])
                continue;
            // What binnedEndTick must give for begin 17 at event 32 with
            // finest bin 2, per bin; bin 6 is untouched.
            std::uint64_t expected = 64;
            if(state.time_bin == 3)
                expected = 36;
            else if(state.time_bin == 4)
                expected = 33;
            else if(state.time_bin == 5)
                expected = 49;
            else if(state.time_bin != 6)
                bad = 1;
            // Ends only move earlier: a bin-spread cap applied at restore can
            // already have shortened the end below the closure's value.
            auto const restored = restored_end.find(state.cell_id);
            if(restored != restored_end.end())
                expected = std::min(expected, restored->second);
            bad = bad || state.end_tick != expected;
            if(state.cell_id == target_id)
            {
                seen = 1;
                bad = bad || state.time_bin != 3;
            }
        }
        MPI_Allreduce(MPI_IN_PLACE, &bad, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, &seen, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        require(seen == 1, "MPI competing-request test lost the target cell");
        require(bad == 0, "competing remote bin requests depended on their arrival order");
    }
}

void testHydroWakeFaceIntervalsMPI()
{
    int rank = 0;
    int rank_count = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &rank_count);
    require(rank_count >= 2, "MPI hydro wake test requires at least two ranks");
    std::vector<Vector3D> points;
    for(double const x : {0.25, 0.75})
        for(double const y : {0.25, 0.75})
            for(double const z : {0.25, 0.75})
                points.emplace_back((rank + x) / rank_count, y, z);
    Voronoi3D tess(Vector3D(0, 0, 0), Vector3D(1, 1, 1));
    tess.BuildParallel(points);
    std::vector<Vector3D> const canonical_points = tess.getAllPoints();
    std::vector<ComputationalCell3D> canonical_cells(canonical_points.size());
    for(std::size_t cell = 0; cell < canonical_cells.size(); ++cell)
    {
        canonical_cells[cell].ID = 810000 + rank * 10000 + cell;
        canonical_cells[cell].density = 1;
        canonical_cells[cell].pressure = 1;
        canonical_cells[cell].internal_energy = 1;
    }
    ActiveMeshView const full_view(tess, canonical_cells.size());
    std::vector<ComputationalCell3D> cells = full_view.gatherOwned(canonical_cells);
    tess.SyncPartialBuildData(cells, canonical_cells);
    std::uint64_t const no_id = std::numeric_limits<std::uint64_t>::max();
    std::uint64_t first_id = no_id;
    for(std::size_t face = 0; face < tess.GetTotalFacesNumber(); ++face)
    {
        auto const pair = tess.GetFaceNeighbors(face);
        if(!tess.BoundaryFace(face) &&
           (pair.first < tess.GetPointNo()) != (pair.second < tess.GetPointNo()))
            first_id = std::min(first_id, static_cast<std::uint64_t>(
                std::min(cells[pair.first].ID, cells[pair.second].ID)));
    }
    MPI_Allreduce(MPI_IN_PLACE, &first_id, 1, MPI_UINT64_T, MPI_MIN, MPI_COMM_WORLD);
    std::uint64_t second_id = no_id;
    for(std::size_t face = 0; face < tess.GetTotalFacesNumber(); ++face)
    {
        auto const pair = tess.GetFaceNeighbors(face);
        if(!tess.BoundaryFace(face) &&
           (pair.first < tess.GetPointNo()) != (pair.second < tess.GetPointNo()) &&
           std::min(cells[pair.first].ID, cells[pair.second].ID) == first_id)
            second_id = std::min(second_id, static_cast<std::uint64_t>(
                std::max(cells[pair.first].ID, cells[pair.second].ID)));
    }
    MPI_Allreduce(MPI_IN_PLACE, &second_id, 1, MPI_UINT64_T, MPI_MIN, MPI_COMM_WORLD);
    require(first_id != no_id && second_id != no_id,
            "MPI hydro wake test found no inter-rank face");

    for(bool const partial : {false, true})
    {
        std::vector<Conserved3D> canonical_extensives(canonical_cells.size());
        for(Conserved3D& extensive : canonical_extensives)
        {
            extensive.mass = 1;
            extensive.energy = 1;
            extensive.internal_energy = 1;
        }
        for(std::uint64_t const tick : {4u, 5u, 8u})
        {
            IndividualStepContext context;
            context.event_tick = tick;
            context.event_time = static_cast<double>(tick);
            context.time_quantum = 1;
            context.active_mask.assign(canonical_cells.size(), 0);
            context.cell_time_steps.assign(canonical_cells.size(), 8);
            context.primitive_ticks.assign(canonical_cells.size(), 0);
            for(std::size_t cell = 0; cell < canonical_cells.size(); ++cell)
            {
                std::size_t const id = canonical_cells[cell].ID;
                if(id == first_id)
                {
                    context.cell_time_steps[cell] = 4;
                    context.primitive_ticks[cell] = tick == 4 ? 0 : 4;
                    context.active_mask[cell] = tick != 5;
                }
                else if(id == second_id)
                {
                    context.cell_time_steps[cell] = tick == 8 ? 3 : 5;
                    context.primitive_ticks[cell] = tick == 8 ? 5 : 0;
                    context.active_mask[cell] = tick != 4;
                }
                if(context.active_mask[cell] != 0)
                    context.active_indices.push_back(cell);
            }
            std::vector<std::size_t> target = context.active_indices;
            if(!partial)
            {
                target.resize(canonical_cells.size());
                std::iota(target.begin(), target.end(), 0);
            }
            tess.BuildPartiallyParallel(canonical_points,
                std::vector<double>(canonical_points.size(), 1.0), target, true, true);
            ActiveMeshView const view(tess, canonical_cells.size());
            cells = view.gatherOwned(canonical_cells);
            std::vector<Conserved3D> extensives = view.gatherOwned(canonical_extensives);
            tess.SyncPartialBuildData(cells, canonical_cells);
            tess.SyncPartialBuildData(extensives, canonical_extensives);
            IndividualStepContext const local_context = view.remapContext(context);
            std::vector<Conserved3D> fluxes(tess.GetTotalFacesNumber());
            int local_face_count = 0;
            int wrong_interval = 0;
            for(std::size_t face = 0; face < fluxes.size(); ++face)
            {
                auto const pair = tess.GetFaceNeighbors(face);
                if(tess.BoundaryFace(face) ||
                   (pair.first >= tess.GetPointNo() && pair.second >= tess.GetPointNo()))
                    continue;
                std::size_t const left_id = cells[pair.first].ID;
                std::size_t const right_id = cells[pair.second].ID;
                if(std::min(left_id, right_id) != first_id ||
                   std::max(left_id, right_id) != second_id)
                    continue;
                ++local_face_count;
                double const expected_interval = tick == 4 ? 4 : (tick == 5 ? 1 : 3);
                wrong_interval = wrong_interval ||
                    !close(local_context.hydroFaceTimeStep(pair.first, pair.second),
                           expected_interval);
                fluxes[face].mass = (left_id == first_id ? 0.05 : -0.05) /
                    tess.GetArea(face);
            }
            int coverage[3] = {local_face_count, wrong_interval,
                partial && tess.GetPointNo() == 0 ? 1 : 0};
            MPI_Allreduce(MPI_IN_PLACE, coverage, 3, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
            require(coverage[0] > 0 && coverage[1] == 0,
                    "MPI hydro wake lost the face or used a stale ghost interval");
            require(!partial || tick == 8 || coverage[2] > 0,
                    "MPI sparse wake did not exercise a zero-owned receiver rank");
            DefaultExtensiveUpdater updater;
            updater.UpdateIndividual(fluxes, tess, local_context, cells, extensives,
                context.event_time, {}, {}, {}, &canonical_cells, &canonical_extensives);
            int invalid = 0;
            for(std::size_t cell = 0; cell < canonical_cells.size(); ++cell)
            {
                std::size_t const id = canonical_cells[cell].ID;
                double const expected_mass = 1 +
                    (id == first_id ? -0.05 * tick : (id == second_id ? 0.05 * tick : 0));
                Conserved3D const& extensive = canonical_extensives[cell];
                invalid = invalid || !close(extensive.mass, expected_mass, 1e-12) ||
                    !close(extensive.energy, 1) || extensive.Erad != 0;
                for(double const group : extensive.Eg)
                    invalid = invalid || group != 0;
            }
            for(std::size_t local = 0; local < view.localSize(); ++local)
                invalid = invalid || !close(extensives[local].mass,
                    canonical_extensives[view.localToGlobal(local)].mass, 1e-12);
            MPI_Allreduce(MPI_IN_PLACE, &invalid, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
            require(invalid == 0,
                    "MPI wake flux missed or repeated mass in a local/canonical recipient");
        }
    }
}

void testHydroRadiationSharedDonorMPI()
{
    int rank = 0;
    int rank_count = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &rank_count);
    require(rank_count >= 2, "MPI radiation donor test requires at least two ranks");
    std::size_t const group_count = Conserved3D().Eg.size();
    require(group_count > 0, "MPI radiation donor test requires radiation groups");
    // Equal group fluxes exhaust the last group's smaller budget first.
    // The one-group build retains the total-radiation exhaustion case.
    std::vector<double> initial_group_energy(group_count,
        group_count > 1 ? (1 - 0.5 / group_count) / (group_count - 1) : 1);
    initial_group_energy.back() = group_count > 1 ? 0.5 / group_count : 1;
    std::vector<Vector3D> points;
    for(double const x : {0.25, 0.75})
        for(double const y : {0.25, 0.75})
            for(double const z : {0.25, 0.75})
                points.emplace_back((rank + x) / rank_count, y, z);
    Voronoi3D tess(Vector3D(0, 0, 0), Vector3D(1, 1, 1));
    tess.BuildParallel(points);
    std::size_t const owned_count = tess.GetPointNo();
    std::vector<ComputationalCell3D> cells(owned_count);
    std::vector<Conserved3D> extensives(owned_count);
    for(std::size_t cell = 0; cell < owned_count; ++cell)
    {
        cells[cell].ID = 850000 + rank * 10000 + cell;
        cells[cell].density = 1;
        cells[cell].pressure = 1;
        cells[cell].internal_energy = 1;
        extensives[cell].mass = 1;
        extensives[cell].energy = 1;
        extensives[cell].internal_energy = 1;
        extensives[cell].Erad = 1;
        for(std::size_t group = 0; group < group_count; ++group)
            extensives[cell].Eg[group] = initial_group_energy[group];
    }
    std::vector<ComputationalCell3D> canonical_cells;
    std::vector<Conserved3D> canonical_extensives;
    tess.SyncPartialBuildData(cells, canonical_cells);
    tess.SyncPartialBuildData(extensives, canonical_extensives);
    std::vector<Conserved3D> const initial_extensives = extensives;
    std::vector<Conserved3D> const initial_canonical_extensives = canonical_extensives;
    std::uint64_t donor_id = std::numeric_limits<std::uint64_t>::max();
    for(std::size_t face = 0; face < tess.GetTotalFacesNumber(); ++face)
    {
        auto const pair = tess.GetFaceNeighbors(face);
        if(!tess.BoundaryFace(face) &&
           (pair.first < owned_count) != (pair.second < owned_count))
            donor_id = std::min(donor_id, static_cast<std::uint64_t>(
                cells[pair.first < owned_count ? pair.first : pair.second].ID));
    }
    MPI_Allreduce(MPI_IN_PLACE, &donor_id, 1, MPI_UINT64_T, MPI_MIN, MPI_COMM_WORLD);
    require(donor_id != std::numeric_limits<std::uint64_t>::max(),
            "MPI radiation donor test found no inter-rank donor");
    IndividualStepContext context;
    context.event_tick = 1;
    context.event_time = 1;
    context.time_quantum = 1;
    context.active_mask.assign(cells.size(), 1);
    context.active_indices.resize(owned_count);
    std::iota(context.active_indices.begin(), context.active_indices.end(), 0);
    context.cell_time_steps.assign(cells.size(), 1);
    context.primitive_ticks.assign(cells.size(), 0);
    std::vector<Conserved3D> fluxes(tess.GetTotalFacesNumber());
    std::vector<int> receiver_faces(owned_count, 0);
    int donor_faces = 0;
    for(std::size_t face = 0; face < fluxes.size(); ++face)
    {
        auto const pair = tess.GetFaceNeighbors(face);
        if(tess.BoundaryFace(face) ||
           (pair.first >= owned_count && pair.second >= owned_count))
            continue;
        bool const donor_first = cells[pair.first].ID == donor_id;
        bool const donor_second = cells[pair.second].ID == donor_id;
        if(!donor_first && !donor_second)
            continue;
        std::size_t const donor = donor_first ? pair.first : pair.second;
        std::size_t const receiver = donor_first ? pair.second : pair.first;
        donor_faces += donor < owned_count ? 1 : 0;
        if(receiver < owned_count)
            ++receiver_faces[receiver];
        double const outward_flux = (donor_first ? 2.0 : -2.0) / tess.GetArea(face);
        fluxes[face].Erad = outward_flux;
        for(double& group_flux : fluxes[face].Eg)
            group_flux = outward_flux / group_count;
        fluxes[face].Erad_dt = 0.5 * outward_flux;
        fluxes[face].Erad_dt_dt = -0.25 * outward_flux;
    }
    MPI_Allreduce(MPI_IN_PLACE, &donor_faces, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    require(donor_faces >= 2, "MPI radiation donor test did not exercise shared outflow");
    double initial_total = static_cast<double>(owned_count);
    MPI_Allreduce(MPI_IN_PLACE, &initial_total, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    DefaultExtensiveUpdater updater;
    updater.UpdateIndividual(fluxes, tess, context, cells, extensives, 1,
        {}, {}, {}, &canonical_cells, &canonical_extensives);
    double donor_remaining = 0;
    for(std::size_t cell = 0; cell < owned_count; ++cell)
        if(cells[cell].ID == donor_id)
            donor_remaining = extensives[cell].Erad;
    MPI_Allreduce(MPI_IN_PLACE, &donor_remaining, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    double const transfer_per_face = (1 - donor_remaining) / donor_faces;
    double const expected_donor_remaining = group_count > 1 ? 0.5 : 0;
    int invalid = donor_remaining < 0 ||
        !close(donor_remaining, expected_donor_remaining, 2e-12);
    std::vector<double> totals(group_count + 1, 0);
    for(std::size_t cell = 0; cell < owned_count; ++cell)
    {
        double const radiation_change = cells[cell].ID == donor_id ?
            donor_remaining - 1 : receiver_faces[cell] * transfer_per_face;
        Conserved3D const& extensive = extensives[cell];
        invalid = invalid || !close(extensive.Erad, 1 + radiation_change, 1e-12) ||
            !close(extensive.Erad_dt, 0.5 * radiation_change, 1e-12) ||
            !close(extensive.Erad_dt_dt, -0.25 * radiation_change, 1e-12) ||
            extensive.mass != 1 || extensive.energy != 1;
        double group_sum = 0;
        for(std::size_t group = 0; group < group_count; ++group)
        {
            double const extent = extensive.Eg[group];
            invalid = invalid || !std::isfinite(extent) || extent < 0 ||
                !close(extent, initial_group_energy[group] +
                    radiation_change / group_count, 1e-12);
            group_sum += extent;
            totals[group + 1] += extent;
        }
        invalid = invalid || !close(extensive.Erad, group_sum, 1e-12);
        if(cells[cell].ID == donor_id)
            invalid = invalid || extensive.Eg.back() >
                2e-12 * initial_group_energy.back();
        totals[0] += extensive.Erad;
    }
    MPI_Allreduce(MPI_IN_PLACE, &invalid, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, totals.data(), static_cast<int>(totals.size()),
                  MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    bool conserved = close(totals[0], initial_total, 1e-12);
    for(std::size_t group = 0; group < group_count; ++group)
        conserved = conserved && close(totals[group + 1],
            initial_total * initial_group_energy[group], 1e-12);
    require(invalid == 0 && conserved,
            "MPI shared radiation donor lost positivity, common scaling, or conservation");

    extensives = initial_extensives;
    canonical_extensives = initial_canonical_extensives;
    for(Conserved3D& flux : fluxes)
        flux.Eg.back() = -flux.Eg.back();
    int mixed_direction_rejected = 0;
    try
    {
        updater.UpdateIndividual(fluxes, tess, context, cells, extensives, 1,
            {}, {}, {}, &canonical_cells, &canonical_extensives);
    }
    catch(std::runtime_error const& error)
    {
        mixed_direction_rejected = std::string(error.what()).find("issue mask 2") !=
            std::string::npos ? 1 : 0;
    }
    int unchanged = 1;
    for(std::size_t cell = 0; cell < owned_count; ++cell)
    {
        unchanged = unchanged && extensives[cell].mass == 1 &&
            extensives[cell].Erad == 1;
        for(std::size_t group = 0; group < group_count; ++group)
            unchanged = unchanged &&
                extensives[cell].Eg[group] == initial_group_energy[group];
    }
    int rejection[2] = {mixed_direction_rejected, unchanged};
    MPI_Allreduce(MPI_IN_PLACE, rejection, 2, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    require(rejection[0] != 0 && rejection[1] != 0,
            "MPI mixed-direction radiation flux was not collectively rejected before mutation");
}

void testHydroMaterialUpdateMPI()
{
	int rank = 0;
	int rank_count = 1;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &rank_count);
	require(rank_count >= 2,
		"MPI material positivity test requires at least two ranks");

	std::vector<Vector3D> points;
	for(double const local_x : {0.25, 0.75})
		for(double const y : {0.25, 0.75})
			for(double const z : {0.25, 0.75})
				points.push_back(Vector3D(
					(static_cast<double>(rank) + local_x) /
						static_cast<double>(rank_count), y, z));
	Voronoi3D tess(Vector3D(0, 0, 0), Vector3D(1, 1, 1));
	tess.BuildParallel(points);
	std::size_t const owned_count = tess.GetPointNo();
	std::vector<ComputationalCell3D> cells(owned_count);
	std::vector<Conserved3D> extensives(owned_count);
	for(std::size_t cell = 0; cell < owned_count; ++cell)
	{
		cells[cell].ID = 915000 +
			static_cast<std::size_t>(rank) * 10000 + cell;
		cells[cell].density = 1;
		cells[cell].pressure = 1;
		cells[cell].internal_energy = 1;
		extensives[cell].mass = 1;
		extensives[cell].energy = 1;
		extensives[cell].internal_energy = 1;
		// This test has no radiation flux, but the updater requires a valid
		// positive total radiation extent before limiting any event.
		extensives[cell].Erad = 1;
		extensives[cell].Eg[0] = 1;
	}
	std::vector<ComputationalCell3D> canonical_cells;
	std::vector<Conserved3D> canonical_extensives;
	tess.SyncPartialBuildData(cells, canonical_cells);
	tess.SyncPartialBuildData(extensives, canonical_extensives);

	IndividualStepContext context;
	context.active_indices.resize(owned_count);
	std::iota(context.active_indices.begin(), context.active_indices.end(), 0);
	context.active_mask.assign(cells.size(), 1);
	context.cell_time_steps.assign(cells.size(), 1);
	context.time_quantum = 1;
	context.event_tick = 1;
	context.event_time = 1;
	context.primitive_ticks.assign(cells.size(), 0);
	std::vector<Conserved3D> fluxes(tess.GetTotalFacesNumber());
	std::size_t local_remote_faces = 0;
	std::vector<double> expected_mass(owned_count, 1);
	std::vector<double> expected_energy(owned_count, 1);
	for(std::size_t face = 0; face < tess.GetTotalFacesNumber(); ++face)
	{
		if(tess.BoundaryFace(face))
			continue;
		auto const neighbors = tess.GetFaceNeighbors(face);
		bool const first_owned = neighbors.first < owned_count;
		bool const second_owned = neighbors.second < owned_count;
		if(first_owned == second_owned)
			continue;
		++local_remote_faces;
		// Both MPI copies describe the same transfer from lower to higher ID,
		// even when their local face-neighbor order is reversed.
		double const face_orientation =
			cells[neighbors.first].ID < cells[neighbors.second].ID ? 1 : -1;
		fluxes[face].mass = face_orientation * 0.25 / tess.GetArea(face);
		fluxes[face].energy = -face_orientation * 0.125 / tess.GetArea(face);
		std::size_t const owned_endpoint = first_owned ? neighbors.first : neighbors.second;
		double const orientation = (first_owned ? -1 : 1) * face_orientation;
		expected_mass[owned_endpoint] += orientation * 0.25;
		expected_energy[owned_endpoint] -= orientation * 0.125;
	}
	std::uint64_t global_remote_faces =
		static_cast<std::uint64_t>(local_remote_faces);
	MPI_Allreduce(MPI_IN_PLACE, &global_remote_faces, 1, MPI_UINT64_T,
		MPI_SUM, MPI_COMM_WORLD);
	require(global_remote_faces > 0,
		"MPI material positivity test created no inter-rank face");

	double local_mass_before = static_cast<double>(owned_count);
	double local_energy_before = static_cast<double>(owned_count);
	double global_before[2] = {local_mass_before, local_energy_before};
	MPI_Allreduce(MPI_IN_PLACE, global_before, 2, MPI_DOUBLE, MPI_SUM,
		MPI_COMM_WORLD);
	DefaultExtensiveUpdater updater;
	updater.UpdateIndividual(fluxes, tess, context, cells, extensives, 0,
		std::vector<Vector3D>(), std::vector<Vector3D>(),
		std::vector<std::pair<ComputationalCell3D, ComputationalCell3D> >(),
		&canonical_cells, &canonical_extensives);
	double local_after[2] = {0, 0};
	int local_invalid = 0;
	for(std::size_t cell = 0; cell < owned_count; ++cell)
	{
		local_invalid = local_invalid || !(extensives[cell].mass > 0) ||
			!(extensives[cell].energy > 0) ||
			!(extensives[cell].internal_energy > 0) ||
			!close(extensives[cell].mass, expected_mass[cell], 1e-12) ||
			!close(extensives[cell].energy, expected_energy[cell], 1e-12);
		local_after[0] += extensives[cell].mass;
		local_after[1] += extensives[cell].energy;
	}
	MPI_Allreduce(MPI_IN_PLACE, local_after, 2, MPI_DOUBLE, MPI_SUM,
		MPI_COMM_WORLD);
	MPI_Allreduce(MPI_IN_PLACE, &local_invalid, 1, MPI_INT, MPI_MAX,
		MPI_COMM_WORLD);
	require(local_invalid == 0,
		"MPI valid material transfer left an invalid extent");
	require(close(global_before[0], local_after[0], 1e-11) &&
		close(global_before[1], local_after[1], 1e-11),
		"MPI material transfer is not conservative");

	// Material fluxes are applied without clipping.  Overspending a donor
	// must produce the same invalid-mass failure on every rank.
	for(Conserved3D& extensive : extensives)
	{
		extensive.mass = 1;
		extensive.energy = 1;
		extensive.internal_energy = 1;
	}
	canonical_extensives = extensives;
	for(Conserved3D& flux : fluxes)
	{
		flux.mass *= 40;
		flux.energy *= 16;
	}
	bool overspend_rejected = false;
	const std::string overspend_notice = captureStandardError([&]()
	{
		try
		{
			updater.UpdateIndividual(fluxes, tess, context, cells, extensives, 0,
				std::vector<Vector3D>(), std::vector<Vector3D>(),
				std::vector<std::pair<ComputationalCell3D, ComputationalCell3D> >(),
				&canonical_cells, &canonical_extensives);
		}
		catch(std::runtime_error const& error)
		{
			std::string const message = error.what();
			overspend_rejected = message.find("produced an invalid state") !=
				std::string::npos && message.find("component mask 16") !=
				std::string::npos;
		}
	});
	int all_rejected = overspend_rejected ? 1 : 0;
	MPI_Allreduce(MPI_IN_PLACE, &all_rejected, 1, MPI_INT, MPI_MIN,
		MPI_COMM_WORLD);
	require(all_rejected == 1, "MPI material overspend did not reject collectively");
	std::uint64_t selected_invalid_id = std::numeric_limits<std::uint64_t>::max();
	double expected_invalid_mass = 0;
	int oversized_updates_exact = 1;
	for(std::size_t cell = 0; cell < owned_count; ++cell)
	{
		double const expected_oversized_mass = 1 + 40 * (expected_mass[cell] - 1);
		double const expected_oversized_energy = 1 + 16 * (expected_energy[cell] - 1);
		oversized_updates_exact = oversized_updates_exact &&
			close(extensives[cell].mass, expected_oversized_mass, 1e-12) &&
			close(extensives[cell].energy, expected_oversized_energy, 1e-12);
		if(extensives[cell].mass < 0 && cells[cell].ID < selected_invalid_id)
		{
			selected_invalid_id = cells[cell].ID;
			expected_invalid_mass = expected_oversized_mass;
		}
	}
	MPI_Allreduce(MPI_IN_PLACE, &oversized_updates_exact, 1, MPI_INT,
		MPI_MIN, MPI_COMM_WORLD);
	require(oversized_updates_exact == 1,
		"MPI rejected material update changed the prescribed face transfers");
	std::uint64_t const local_invalid_id = selected_invalid_id;
	MPI_Allreduce(MPI_IN_PLACE, &selected_invalid_id, 1, MPI_UINT64_T,
		MPI_MIN, MPI_COMM_WORLD);
	int overspend_diagnostic_valid = 1;
	try
	{
		if(local_invalid_id == selected_invalid_id)
		{
			const DiagnosticFile diagnostic = readDiagnosticFile(overspend_notice);
			std::ostringstream expected_post_mass;
			expected_post_mass << expected_invalid_mass;
			overspend_diagnostic_valid =
				diagnostic.contents.find("reconstructed_pre_mass=1") != std::string::npos &&
				diagnostic.contents.find("post_mass=" + expected_post_mass.str() + " ") != std::string::npos &&
				diagnostic.contents.find("delta_mass=-10 ") != std::string::npos &&
				diagnostic.contents.find("replayed_post_mass=" + expected_post_mass.str() +
					" actual_post_mass=" + expected_post_mass.str() + "\n") != std::string::npos;
		}
		else
			overspend_diagnostic_valid = overspend_notice.empty();
	}
	catch(std::exception const&)
	{
		overspend_diagnostic_valid = 0;
	}
	MPI_Allreduce(MPI_IN_PLACE, &overspend_diagnostic_valid, 1, MPI_INT,
		MPI_MIN, MPI_COMM_WORLD);
	require(overspend_diagnostic_valid == 1,
		"MPI overspend diagnostic lost the exact face withdrawal and replay");

	// A one-rank NaN mass must also fail collectively after delta application.
	extensives.assign(owned_count, Conserved3D());
	for(Conserved3D& extensive : extensives)
	{
		extensive.mass = 1;
		extensive.energy = 1;
		extensive.internal_energy = 1;
		extensive.Erad = 1;
		extensive.Eg[0] = 1;
	}
	canonical_extensives = extensives;
	if(rank == 0 && !extensives.empty())
		extensives[0].mass = std::numeric_limits<double>::quiet_NaN();
	bool threw = false;
	const std::string invalid_notice = captureStandardError([&]()
	{
		try
		{
			updater.UpdateIndividual(
				std::vector<Conserved3D>(tess.GetTotalFacesNumber()), tess,
				context, cells, extensives, 0, std::vector<Vector3D>(),
				std::vector<Vector3D>(),
				std::vector<std::pair<ComputationalCell3D,
					ComputationalCell3D> >(), &canonical_cells,
				&canonical_extensives);
		}
		catch(std::runtime_error const&)
		{
			threw = true;
		}
	});
	int collective_throw[2] = {threw ? 1 : 0, threw ? 1 : 0};
	MPI_Allreduce(MPI_IN_PLACE, &collective_throw[0], 1, MPI_INT, MPI_MIN,
		MPI_COMM_WORLD);
	MPI_Allreduce(MPI_IN_PLACE, &collective_throw[1], 1, MPI_INT, MPI_MAX,
		MPI_COMM_WORLD);
	require(collective_throw[0] == 1 && collective_throw[1] == 1,
		"invalid material input did not throw collectively on every rank");
	int invalid_diagnostic_valid = 1;
	try
	{
		if(rank == 0)
		{
			const DiagnosticFile invalid_diagnostic = readDiagnosticFile(invalid_notice);
			invalid_diagnostic_valid = invalid_diagnostic.contents.find(
				"INDIVIDUAL_HYDRO_INVALID_INPUT") != std::string::npos &&
				invalid_diagnostic.contents.find("phase=before_flux_application") != std::string::npos &&
				invalid_diagnostic.contents.find("mass=nan") != std::string::npos;
		}
		else
			invalid_diagnostic_valid = invalid_notice.empty();
	}
	catch(std::exception const&)
	{
		invalid_diagnostic_valid = 0;
	}
	MPI_Allreduce(MPI_IN_PLACE, &invalid_diagnostic_valid, 1, MPI_INT,
		MPI_MIN, MPI_COMM_WORLD);
	require(invalid_diagnostic_valid == 1,
		"collective invalid-mass diagnostic lost the failing state");
}

void testIndividualBoundaryGhostPredictionMPI()
{
    int rank = 0;
    int rank_count = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &rank_count);
    require(rank_count >= 2,
            "MPI boundary-predictor test requires at least two ranks");

    std::vector<Vector3D> points;
    for(double const local_x : {0.25, 0.75})
        for(double const y : {0.25, 0.75})
            for(double const z : {0.25, 0.75})
                points.push_back(Vector3D(
                    (static_cast<double>(rank) + local_x) /
                        static_cast<double>(rank_count),
                    y, z));

    Voronoi3D tess(Vector3D(0, 0, 0), Vector3D(1, 1, 1));
    tess.BuildParallel(points);
    std::vector<ComputationalCell3D> cells(tess.GetPointNo());
    for(std::size_t index = 0; index < cells.size(); ++index)
    {
        cells[index].ID = 920000 +
            static_cast<std::size_t>(rank) * 10000 + index;
        cells[index].density = 1;
        cells[index].pressure = 1;
        cells[index].internal_energy = 1.5;
    }
    std::vector<ComputationalCell3D> canonical_cells;
    tess.SyncPartialBuildData(cells, canonical_cells);

    std::size_t const owned_cell_count = tess.GetPointNo();
    std::size_t local_boundary_ghosts = 0;
    std::size_t remote_face = tess.GetTotalFacesNumber();
    std::size_t remote_cell = cells.size();
    bool remote_is_first = false;
    for(std::size_t face = 0; face < tess.GetTotalFacesNumber(); ++face)
    {
        auto const neighbors = tess.GetFaceNeighbors(face);
        if(tess.BoundaryFace(face))
        {
            for(std::size_t const neighbor : {neighbors.first, neighbors.second})
                if(neighbor >= owned_cell_count &&
                   neighbor < cells.size() && cells[neighbor].density == 0)
                    ++local_boundary_ghosts;
            continue;
        }
        bool const first_owned = neighbors.first < owned_cell_count;
        bool const second_owned = neighbors.second < owned_cell_count;
        if(remote_face == tess.GetTotalFacesNumber() &&
           first_owned != second_owned)
        {
            remote_face = face;
            remote_is_first = !first_owned;
            remote_cell = remote_is_first ? neighbors.first : neighbors.second;
            if(remote_cell >= cells.size())
                remote_face = tess.GetTotalFacesNumber();
        }
    }
    int local_has_boundary_ghost = local_boundary_ghosts > 0 ? 1 : 0;
    int global_has_boundary_ghost = 0;
    MPI_Allreduce(&local_has_boundary_ghost, &global_has_boundary_ghost, 1,
                  MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    require(global_has_boundary_ghost != 0,
            "MPI boundary-predictor test created no physical ghost cells");
    int local_has_remote_ghost =
        remote_face < tess.GetTotalFacesNumber() ? 1 : 0;
    int global_has_remote_ghost = 0;
    MPI_Allreduce(&local_has_remote_ghost, &global_has_remote_ghost, 1,
                  MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    require(global_has_remote_ghost != 0,
            "MPI boundary-predictor test created no in-domain remote ghost");

    IndividualStepContext context;
    context.previous_event_tick = 0;
    context.event_tick = 1;
    context.previous_event_time = 0;
    context.event_time = 1;
    context.time_origin = 0;
    context.time_quantum = 1;
    context.active_mask.assign(cells.size(), 0);
    context.cell_time_steps.assign(cells.size(), 1);
    context.primitive_ticks.assign(cells.size(), 0);
    context.cached_accelerations.assign(cells.size(), Vector3D());
    context.active_indices.resize(tess.GetPointNo());
    std::iota(context.active_indices.begin(), context.active_indices.end(), 0);
    std::fill(context.active_mask.begin(),
              context.active_mask.begin() + tess.GetPointNo(), 1);

    class EveryFace : public ConditionActionFlux1::Condition3D
    {
    public:
        std::pair<bool, bool> operator()(
            std::size_t, Tessellation3D const&,
            std::vector<ComputationalCell3D> const&) const override
        {
            return std::make_pair(true, false);
        }
    } every_face;
    ZeroFlux3D zero_flux;
    std::vector<std::pair<ConditionActionFlux1::Condition3D const*,
                          ConditionActionFlux1::Action3D const*> > sequence{
        std::make_pair(&every_face, &zero_flux)};
    IdealGas eos(5.0 / 3.0);
    RigidWallGenerator3D ghost;
    LinearGauss3D reconstruction(eos, ghost);
    ConditionActionFlux1 calculator(sequence, reconstruction);
    std::vector<std::pair<ComputationalCell3D, ComputationalCell3D> >
        spatial_face_values;
    reconstruction.InterpolateIndividual(tess, cells, context.event_time,
        context.active_mask, spatial_face_values);
    double const remote_acceleration_x = 2;
    double remote_prediction_offset = 0;
    if(local_has_remote_ghost != 0)
    {
        context.cached_accelerations[remote_cell].x = remote_acceleration_x;
        auto const remote_neighbors = tess.GetFaceNeighbors(remote_face);
        double const remote_face_dt = context.hydroFaceTimeStep(
            remote_neighbors.first, remote_neighbors.second);
        remote_prediction_offset = context.event_time - 0.5 * remote_face_dt -
            (context.time_origin + context.time_quantum *
             static_cast<double>(context.primitive_ticks[remote_cell]));
    }
    std::vector<Vector3D> face_velocities(
        tess.GetTotalFacesNumber(), Vector3D());
    std::vector<Conserved3D> fluxes;
    std::vector<Conserved3D> extensives;
    std::vector<std::pair<ComputationalCell3D, ComputationalCell3D> >
        face_values;

    std::feclearexcept(FE_ALL_EXCEPT);
    calculator.CalculateIndividual(fluxes, tess, face_velocities, cells,
        extensives, eos, context, face_values);
    int local_invalid =
        std::fetestexcept(FE_INVALID | FE_DIVBYZERO) == 0 ? 0 : 1;
    int local_remote_prediction_error = 0;
    if(local_has_remote_ghost != 0)
    {
        ComputationalCell3D const& spatial_value = remote_is_first ?
            spatial_face_values[remote_face].first :
            spatial_face_values[remote_face].second;
        ComputationalCell3D const& predicted_value = remote_is_first ?
            face_values[remote_face].first : face_values[remote_face].second;
        double const expected_velocity_x = spatial_value.velocity.x +
            remote_prediction_offset * remote_acceleration_x;
        local_remote_prediction_error =
            close(predicted_value.velocity.x, expected_velocity_x, 1e-12) ? 0 : 1;
    }
    int local_errors[2] = {local_invalid, local_remote_prediction_error};
    MPI_Allreduce(MPI_IN_PLACE, local_errors, 2, MPI_INT, MPI_MAX,
                  MPI_COMM_WORLD);
    std::feclearexcept(FE_ALL_EXCEPT);
    require(local_errors[0] == 0,
            "physical boundary ghost entered the temporal predictor");
    require(local_errors[1] == 0,
            "in-domain remote ghost skipped the temporal predictor");

    // Make the ghost slopes nonuniform, then deliberately reorder only the
    // sender-side duplicate metadata.  A positional exchange now assigns at
    // least one current slope to the wrong ghost; ID-matched exchange must not.
    for(std::size_t index = 0; index < tess.GetPointNo(); ++index)
    {
        Vector3D const point = tess.GetMeshPoint(index);
        cells[index].density =
            1 + 0.3 * point.x * point.x + 0.2 * point.y * point.z;
        cells[index].pressure =
            1 + 0.25 * point.y * point.y + 0.15 * point.x * point.z;
        cells[index].internal_energy =
            cells[index].pressure / ((5.0 / 3.0 - 1) * cells[index].density);
        cells[index].velocity = Vector3D(
            0.07 * point.x * point.y,
            -0.05 * point.y * point.z,
            0.03 * point.x * point.x);
    }
    tess.SyncPartialBuildData(cells, canonical_cells);

    LinearGauss3D full_reconstruction(eos, ghost);
    std::vector<std::pair<ComputationalCell3D, ComputationalCell3D> >
        full_face_values;
    std::feclearexcept(FE_ALL_EXCEPT);
    full_reconstruction.InterpolateIndividual(tess, cells,
        context.event_time, context.active_mask, full_face_values);
    int unequal_pressure_invalid =
        std::fetestexcept(FE_INVALID | FE_DIVBYZERO) == 0 ? 0 : 1;
    MPI_Allreduce(MPI_IN_PLACE, &unequal_pressure_invalid, 1, MPI_INT,
                  MPI_MAX, MPI_COMM_WORLD);
    std::feclearexcept(FE_ALL_EXCEPT);
    require(unequal_pressure_invalid == 0,
            "unequal positive pressures raised a reconstruction exception");
    std::vector<Slope3D> const expected_slopes =
        full_reconstruction.GetSlopes();

    auto reverse_duplicate_order = [&]()
    {
        for(std::vector<std::size_t>& peer_points :
            tess.GetDuplicatedPoints())
            if(peer_points.size() > 1)
                std::reverse(peer_points.begin(), peer_points.end());
    };
    reverse_duplicate_order();

    std::vector<Slope3D> legacy_slopes = expected_slopes;
    std::vector<Slope3D> legacy_canonical_slopes;
    tess.SyncPartialBuildData(legacy_slopes, legacy_canonical_slopes);

    std::size_t permuted_face = tess.GetTotalFacesNumber();
    bool permuted_remote_is_first = false;
    for(std::size_t face = 0; face < tess.GetTotalFacesNumber(); ++face)
    {
        if(tess.BoundaryFace(face))
            continue;
        auto const neighbors = tess.GetFaceNeighbors(face);
        bool const first_owned = neighbors.first < tess.GetPointNo();
        bool const second_owned = neighbors.second < tess.GetPointNo();
        if(first_owned == second_owned)
            continue;
        std::size_t const ghost = first_owned ?
            neighbors.second : neighbors.first;
        if(ghost >= expected_slopes.size() || ghost >= legacy_slopes.size())
            continue;
        Slope3D const& expected = expected_slopes[ghost];
        Slope3D const& legacy = legacy_slopes[ghost];
        if(!close(expected.xderivative.density,
                  legacy.xderivative.density, 1e-12) ||
           !close(expected.yderivative.density,
                  legacy.yderivative.density, 1e-12) ||
           !close(expected.zderivative.density,
                  legacy.zderivative.density, 1e-12))
        {
            permuted_face = face;
            permuted_remote_is_first = !first_owned;
            break;
        }
    }
    int local_has_permuted_face =
        permuted_face < tess.GetTotalFacesNumber() ? 1 : 0;
    int global_has_permuted_face = local_has_permuted_face;
    MPI_Allreduce(MPI_IN_PLACE, &global_has_permuted_face, 1, MPI_INT,
                  MPI_MAX, MPI_COMM_WORLD);
    require(global_has_permuted_face != 0,
            "MPI ghost-slope regression did not perturb a boundary slope");

    reverse_duplicate_order();
    std::vector<unsigned char> sparse_active(cells.size(), 0);
    if(local_has_permuted_face != 0)
    {
        auto const neighbors = tess.GetFaceNeighbors(permuted_face);
        std::size_t const owned = neighbors.first < tess.GetPointNo() ?
            neighbors.first : neighbors.second;
        sparse_active[owned] = 1;
    }
    std::vector<unsigned char> canonical_sparse_active(
        canonical_cells.size(), 0);
    tess.SyncPartialBuildData(sparse_active, canonical_sparse_active);
    reverse_duplicate_order();

    std::vector<std::pair<ComputationalCell3D, ComputationalCell3D> >
        sparse_face_values;
    int local_sparse_reconstruction_error = 0;
    try
    {
        reconstruction.InterpolateIndividual(tess, cells, context.event_time,
            sparse_active, sparse_face_values);
    }
    catch(...)
    {
        local_sparse_reconstruction_error = 1;
    }
    int global_sparse_reconstruction_error = local_sparse_reconstruction_error;
    MPI_Allreduce(MPI_IN_PLACE, &global_sparse_reconstruction_error, 1,
                  MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    require(global_sparse_reconstruction_error == 0,
            "sparse MPI ghost reconstruction failed");

    int local_sparse_comparison_error = 0;
    if(local_has_permuted_face != 0)
    {
        ComputationalCell3D const& sparse_remote = permuted_remote_is_first ?
            sparse_face_values[permuted_face].first :
            sparse_face_values[permuted_face].second;
        ComputationalCell3D const& full_remote = permuted_remote_is_first ?
            full_face_values[permuted_face].first :
            full_face_values[permuted_face].second;
        local_sparse_comparison_error =
            close(sparse_remote.density, full_remote.density, 2e-7) &&
            close(sparse_remote.pressure, full_remote.pressure, 2e-7) &&
            close(sparse_remote.internal_energy,
                  full_remote.internal_energy, 2e-7) &&
            close(sparse_remote.velocity, full_remote.velocity, 2e-7) ? 0 : 1;
    }
    int global_sparse_comparison_error = local_sparse_comparison_error;
    MPI_Allreduce(MPI_IN_PLACE, &global_sparse_comparison_error, 1,
                  MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    require(global_sparse_comparison_error == 0,
            "sparse MPI ghost reconstruction used a stale or mis-mapped slope");
}

void testIndividualCellUpdateOnSparsePartialMeshMPI()
{
    int rank = 0;
    int rank_count = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &rank_count);
    require(rank_count >= 2,
            "MPI sparse cell-update test requires at least two ranks");

    std::vector<Vector3D> seed_points;
    for(std::size_t ix = 0; ix < 12; ++ix)
        for(std::size_t iy = 0; iy < 4; ++iy)
            for(std::size_t iz = 0; iz < 4; ++iz)
                seed_points.push_back(Vector3D(
                    (static_cast<double>(rank) +
                     (static_cast<double>(ix) + 0.5) / 12.0) /
                        static_cast<double>(rank_count),
                    (static_cast<double>(iy) + 0.5) / 4.0,
                    (static_cast<double>(iz) + 0.5) / 4.0));

    Voronoi3D tess(Vector3D(0, 0, 0), Vector3D(1, 1, 1));
    tess.BuildParallel(seed_points);
    std::vector<Vector3D> const canonical_points = tess.getAllPoints();
    require(tess.GetPointNo() == canonical_points.size(),
            "full MPI build has inconsistent owned/all-point counts");

    IdealGas eos(5.0 / 3.0, 1.5, 1.0, 0.0);
    std::vector<std::string> const saved_tracer_names =
        ComputationalCell3D::tracerNames;
    ComputationalCell3D::tracerNames.assign(1, "Entropy");
    std::vector<ComputationalCell3D> canonical_cells(
        canonical_points.size());
    std::vector<Conserved3D> canonical_extensives(
        canonical_points.size());
    for(std::size_t point = 0; point < canonical_points.size(); ++point)
    {
        ComputationalCell3D& cell = canonical_cells[point];
        cell.ID = 930000 + static_cast<std::size_t>(rank) * 10000 + point;
        cell.density = 1.0 + 0.1 * canonical_points[point].x;
        cell.pressure = 0.8 + 0.1 * canonical_points[point].y;
        cell.internal_energy = eos.dp2e(
            cell.density, cell.pressure, cell.tracers,
            ComputationalCell3D::tracerNames);
        cell.tracers[0] = eos.dp2s(
            cell.density, cell.pressure, cell.tracers,
            ComputationalCell3D::tracerNames);
        cell.temperature = eos.de2T(
            cell.density, cell.internal_energy, cell.tracers,
            ComputationalCell3D::tracerNames);
        double const equilibrium_radiation_energy_density =
            CG::radiation_constant * std::pow(cell.temperature, 4);
        cell.Erad =
            (1.1 + 0.05 * canonical_points[point].z) *
            equilibrium_radiation_energy_density / cell.density;
        PrimitiveToConserved(
            cell, tess.GetVolume(point), canonical_extensives[point]);
    }

    std::vector<std::size_t> target;
    if(rank % 2 == 0 && !canonical_points.empty())
    {
        for(std::vector<std::size_t> const& peer_points :
            tess.GetDuplicatedAllPointIndices())
            if(!peer_points.empty())
            {
                target.push_back(peer_points.front());
                break;
            }
        if(target.empty())
            target.push_back(0);
    }
    tess.BuildPartiallyParallel(
        canonical_points,
        std::vector<double>(canonical_points.size(), 1.0),
        target, true, true);
    int global_has_zero_owned_rank = tess.GetPointNo() == 0 ? 1 : 0;
    MPI_Allreduce(MPI_IN_PLACE, &global_has_zero_owned_rank, 1, MPI_INT,
                  MPI_MAX, MPI_COMM_WORLD);
    require(global_has_zero_owned_rank != 0,
            "sparse cell-update test created no zero-owned rank");
    ActiveMeshView const view(tess, canonical_points.size());
    std::vector<ComputationalCell3D> local_cells =
        view.gatherOwned(canonical_cells);
    std::vector<Conserved3D> local_extensives =
        view.gatherOwned(canonical_extensives);
    tess.SyncPartialBuildData(local_cells, canonical_cells);
    tess.SyncPartialBuildData(local_extensives, canonical_extensives);

    bool local_has_wide_duplicate = false;
    for(std::vector<std::size_t> const& peer_points :
        tess.GetDuplicatedAllPointIndices())
        for(std::size_t const point : peer_points)
            local_has_wide_duplicate = local_has_wide_duplicate ||
                point >= local_extensives.size();
    int global_has_wide_duplicate = local_has_wide_duplicate ? 1 : 0;
    MPI_Allreduce(MPI_IN_PLACE, &global_has_wide_duplicate, 1, MPI_INT,
                  MPI_MAX, MPI_COMM_WORLD);
    require(global_has_wide_duplicate != 0,
            "sparse cell-update test did not create an all-point/compact-index mismatch");
    int local_has_boundary_ghost = 0;
    for(std::size_t point = tess.GetPointNo();
        point < tess.getMeshPoints().size(); ++point)
        if(tess.IsPointOutsideBox(point))
        {
            local_has_boundary_ghost = 1;
            break;
        }
    int global_has_boundary_ghost = local_has_boundary_ghost;
    MPI_Allreduce(MPI_IN_PLACE, &global_has_boundary_ghost, 1, MPI_INT,
                  MPI_MAX, MPI_COMM_WORLD);
    require(global_has_boundary_ghost != 0,
            "sparse radiation test created no physical boundary ghost");

    int passive_face_checks[3] = {0, 0, 0};
    for(std::size_t face = 0; face < tess.GetTotalFacesNumber(); ++face)
    {
        if(tess.BoundaryFace(face))
            continue;
        auto const neighbors = tess.GetFaceNeighbors(face);
        bool const left_owned = neighbors.first < tess.GetPointNo();
        bool const right_owned = neighbors.second < tess.GetPointNo();
        if(left_owned == right_owned)
            continue;
        std::size_t const passive = left_owned ?
            neighbors.second : neighbors.first;
        if(passive >= local_cells.size())
        {
            passive_face_checks[2] = 1;
            continue;
        }
        std::size_t const passive_id = local_cells[passive].ID;
        bool const owned_canonically = std::any_of(
            canonical_cells.begin(), canonical_cells.end(),
            [passive_id](ComputationalCell3D const& cell)
            {return cell.ID == passive_id;});
        passive_face_checks[owned_canonically ? 0 : 1] = 1;
    }
    MPI_Allreduce(MPI_IN_PLACE, passive_face_checks, 3, MPI_INT,
                  MPI_MAX, MPI_COMM_WORLD);
    require(passive_face_checks[2] == 0,
            "sparse radiation test has an unmapped passive face endpoint");
    require(passive_face_checks[0] != 0,
            "sparse radiation test created no same-rank canonical-only passive face");
    require(passive_face_checks[1] != 0,
            "sparse radiation test created no remote passive face");

    IndividualStepContext context;
    context.active_mask.assign(local_cells.size(), 0);
    context.active_indices.resize(tess.GetPointNo());
    std::iota(context.active_indices.begin(), context.active_indices.end(), 0);
    std::fill(context.active_mask.begin(),
              context.active_mask.begin() + tess.GetPointNo(), 1);
    DefaultCellUpdater updater;
    int local_update_failed = 0;
    try
    {
        updater.UpdateIndividual(
            local_cells, eos, tess, local_extensives, context);
    }
    catch(...)
    {
        local_update_failed = 1;
    }
    int global_update_failed = local_update_failed;
    MPI_Allreduce(MPI_IN_PLACE, &global_update_failed, 1, MPI_INT,
                  MPI_MAX, MPI_COMM_WORLD);
    if(global_update_failed != 0)
    {
        ComputationalCell3D::tracerNames = saved_tracer_names;
        require(false,
                "sparse individual cell update threw on at least one rank");
    }

    int local_invalid_primitive = 0;
    for(std::size_t point = 0; point < tess.GetPointNo(); ++point)
        if(!std::isfinite(local_cells[point].density) ||
           !(local_cells[point].density > 0) ||
           !std::isfinite(local_cells[point].pressure) ||
           !(local_cells[point].pressure > 0))
            local_invalid_primitive = 1;
    int global_invalid_primitive = local_invalid_primitive;
    MPI_Allreduce(MPI_IN_PLACE, &global_invalid_primitive, 1, MPI_INT,
                  MPI_MAX, MPI_COMM_WORLD);
    if(global_invalid_primitive != 0)
    {
        ComputationalCell3D::tracerNames = saved_tracer_names;
        require(false,
                "sparse individual cell update produced an invalid primitive");
    }

    view.scatterOwned(local_cells, canonical_cells);
    view.scatterOwned(local_extensives, canonical_extensives);

    IndividualStepContext radiation_context;
    radiation_context.previous_event_tick = 0;
    radiation_context.event_tick = 1;
    radiation_context.previous_event_time = 0;
    radiation_context.event_time = 1e-4;
    radiation_context.time_origin = 0;
    radiation_context.time_quantum = 1e-4;
    radiation_context.active_indices = target;
    radiation_context.active_mask.assign(canonical_cells.size(), 0);
    for(std::size_t point : target)
        radiation_context.active_mask.at(point) = 1;
    radiation_context.cell_time_steps.assign(canonical_cells.size(), 1e-4);
    radiation_context.primitive_ticks.assign(canonical_cells.size(), 0);

    class BoundaryStateTrackingOpacity final : public OpacityCalculator
    {
    public:
        double CalcDiffusionCoefficient(
            ComputationalCell3D const& cell) const override
        {
            recordInvalidState(cell);
            return 1e-12;
        }

        double CalcPlanckOpacity(
            ComputationalCell3D const& cell) const override
        {
            recordInvalidState(cell);
            return 1e-12;
        }

        bool invalidStateSeen(void) const
        {
            return invalid_state_seen_;
        }

    private:
        void recordInvalidState(ComputationalCell3D const& cell) const
        {
            if(!(cell.density > 0) || !(cell.temperature > 0))
                invalid_state_seen_ = true;
        }

        mutable bool invalid_state_seen_ = false;
    } opacity;
    DiffusionClosedBox boundary;
    Diffusion diffusion(
        opacity, boundary, eos, {}, true, true, false, false);
    ProgressTracker tracker;
    RadiationStep radiation_step(
        tess, canonical_cells, canonical_extensives, tracker,
#ifdef RICH_MPI
        nullptr,
#endif
        diffusion, false);
    std::string first_radiation_retry;
    radiation_step.setStepRetryReporter(
        [&first_radiation_retry](StepRetryRecord const& retry)
        {
            if(!first_radiation_retry.empty())
                return;
            std::ostringstream record;
            record.precision(17);
            record << "reason=" << retry.reason
                   << " cell=" << retry.representative_cell
                   << " dt_min=" << retry.attempted_dt_min
                   << " dt_max=" << retry.attempted_dt_max
                   << " diagnostics=" << retry.diagnostics;
            first_radiation_retry = record.str();
        });
    std::string radiation_exception;
    int local_radiation_failed = 0;
    try
    {
        radiation_step.stepIndividual(radiation_context);
    }
    catch(std::exception const& error)
    {
        local_radiation_failed = 1;
        radiation_exception = error.what();
    }
    catch(...)
    {
        local_radiation_failed = 1;
        radiation_exception = "unknown exception";
    }
    int global_radiation_failed = local_radiation_failed;
    MPI_Allreduce(MPI_IN_PLACE, &global_radiation_failed, 1, MPI_INT,
                  MPI_MAX, MPI_COMM_WORLD);
    if(global_radiation_failed != 0)
    {
        ComputationalCell3D::tracerNames = saved_tracer_names;
        require(false,
                "sparse individual gray diffusion threw on at least one rank; "
                "first_retry={" + first_radiation_retry +
                "}; local_exception=" + radiation_exception);
    }

    int local_invalid_radiation = 0;
    for(std::size_t point : target)
        if(!std::isfinite(canonical_cells.at(point).Erad) ||
           canonical_cells.at(point).Erad < 0 ||
           !std::isfinite(canonical_extensives.at(point).Erad) ||
           canonical_extensives.at(point).Erad < 0)
            local_invalid_radiation = 1;
    int validation_errors[2] = {
        local_invalid_radiation, opacity.invalidStateSeen() ? 1 : 0};
    MPI_Allreduce(MPI_IN_PLACE, validation_errors, 2, MPI_INT,
                  MPI_MAX, MPI_COMM_WORLD);
    ComputationalCell3D::tracerNames = saved_tracer_names;
    require(validation_errors[1] == 0,
            "gray diffusion evaluated opacity on a zero-state boundary ghost");
    require(validation_errors[0] == 0,
            "sparse individual gray diffusion produced invalid radiation energy");
}

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
        // This synthetic driver tests two spectral groups even in a gray build.
        cells[cell].Eg.resize(2, 0);
        extensives[cell].Eg.resize(2, 0);
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

void testActiveHilbertSelectorCompatibility()
{
    char const* const selector =
        std::getenv("RICH_INDIVIDUAL_ACTIVE_HILBERT_CACHE");
    require(selector == nullptr || selector[0] == '\0' ||
            std::strcmp(selector, "0") == 0 ||
            std::strcmp(selector, "1") == 0,
            "active-Hilbert selector test requires unset, 0, or 1");
    bool const expect_required_error =
        selector != nullptr && std::strcmp(selector, "1") == 0;

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
    points = {Vector3D(0.2, 0.2, 0.2),
              Vector3D(0.8, 0.2, 0.2),
              Vector3D(0.2, 0.8, 0.8),
              Vector3D(0.8, 0.8, 0.8)};
    tess.Build(points);
#endif

    std::vector<ComputationalCell3D> cells(tess.GetPointNo());
    for(std::size_t index = 0; index < cells.size(); ++index)
    {
        cells[index].ID = 900000 +
            static_cast<std::size_t>(rank) * 10000 + index;
        cells[index].density = 1;
        cells[index].pressure = 1;
        cells[index].internal_energy = 1.5;
    }
    IdealGas eos(5.0 / 3.0);
    Simulation simulation(tess, cells, eos);
    simulation.SetTimeStepFunction(std::make_shared<ManualTimeStep>(0.125));
    std::shared_ptr<NonRebalancingIndividualStep> const physics =
        std::make_shared<NonRebalancingIndividualStep>();
    simulation.addPhysics(physics);
    IndividualTimeStepOptions options;
    options.initial_bin = 0;
    options.time_quantum = 0.125;
    simulation.EnableIndividualTimeSteps(options);

    bool caught = false;
    std::string diagnostic;
    try
    {
        simulation.step();
    }
    catch(std::logic_error const& error)
    {
        caught = true;
        diagnostic = error.what();
    }

    if(expect_required_error)
    {
        require(caught,
                "explicit active-Hilbert request accepted an incompatible setup");
#ifdef RICH_MPI
        std::string const expected =
            "Active Hilbert balancing has no supporting physics step";
#else
        std::string const expected =
            "Active Hilbert balancing requires an MPI build";
#endif
        require(diagnostic == expected,
                "explicit active-Hilbert request returned the wrong diagnostic: " +
                diagnostic);
        require(physics->individualSteps() == 0,
                "explicit active-Hilbert incompatibility reached physics");
    }
    else
    {
        require(!caught,
                "automatic or disabled active-Hilbert mode rejected legacy setup");
        require(physics->individualSteps() == 1,
                "automatic or disabled active-Hilbert mode missed physics");
    }
}

void testRuntimeStepStdout()
{
    ScopedEnvironmentVariable runtime_log("RICH_RUNTIME_LOG");
    runtime_log.set("summary");
    ScopedEnvironmentVariable runtime_color("RICH_RUNTIME_COLOR");
    runtime_color.set("never");
    ScopedEnvironmentVariable no_color("NO_COLOR");
    no_color.set(nullptr);
    ScopedEnvironmentVariable active_hilbert(
        "RICH_INDIVIDUAL_ACTIVE_HILBERT_CACHE");
    active_hilbert.set("0");

    int rank = 0;
    int rank_count = 1;
#ifdef RICH_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &rank_count);
#endif
    Voronoi3D tess(Vector3D(0, 0, 0), Vector3D(1, 1, 1));
    std::vector<Vector3D> points;
#ifdef RICH_MPI
    std::size_t global_index = 0;
    for(std::size_t i = 0; i < 2; ++i)
        for(std::size_t j = 0; j < 2; ++j)
            for(std::size_t k = 0; k < 2; ++k, ++global_index)
                if(global_index % static_cast<std::size_t>(rank_count) ==
                   static_cast<std::size_t>(rank))
                    points.push_back(Vector3D(
                        (static_cast<double>(i) + 0.5) / 2.0,
                        (static_cast<double>(j) + 0.5) / 2.0,
                        (static_cast<double>(k) + 0.5) / 2.0));
    tess.BuildParallel(points);
#else
    points = {Vector3D(0.25, 0.25, 0.25),
              Vector3D(0.75, 0.25, 0.25),
              Vector3D(0.25, 0.75, 0.75),
              Vector3D(0.75, 0.75, 0.75)};
    tess.Build(points);
#endif

    std::vector<Vector3D> const canonical_points = tess.getAllPoints();
    std::vector<ComputationalCell3D> cells(tess.GetPointNo());
    require(canonical_points.size() == cells.size(),
            "runtime stdout test has inconsistent canonical cell state");
    for(std::size_t index = 0; index < cells.size(); ++index)
    {
        cells[index].ID = static_cast<std::size_t>(rank) * 1000 + index;
        cells[index].density = 1;
        cells[index].pressure = 1;
        cells[index].internal_energy = 1.5;
    }
    std::uint64_t expected_active_cells = cells.size();
#ifdef RICH_MPI
    MPI_Allreduce(MPI_IN_PLACE, &expected_active_cells, 1, MPI_UINT64_T,
                  MPI_SUM, MPI_COMM_WORLD);
#endif
    IdealGas eos(5.0 / 3.0);

    auto capture_step = [](Simulation& simulation) {
        std::ostringstream output;
        std::streambuf* const original = std::cout.rdbuf(output.rdbuf());
        try
        {
            simulation.step();
        }
        catch(...)
        {
            std::cout.rdbuf(original);
            throw;
        }
        std::cout.rdbuf(original);
        return output.str();
    };
    auto require_fields_in_order = [](
        std::string const& output, std::string const& prefix,
        std::vector<std::string> const& fields)
    {
        std::size_t position = output.find(prefix);
        require(position != std::string::npos,
                "missing runtime record prefix " + prefix);
        std::size_t const line_end = output.find('\n', position);
        for(std::string const& field : fields)
        {
            std::size_t const next = output.find(field, position);
            require(next != std::string::npos &&
                        (line_end == std::string::npos || next < line_end),
                    "runtime field is missing or out of order: " + field);
            position = next + field.size();
        }
    };

    Simulation individual(tess, cells, eos);
    individual.SetTimeStepFunction(std::make_shared<ManualTimeStep>(0.0004));
    std::shared_ptr<NonRebalancingIndividualStep> const individual_physics =
        std::make_shared<NonRebalancingIndividualStep>();
    SourceStepTiming source_timing;
    source_timing.first_seconds = 0.01;
    source_timing.second_seconds = 0.02;
    source_timing.calls = 2;
    StepRetryRecord retry;
    retry.attempted_dt_min = 0.0004;
    retry.attempted_dt_max = 0.0004;
    retry.elapsed_seconds = 0.03;
    retry.reason = "test rejection";
    retry.representative_cell = 17;
    individual_physics->setRuntimeDiagnostics(source_timing, {retry});
    std::vector<double> individual_limits(cells.size(), 0.0008);
    if(!individual_limits.empty())
        individual_limits[0] = 0.0004;
    individual_physics->setIndividualTimeStepLimits(
        std::move(individual_limits));
    individual.addPhysics(individual_physics);
    IndividualTimeStepOptions options;
    options.initial_bin = 0;
    options.time_quantum = 0.0004;
    individual.EnableIndividualTimeSteps(options);
    individual.SetIndividualAMR([](IndividualStepContext const&) {
        IndividualAMRChangeSet changes;
        changes.mesh_build_timing.seconds = 0.06;
        changes.mesh_build_timing.builds = 2;
        return changes;
    });
    std::string const individual_output = capture_step(individual);
    if(rank == 0)
    {
        std::string const expected_histogram =
            "active_bins=[bin=0,count=" +
            std::to_string(expected_active_cells) + ",dt=0.0004]";
        std::string const expected_work_line =
            "  work   | active_cells=" +
            std::to_string(expected_active_cells) + " | total_cells=" +
            std::to_string(expected_active_cells) + " | " +
            expected_histogram + "\n";
        std::string const expected_retry_line =
            "  attempt | active_cells=" +
            std::to_string(expected_active_cells) + " | " +
            expected_histogram +
            " | attempted_dt_min=0.0004 | attempted_dt_max=0.0004\n";
        require(individual_output.find("RICH_STEP mode=individual") !=
                    std::string::npos,
                "missing individual RICH_STEP summary");
        require(individual_output.find("  time   | ") != std::string::npos &&
                    individual_output.find("  work   | ") != std::string::npos &&
                    individual_output.find("  phases | ") != std::string::npos &&
                    individual_output.find("  mesh   | ") != std::string::npos &&
                    individual_output.find("  source | ") != std::string::npos,
                "missing individual runtime subjects");
        require(individual_output.find(expected_work_line) != std::string::npos,
                "accepted active cells or bin count is not the MPI total");
        require(individual_output.find(expected_retry_line) !=
                    std::string::npos,
                "retry active cells or bin count is not the MPI total");
        require(individual_output.find(" event_dt=0.0004") !=
                    std::string::npos &&
                    individual_output.find(" applied_dt_min=0.0004") !=
                        std::string::npos &&
                    individual_output.find(" next_event_dt=0.0004") !=
                        std::string::npos,
                "individual RICH_STEP timestep fields are ambiguous or noisy");
        require(individual_output.find("Individual cycle") == std::string::npos &&
                    individual_output.find("INDIVIDUAL_PERF") ==
                        std::string::npos,
                "new runtime writer emitted a historical record");
        require(individual_output.find("RICH_RETRY mode=individual") !=
                    std::string::npos &&
                    individual_output.find(" reason=test_rejection | cell=17") !=
                        std::string::npos,
                "individual retry record is missing or malformed");
        require(individual_output.find(" source_calls=2\n\n") !=
                    std::string::npos &&
                    individual_output.find(" source_s=0.030000") !=
                        std::string::npos,
                "individual source timing summary is missing");
        require(individual_output.find("| cell=17\n\n") != std::string::npos,
                "individual retry block is missing its trailing blank line");
        require(individual_output.find(" source_first_s=") == std::string::npos &&
                    individual_output.find("RICH_STEP_DETAIL") ==
                        std::string::npos &&
                    individual_output.find("Running physics:") ==
                        std::string::npos &&
                    individual_output.find("Changed load balance") ==
                        std::string::npos &&
                    individual_output.find("INDIVIDUAL_HYDRO_PHASE_TIMING") ==
                        std::string::npos,
                "summary mode emitted detailed runtime noise");
        require_fields_in_order(
            individual_output, "RICH_STEP mode=individual",
            {" cycle="});
        require_fields_in_order(
            individual_output, "  time   |",
            {" t_start=", " t_end=", " event_dt=",
             " applied_dt_min=", " applied_dt_max=", " next_event_dt="});
        require_fields_in_order(
            individual_output, "  work   |",
            {" active_cells=", " total_cells=", " active_bins="});
        require_fields_in_order(
            individual_output, "  phases |",
            {" step_s=", " hydro_s=", " gravity_s=", " radiation_s=",
             " amr_s="});
        require_fields_in_order(
            individual_output, "  mesh   |",
            {" mesh_s=", " mesh_builds="});
        require(individual_output.find(
                    " mesh_s=0.100000 | mesh_builds=5") != std::string::npos,
                "individual AMR mesh-build diagnostics are missing");
        require_fields_in_order(
            individual_output, "  source |",
            {" source_s=", " source_pct=", " source_calls="});
        require_fields_in_order(
            individual_output, "RICH_RETRY mode=individual",
            {" cycle=", " physics=", " attempt="});
        require_fields_in_order(
            individual_output, "  attempt |",
            {" active_cells=", " active_bins=", " attempted_dt_min=",
             " attempted_dt_max="});
        require_fields_in_order(
            individual_output, "  failure |",
            {" retry_s=", " reason=", " cell="});
    }
    else
        require(individual_output.empty(),
                "nonzero MPI rank emitted runtime stdout");

    capture_step(individual); // Alignment event before bin-1 cells become inactive.
    std::vector<std::size_t> partial_target;
#ifdef RICH_MPI
    if(rank == 0 && !canonical_points.empty())
#else
    if(!canonical_points.empty())
#endif
        partial_target.push_back(0);
#ifdef RICH_MPI
    tess.BuildPartiallyParallel(
        canonical_points, std::vector<double>(canonical_points.size(), 1),
        partial_target, true, true);
#else
    tess.BuildPartially(canonical_points, partial_target);
#endif
    std::uint64_t materialized_partial_cells = tess.GetPointNo();
#ifdef RICH_MPI
    MPI_Allreduce(MPI_IN_PLACE, &materialized_partial_cells, 1,
                  MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
#endif
    require(materialized_partial_cells < expected_active_cells,
            "runtime total-cell test did not create a partial mesh view");
    std::string const partial_output = capture_step(individual);
    std::uint64_t expected_partial_active_cells = cells.empty() ? 0 : 1;
#ifdef RICH_MPI
    MPI_Allreduce(MPI_IN_PLACE, &expected_partial_active_cells, 1,
                  MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
#endif
    if(rank == 0)
    {
        std::string const expected_partial_work_line =
            "  work   | active_cells=" +
            std::to_string(expected_partial_active_cells) +
            " | total_cells=" + std::to_string(expected_active_cells) +
            " | active_bins=[bin=0,count=" +
            std::to_string(expected_partial_active_cells) +
            ",dt=0.0004]\n";
        require(expected_partial_active_cells < expected_active_cells,
                "runtime total-cell test did not create inactive cells");
        require(partial_output.find(expected_partial_work_line) !=
                    std::string::npos,
                "runtime total_cells does not include inactive owned cells");
    }
    else
        require(partial_output.empty(),
                "nonzero MPI rank emitted partial-event runtime stdout");

    std::string const partial_view_all_active_output = capture_step(individual);
    if(rank == 0)
    {
        std::string const expected_all_active_work_prefix =
            "  work   | active_cells=" +
            std::to_string(expected_active_cells) + " | total_cells=" +
            std::to_string(expected_active_cells) + " | active_bins=";
        require(partial_view_all_active_output.find(
                    expected_all_active_work_prefix) != std::string::npos,
                "runtime total_cells used the materialized partial mesh size");
    }
    else
        require(partial_view_all_active_output.empty(),
                "nonzero MPI rank emitted aligned partial-view stdout");

#ifdef RICH_MPI
    tess.BuildParallel(canonical_points, true, true);
#else
    tess.Build(canonical_points);
#endif

    auto capture_amr = [&individual](
        std::string const& mode,
        std::uint64_t const local_added_cells,
        std::uint64_t const local_cells_after)
    {
        std::ostringstream output;
        std::streambuf* const original = std::cout.rdbuf(output.rdbuf());
        try
        {
            individual.ReportRuntimeAMREvent(
                mode, 77, 1.25,
                static_cast<std::uint64_t>(individual.getCells().size()),
                local_added_cells, 0, local_cells_after);
        }
        catch(...)
        {
            std::cout.rdbuf(original);
            throw;
        }
        std::cout.rdbuf(original);
        return output.str();
    };
    std::uint64_t const local_total_cells = static_cast<std::uint64_t>(
        individual.getCells().size());
    std::string const empty_amr_output = capture_amr(
        "individual", 0, local_total_cells);
    std::string const actual_amr_output = capture_amr(
        "individual", 1, local_total_cells + 1);
    std::string const global_amr_output = capture_amr(
        "global", 1, local_total_cells + 1);
    bool rejected_unreported_cell_change = false;
    try
    {
        capture_amr("individual", 0, local_total_cells + 1);
    }
    catch(std::logic_error const&)
    {
        rejected_unreported_cell_change = true;
    }
    require(rejected_unreported_cell_change,
            "AMR cell-count change with an empty change set was not rejected");
    std::uint64_t expected_added_cells = 1;
#ifdef RICH_MPI
    MPI_Allreduce(MPI_IN_PLACE, &expected_added_cells, 1, MPI_UINT64_T,
                  MPI_SUM, MPI_COMM_WORLD);
#endif
    if(rank == 0)
    {
        require(empty_amr_output.empty(),
                "empty AMR change emitted a runtime record");
        std::string const expected_amr_line =
            "RICH_AMR mode=individual cycle=77 time=1.25 cells_before=" +
            std::to_string(expected_active_cells) + " added_cells=" +
            std::to_string(expected_added_cells) +
            " removed_cells=0 cells_after=" +
            std::to_string(expected_active_cells + expected_added_cells) +
            "\n\n";
        std::size_t const first_amr = actual_amr_output.find(expected_amr_line);
        require(first_amr != std::string::npos &&
                    actual_amr_output.find("RICH_AMR", first_amr + 1) ==
                        std::string::npos,
                 "actual AMR change did not emit exactly one global record");
        std::string global_expected = expected_amr_line;
        global_expected.replace(
            global_expected.find("mode=individual"),
            std::string("mode=individual").size(), "mode=global");
        require(global_amr_output == global_expected,
                "global AMR change did not emit the standalone record");
    }
    else
        require(empty_amr_output.empty() && actual_amr_output.empty() &&
                    global_amr_output.empty(),
                "nonzero MPI rank emitted an AMR runtime record");

    Simulation global(tess, cells, eos);
    global.SetTimeStepFunction(std::make_shared<ManualTimeStep>(0.125));
    global.addPhysics(std::make_shared<NonRebalancingIndividualStep>());
    std::string const global_output = capture_step(global);
    if(rank == 0)
    {
        require(global_output.find("RICH_STEP mode=global") != std::string::npos,
                "missing global RICH_STEP summary");
        require(global_output.find("  time   | ") != std::string::npos &&
                    global_output.find("  work   | ") != std::string::npos &&
                    global_output.find("  phases | ") != std::string::npos &&
                    global_output.find("  mesh   | ") != std::string::npos &&
                    global_output.find("  source | ") != std::string::npos,
                "missing global runtime subjects");
        require(global_output.find(" active_bins=global\n") != std::string::npos,
                "global RICH_STEP has an invalid active-bin field");
        require_fields_in_order(
            global_output, "RICH_STEP mode=global",
            {" cycle="});
        require_fields_in_order(
            global_output, "  time   |",
            {" t_start=", " t_end=", " event_dt=",
             " applied_dt_min=", " applied_dt_max=", " next_event_dt="});
        require_fields_in_order(
            global_output, "  work   |",
            {" active_cells=", " total_cells=", " active_bins="});
        require_fields_in_order(
            global_output, "  phases |",
            {" step_s=", " hydro_s=", " gravity_s=", " radiation_s=",
             " amr_s="});
        require_fields_in_order(
            global_output, "  mesh   |",
            {" mesh_s=", " mesh_builds="});
        require_fields_in_order(
            global_output, "  source |",
            {" source_s=", " source_pct=", " source_calls="});
    }
    else
        require(global_output.empty(),
                "nonzero MPI rank emitted global runtime stdout");

    runtime_log.set("detailed");
    if(rank == 0)
    {
        std::ostringstream trace_stdout;
        std::ostringstream trace_stderr;
        std::streambuf* const original_stdout =
            std::cout.rdbuf(trace_stdout.rdbuf());
        std::streambuf* const original_stderr =
            std::clog.rdbuf(trace_stderr.rdbuf());
        RuntimeTraceStream() << "RICH_RUNTIME_TRACE_TEST" << std::endl;
        std::cout.rdbuf(original_stdout);
        std::clog.rdbuf(original_stderr);
        require(trace_stdout.str() == "RICH_RUNTIME_TRACE_TEST\n",
                "detailed runtime trace did not use stdout");
        require(trace_stderr.str().empty(),
                "detailed runtime trace leaked to stderr");
    }
    Simulation detailed(tess, cells, eos);
    detailed.SetTimeStepFunction(std::make_shared<ManualTimeStep>(0.125));
    std::shared_ptr<NonRebalancingIndividualStep> const detailed_physics =
        std::make_shared<NonRebalancingIndividualStep>();
    detailed_physics->setRuntimeDiagnostics(source_timing, {});
    detailed.addPhysics(detailed_physics);
    std::string const detailed_output = capture_step(detailed);
    if(rank == 0)
    {
        require(detailed_output.find(" source_first_s=0.010000") !=
                    std::string::npos &&
                    detailed_output.find(" source_second_s=0.020000") !=
                        std::string::npos,
                "detailed runtime source halves are missing");
        require_fields_in_order(
            detailed_output, "  source |",
            {" source_s=", " source_first_s=", " source_second_s=",
             " source_pct=", " source_calls="});
    }

    runtime_log.set("summary");
    runtime_color.set("always");
    Simulation colored(tess, cells, eos);
    colored.SetTimeStepFunction(std::make_shared<ManualTimeStep>(0.125));
    std::shared_ptr<NonRebalancingIndividualStep> const colored_physics =
        std::make_shared<NonRebalancingIndividualStep>();
    colored_physics->setRuntimeDiagnostics(SourceStepTiming(), {retry});
    colored.addPhysics(colored_physics);
    std::string const colored_output = capture_step(colored);
    if(rank == 0)
    {
        require(colored_output.find(
                    "\033[1;36mRICH_STEP\033[0m mode=global") !=
                    std::string::npos,
                "always color mode did not color the accepted header");
        require(colored_output.find("  \033[36mtime  \033[0m |") !=
                    std::string::npos &&
                    colored_output.find("  \033[34mwork  \033[0m |") !=
                        std::string::npos &&
                    colored_output.find("  \033[35mphases\033[0m |") !=
                        std::string::npos &&
                    colored_output.find("  \033[32msource\033[0m |") !=
                        std::string::npos,
                "always color mode did not color every accepted label");
        require(colored_output.find(
                    "\033[1;33mRICH_RETRY\033[0m mode=global") !=
                    std::string::npos &&
                    colored_output.find("  \033[33mattempt\033[0m |") !=
                        std::string::npos &&
                    colored_output.find("  \033[31mfailure\033[0m |") !=
                        std::string::npos,
                "always color mode did not color every retry label");
    }

    runtime_color.set("auto");
    no_color.set("1");
    Simulation uncolored(tess, cells, eos);
    uncolored.SetTimeStepFunction(std::make_shared<ManualTimeStep>(0.125));
    uncolored.addPhysics(std::make_shared<NonRebalancingIndividualStep>());
    std::string const uncolored_output = capture_step(uncolored);
    if(rank == 0)
        require(uncolored_output.find("\033[") == std::string::npos,
                "NO_COLOR did not disable automatic runtime colors");

    runtime_color.set("never");
    Simulation fatal(tess, cells, eos);
    fatal.SetTimeStepFunction(std::make_shared<ManualTimeStep>(0.125));
    std::shared_ptr<NonRebalancingIndividualStep> const fatal_physics =
        std::make_shared<NonRebalancingIndividualStep>();
    fatal_physics->setRuntimeDiagnostics(SourceStepTiming(), {retry});
    fatal_physics->throwAfterRuntimeDiagnostics(true);
    fatal.addPhysics(fatal_physics);
    std::ostringstream fatal_stream;
    std::streambuf* const original = std::cout.rdbuf(fatal_stream.rdbuf());
    bool fatal_caught = false;
    try
    {
        fatal.step();
    }
    catch(std::runtime_error const&)
    {
        fatal_caught = true;
    }
    catch(...)
    {
        std::cout.rdbuf(original);
        throw;
    }
    std::cout.rdbuf(original);
    require(fatal_caught, "fatal retry test did not throw after reporting");
    if(rank == 0)
    {
        std::string const fatal_output = fatal_stream.str();
        require(fatal_output.find("RICH_RETRY mode=global") !=
                    std::string::npos,
                "fatal physics step lost its immediate retry record");
        require(fatal_output.find("RICH_STEP mode=global") ==
                    std::string::npos,
                "fatal physics step emitted an accepted-step record");
        require_fields_in_order(
            fatal_output, "RICH_RETRY mode=global",
            {" cycle=", " physics=", " attempt="});
        require_fields_in_order(
            fatal_output, "  attempt |",
            {" active_cells=", " active_bins=", " attempted_dt_min=",
             " attempted_dt_max="});
        require_fields_in_order(
            fatal_output, "  failure |",
            {" retry_s=", " reason=", " cell="});
    }

    runtime_log.set("invalid-test-value");
    Simulation invalid(tess, cells, eos);
    bool invalid_caught = false;
    try
    {
        invalid.step();
    }
    catch(std::invalid_argument const&)
    {
        invalid_caught = true;
    }
    require(invalid_caught, "invalid RICH_RUNTIME_LOG was accepted");

    runtime_log.set("summary");
    runtime_color.set("invalid-test-value");
    Simulation invalid_color(tess, cells, eos);
    bool invalid_color_caught = false;
    try
    {
        invalid_color.step();
    }
    catch(std::invalid_argument const&)
    {
        invalid_color_caught = true;
    }
    require(invalid_color_caught, "invalid RICH_RUNTIME_COLOR was accepted");
    runtime_color.set("never");

#ifdef RICH_MPI
    if(rank_count > 1)
    {
        runtime_log.set(rank == 0 ? "summary" : "detailed");
        Simulation inconsistent(tess, cells, eos);
        bool inconsistent_caught = false;
        try
        {
            inconsistent.step();
        }
        catch(std::invalid_argument const&)
        {
            inconsistent_caught = true;
        }
        require(inconsistent_caught,
                "MPI-inconsistent RICH_RUNTIME_LOG was accepted");

        runtime_log.set("summary");
        runtime_color.set(rank == 0 ? "auto" : "never");
        Simulation inconsistent_color(tess, cells, eos);
        bool inconsistent_color_caught = false;
        try
        {
            inconsistent_color.step();
        }
        catch(std::invalid_argument const&)
        {
            inconsistent_color_caught = true;
        }
        require(inconsistent_color_caught,
                "MPI-inconsistent RICH_RUNTIME_COLOR was accepted");
    }
#endif
    runtime_log.set("summary");
    runtime_color.set("never");

    MeshBuildTiming failed_build_timing;
    try
    {
        MeshBuildTimer failed_build_timer(failed_build_timing);
        throw std::runtime_error("expected mesh-build failure");
    }
    catch(std::runtime_error const&)
    {}
    require(failed_build_timing.builds == 1 &&
                failed_build_timing.seconds >= 0,
            "failed mesh-build attempt was not recorded");

    std::vector<Conserved3D> remesh_extensives(cells.size());
    auto remesh_generator = [](Tessellation3D const& current_tess, double) {
        std::vector<Vector3D> const& mesh_points =
            current_tess.getMeshPoints();
        return std::vector<Vector3D>(
            mesh_points.begin(),
            mesh_points.begin() + current_tess.GetPointNo());
    };
    std::shared_ptr<RemeshStep> const remesh_step =
        std::make_shared<RemeshStep>(
            tess, cells, remesh_extensives, remesh_generator);
    Simulation remesh_simulation(tess, cells, eos);
    remesh_simulation.SetTimeStepFunction(
        std::make_shared<ManualTimeStep>(0.125));
    remesh_simulation.addPhysics(remesh_step);
    std::string const first_remesh_output = capture_step(remesh_simulation);
    MeshBuildTiming const first_remesh_timing =
        remesh_step->getMeshBuildTiming();
    requireCollectively(first_remesh_timing.builds == 1 &&
                            first_remesh_timing.seconds >= 0,
                        "RemeshStep did not record its tessellation build");

    std::string const second_remesh_output = capture_step(remesh_simulation);
    MeshBuildTiming const second_remesh_timing =
        remesh_step->getMeshBuildTiming();
    requireCollectively(second_remesh_timing.builds == 1 &&
                            second_remesh_timing.seconds >= 0,
                        "RemeshStep mesh diagnostics accumulated across steps");

#ifdef RICH_MPI
    Simulation balance_simulation(tess, cells, eos);
    balance_simulation.SetTimeStepFunction(
        std::make_shared<ManualTimeStep>(0.125));
    std::shared_ptr<RebalanceLoggingStep> const balance_step =
        std::make_shared<RebalanceLoggingStep>(
            tess.GetPointNo(), "timed-new-balance");
    balance_simulation.addPhysics(balance_step);
    std::string const first_balance_output = capture_step(balance_simulation);

    balance_simulation.storeLoadBalance(
        "timed-stored-balance", tess.GetLoadBalancer());
    balance_step->setRequiredLoadBalance("timed-stored-balance");
    std::string const stored_balance_output = capture_step(balance_simulation);
#endif

    std::uint64_t const missing_counter =
        std::numeric_limits<std::uint64_t>::max();
    std::uint64_t const first_remesh_builds =
        runtimeUnsignedField(first_remesh_output, "mesh_builds");
    requireCollectively(
        rank != 0 || (first_remesh_builds != missing_counter &&
                      first_remesh_builds >= first_remesh_timing.builds),
        "global runtime output omitted the RemeshStep build; "
        "captured output:\n" + first_remesh_output);

    std::uint64_t const second_remesh_builds =
        runtimeUnsignedField(second_remesh_output, "mesh_builds");
    requireCollectively(
        rank != 0 || (second_remesh_builds != missing_counter &&
                      second_remesh_builds >= second_remesh_timing.builds),
        "second global runtime output omitted the RemeshStep build; "
        "captured output:\n" + second_remesh_output);

#ifdef RICH_MPI
    requireCollectively(
        rank != 0 ||
            runtimeUnsignedField(first_balance_output, "mesh_builds") == 1,
        "global runtime output omitted the first load-balancing "
        "build; captured output:\n" + first_balance_output);
    requireCollectively(
        rank != 0 ||
            runtimeUnsignedField(stored_balance_output, "mesh_builds") == 1,
        "global runtime output omitted the stored load-balancer "
        "build; captured output:\n" + stored_balance_output);
#endif
}

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

// restore() caps an over-cap checkpoint before any closure or commit: at tick
// 24 the ceiling is anchor bin 2 + K = 4 (the finest cell, bin 2, is on the
// last rank and sets the begin-aware ends); a bin-6
// cell that began at 0 is past a bin-4 allowance and ends at the next bin-2
// tick, 28; one that began at 16 is within it and ends at 32.  Runs in the
// K = 2 invocation only.
void testRestoreBinSpreadCap()
{
    if(IndividualTimeStepScheduler::maximumBinSpread() != 2)
        return;
    int rank = 0;
    int rank_count = 1;
#ifdef RICH_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &rank_count);
#endif
    std::vector<ComputationalCell3D> cells(6);
    for(std::size_t i = 0; i < cells.size(); ++i)
        cells[i].ID = 3100 + 100 * static_cast<std::size_t>(rank) + i;
    std::vector<CellTimeState> states(cells.size());
    for(std::size_t i = 0; i < cells.size(); ++i)
    {
        bool const finest = rank == rank_count - 1 && i == 0;
        states[i].cell_id = cells[i].ID;
        states[i].begin_tick = finest ? 24 : (i % 2 == 0 ? 0 : 16);
        states[i].last_primitive_tick = states[i].begin_tick;
        states[i].end_tick = finest ? 28 : (i % 2 == 0 ? 64 : 80);
        states[i].time_bin = finest ? 2 : 6;
    }
    IndividualTimeStepOptions options;
    options.time_quantum = 1;
    options.initial_bin = 2;
    options.maximum_bin = 8;
    IndividualTimeStepScheduler scheduler(options);
    scheduler.restore(cells, 0, 1, 24, states);
    int bad = scheduler.minimumOccupiedBin() == 2 ? 0 : 1;
    for(CellTimeState const& state : scheduler.states())
    {
        if(state.time_bin == 2)
            continue;
        bad = bad || state.time_bin != 4 || state.end_tick != (state.begin_tick == 0 ? 28u : 32u);
    }
    requireCollectively(bad == 0, "restore did not cap an over-cap checkpoint before its first event");
    // Every cell above the ceiling: all take bin 4 and the cached minimum
    // follows them down to 4 (not the stored 6); the begin-0 cells are overdue
    // and end at the next bin-4 tick, 32, as do the begin-16 cells.
    for(std::size_t i = 0; i < cells.size(); ++i)
    {
        states[i].begin_tick = i % 2 == 0 ? 0 : 16;
        states[i].last_primitive_tick = states[i].begin_tick;
        states[i].end_tick = i % 2 == 0 ? 64 : 80;
        states[i].time_bin = 6;
    }
    IndividualTimeStepScheduler above(options);
    above.restore(cells, 0, 1, 24, states);
    bad = above.minimumOccupiedBin() == 4 ? 0 : 1;
    for(CellTimeState const& state : above.states())
        bad = bad || state.time_bin != 4 || state.end_tick != 32u;
    requireCollectively(bad == 0, "restore of an all-above-ceiling checkpoint left a stale minimum bin");
    // AMR after that restore: a child of a capped cell inherits the capped bin and end.
    std::vector<ComputationalCell3D> refined = cells;
    refined.push_back(cells[1]);
    refined.back().ID = 3900 + 100 * static_cast<std::size_t>(rank);
    IndividualAMRChangeSet changes;
    changes.child_parent_ids.emplace_back(refined.back().ID, cells[1].ID);
    above.applyAMRChangeSet(refined, changes);
    CellTimeState const& child = above.states().back();
    requireCollectively(child.cell_id == refined.back().ID && child.time_bin == 4 && child.end_tick == 32u &&
        above.minimumOccupiedBin() == 4, "a child after a capped restore did not inherit the capped bin and end");
}

} // namespace

// Friend of Simulation (so outside the anonymous namespace): drives its
// adaptive controller with synthetic step walls and advances, on a Simulation
// in individual mode.
struct AdaptiveControllerTestAccess
{
    static Simulation::AdaptiveModeState& state(Simulation& simulation) { return simulation.adaptiveMode; }
    static void step(Simulation& simulation, double seconds, double advance)
    {
        simulation.lastStepSecondsMax = seconds;
        simulation.lastStepAdvance = advance;
        simulation.adaptiveAfterStep(simulation.GetTimeIntegrationMode());
    }
};

namespace {

// The controller's dwell backoff (job 10222326: a multiplier of 16 held a slow
// individual phase for 1036 events).  A domain change, in a dwell or a probe,
// restarts the samples at multiplier 1 and keeps a probe's spent wall and
// budget; the dwell also ends after RICH_ADAPTIVE_DWELL_WALL_MAX seconds (read
// here as the controller reads it; 0 disables) but never before the ramp and
// sample gates; a stale-baseline revert requests the switch at multiplier 1.
// Assumes the other RICH_ADAPTIVE_* settings are unset (ramp 12 events, 6
// samples, dwell 64 steps).
void testAdaptiveControllerBackoff()
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
    char const* const wall_setting = std::getenv("RICH_ADAPTIVE_DWELL_WALL_MAX");
    double const wall_limit = wall_setting != nullptr && wall_setting[0] != '\0' ? std::atof(wall_setting) : 1800;
    std::size_t const gate = 12 + 6;
    using Access = AdaptiveControllerTestAccess;
    auto const fresh = [&](Voronoi3D& tess)
    {
        tess.Build(points);
        auto simulation = std::make_unique<Simulation>(tess, cells, eos);
        IndividualTimeStepOptions options;
        simulation->EnableIndividualTimeSteps(options);
        simulation->SetAdaptiveIntegrationMode(true, options);
        return simulation;
    };
    {
        // Domain change during a backed-off dwell, then the dwell at multiplier 1.
        Voronoi3D tess(Vector3D(0, 0, 0), Vector3D(1, 1, 1));
        std::unique_ptr<Simulation> simulation = fresh(tess);
        // 12 + 64 steps stay within half of any wall limit, so only the step threshold ends the dwell.
        double const seconds = wall_limit > 0 ? wall_limit / (2 * (12 + 64)) : 10;
        Access::state(*simulation).dwellMultiplier = 16;
        for(std::size_t i = 0; i < gate + 2; ++i)
            Access::step(*simulation, seconds, 1e-3);
        simulation->NotifyDomainChanged();
        Simulation::AdaptiveModeState const& a = Access::state(*simulation);
        require(a.dwellMultiplier == 1 && a.stepsInMode == 0 && a.wallTotalInMode == 0 && a.wallMeasured == 0 &&
            !a.probing, "a domain change during a dwell kept its backoff or samples");
        for(std::size_t i = 0; i + 1 < 12 + 64; ++i)
            Access::step(*simulation, seconds, 1e-3);
        require(!a.probing, "the dwell after a domain change ended early");
        Access::step(*simulation, seconds, 1e-3);
        require(a.probing && a.switchToGlobalRequested, "the dwell after a domain change did not end at multiplier 1");
    }
    {
        // Domain change during a probe: its spent wall and budget stay.
        Voronoi3D tess(Vector3D(0, 0, 0), Vector3D(1, 1, 1));
        std::unique_ptr<Simulation> simulation = fresh(tess);
        Simulation::AdaptiveModeState& a = Access::state(*simulation);
        a.dwellMultiplier = 16;
        a.probing = true;
        a.probeWallBudget = 500;
        for(int i = 0; i < 5; ++i)
            Access::step(*simulation, 10, 1e-3);
        simulation->NotifyDomainChanged();
        require(a.dwellMultiplier == 1 && a.stepsInMode == 0 && a.wallMeasured == 0 && a.probing &&
            a.wallTotalInMode == 50 && a.probeWallBudget == 500, "a domain change during a probe reset it wrongly");
    }
    {
        // A backed-off dwell at 100 s per event: with the wall limit it ends at
        // the first event past both the gates and the limit; without it only
        // after 12 + 64 x 16 events.
        Voronoi3D tess(Vector3D(0, 0, 0), Vector3D(1, 1, 1));
        std::unique_ptr<Simulation> simulation = fresh(tess);
        Simulation::AdaptiveModeState const& a = Access::state(*simulation);
        Access::state(*simulation).dwellMultiplier = 16;
        std::size_t const by_steps = 12 + 64 * 16;
        std::size_t const expected = wall_limit > 0 ?
            std::min(by_steps, std::max(gate + 1, static_cast<std::size_t>(std::ceil(wall_limit / 100)))) : by_steps;
        for(std::size_t i = 0; i + 1 < expected; ++i)
            Access::step(*simulation, 100, 1e-3);
        require(!a.probing, "a backed-off dwell ended before its gates or wall limit");
        Access::step(*simulation, 100, 1e-3);
        require(a.probing && a.switchToGlobalRequested, "a backed-off dwell did not end at its wall limit or step count");
    }
    {
        // A probe whose global baseline predates a domain change reverts to
        // global at multiplier 1.
        Voronoi3D tess(Vector3D(0, 0, 0), Vector3D(1, 1, 1));
        std::unique_ptr<Simulation> simulation = fresh(tess);
        Simulation::AdaptiveModeState& a = Access::state(*simulation);
        a.dwellMultiplier = 16;
        a.probing = true;
        a.probeWallBudget = 0;
        a.tauGlobal = 1;
        a.tauGlobalEpoch = 0;
        a.domainEpoch = 1;
        for(std::size_t i = 0; i < gate; ++i)
            Access::step(*simulation, 10, 1e-3);
        require(!a.probing && a.switchToGlobalRequested && a.dwellMultiplier == 1,
            "a stale-baseline revert did not request global at multiplier 1");
    }
}

// A global step records the anchor reference; an individual phase with an
// explicit quantum must keep that quantum instead of conflicting with it.
// First-interval generator velocities after a global -> individual switch
// (setInitialPointVelocities): assigned by stable ID on a fresh timeline,
// unknown IDs ignored, rejected before initialization.
void testInitialPointVelocities()
{
    std::vector<ComputationalCell3D> cells(3);
    for(std::size_t i = 0; i < cells.size(); ++i)
        cells[i].ID = 50 + i;
    IndividualTimeStepOptions options;
    IndividualTimeStepScheduler scheduler(options);
    bool rejected = false;
    try
    {
        scheduler.setInitialPointVelocities({{50, Vector3D(1, 0, 0)}});
    }
    catch(std::logic_error const&)
    {
        rejected = true;
    }
    require(rejected, "initial point velocities were accepted before initialization");
    scheduler.initialize(cells, 0, 0.125, 0);
    scheduler.setInitialPointVelocities({{52, Vector3D(0, 2, 0)}, {50, Vector3D(1, 0, 0)}, {99, Vector3D(5, 5, 5)}});
    std::vector<CellTimeState> const& states = scheduler.states();
    require(states[0].point_velocity.x == 1 && states[0].point_velocity.y == 0, "cell 50 lost its velocity");
    require(states[1].point_velocity.x == 0 && states[1].point_velocity.y == 0, "cell 51 received a velocity");
    require(states[2].point_velocity.y == 2 && states[2].point_velocity.x == 0, "cell 52 lost its velocity");

    // Explicit quantum: the first interval never exceeds the initial step, and
    // a step below one quantum is rejected.
    IndividualTimeStepOptions explicit_options;
    explicit_options.time_quantum = 0.001;
    explicit_options.initial_bin = 2;
    IndividualTimeStepScheduler bounded(explicit_options);
    bounded.initialize(cells, 0, 0.0025, 0);
    require(bounded.nextEventTimeStep() <= 0.0025 && bounded.nextEventTimeStep() == 0.002,
        "an explicit-quantum first interval exceeded the initial step");
    IndividualTimeStepScheduler below(explicit_options);
    bool below_rejected = false;
    try
    {
        below.initialize(cells, 0, 0.0005, 0);
    }
    catch(std::exception const&)
    {
        below_rejected = true;
    }
    require(below_rejected, "an initial step below one explicit quantum was accepted");
}

void testExplicitQuantumAfterGlobalStep()
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
    simulation.step();
    IndividualTimeStepOptions options;
    options.initial_bin = 0;
    options.time_quantum = 0.0625;
    simulation.EnableIndividualTimeSteps(options);
    simulation.step();
    IndividualTimeStepScheduler const* scheduler = simulation.GetIndividualTimeStepScheduler();
    require(scheduler != nullptr && scheduler->initialized() && scheduler->timeQuantum() == 0.0625,
        "an explicit individual quantum did not survive a preceding global step");
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

// Localisation aid: every rank announces the case it is about to enter,
// flushed, so an MPI abort names the case instead of leaving the last
// diagnostic line to be guessed.
void testMarker(char const* const name)
{
    int marker_rank = 0;
#ifdef RICH_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &marker_rank);
#endif
    std::cerr << "TEST_MARKER rank=" << marker_rank << " case=" << name
              << std::endl;
}
#define TEST_MARKER(name) testMarker(name)

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

// Fake individual radiation driver for the entry-probe controller tests: it
// records every attempted interval fraction, rejects any candidate longer than
// accept_limit.
class EntryProbeIndividualRadiationDriver final : public RadiationDriver
{
public:
    EntryProbeIndividualRadiationDriver(EquationOfState const& eos, double const accept_limit) :
        RadiationDriver(eos, std::vector<std::string>(), false, false, false),
        accept_limit_(accept_limit)
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
        attempted.push_back(interval_fraction);
        if(interval_fraction > accept_limit_ + 1e-12) {
            ++rejections;
            setStepFailure("forced entry probe rejection", cells.at(0).ID);
            return false;
        }
        ++acceptances;
        accepted_fraction += interval_fraction;
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

    mutable std::vector<double> attempted;
    mutable std::size_t rejections = 0;
    mutable std::size_t acceptances = 0;
    mutable double accepted_fraction = 0;

private:
    double const accept_limit_;
};

// RICH_INDIVIDUAL_RADIATION_ENTRY_PROBE.  The suite is run twice, once with the
// flag off and once with it on (the flag is cached statically, so the two cases
// need separate processes); every case below states both expectations.
void testIndividualRadiationEntryProbeController()
{
    char const* const flag_value =
        std::getenv("RICH_INDIVIDUAL_RADIATION_ENTRY_PROBE");
    bool const entry_probe =
        flag_value != nullptr && std::string(flag_value) == "1";

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
            "entry-probe test requires two owned cells per MPI rank");
    std::vector<ComputationalCell3D> cells(tess.GetPointNo());
    for(std::size_t cell = 0; cell < cells.size(); ++cell)
        cells[cell].ID = 901 + static_cast<std::size_t>(rank) * 100000 + cell;
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

    auto seeded = [](double const ceiling, std::uint64_t const accepted,
                     std::uint64_t const required)
    {
        IndividualRadiationDefectAccounting accounting;
        // Persisted retry history: this is what makes the controller restore
        // its ceiling and backoff instead of starting fresh.
        accounting.defect_rejections = 1;
        accounting.cooldown_fraction_ceiling = ceiling;
        accounting.cooldown_accepted_candidates = accepted;
        accounting.cooldown_required_candidates = required;
        return accounting;
    };

    struct EventResult
    {
        std::vector<double> attempted;
        std::size_t rejections = 0;
        std::size_t acceptances = 0;
        double accepted_fraction = 0;
    };

    auto run_event = [&](IndividualRadiationDefectAccounting& accounting,
                         double const accept_limit,
                         std::size_t const active_cells)
    {
        EntryProbeIndividualRadiationDriver driver(eos, accept_limit);
        RadiationStep step(tess, cells, extensives, tracker,
#ifdef RICH_MPI
            nullptr,
#endif
            driver, false);
        IndividualStepContext local = context;
        local.radiation_defect_accounting = &accounting;
        // Keep the mask and the index list describing the same active set; a
        // shorter index list alone is not a valid context.
        local.active_indices.resize(std::min(active_cells, cells.size()));
        local.active_mask.assign(cells.size(), 0);
        for(std::size_t active : local.active_indices)
            local.active_mask.at(active) = 1;
        step.stepIndividual(local);
        EventResult result;
        result.attempted = driver.attempted;
        result.rejections = driver.rejections;
        result.acceptances = driver.acceptances;
        result.accepted_fraction = driver.accepted_fraction;
        return result;
    };

    // 1. Cross-event recovery out of the 1/2 trap: an earned entry probe tests
    //    the full interval as the event's first candidate and, once accepted,
    //    commits the recovered ceiling.  Without the flag the two accepted
    //    halves fill the event exactly, no probe is ever reachable, and the
    //    ceiling stays at 1/2 forever -- the defect this feature fixes.
    {
        TEST_MARKER("entry_probe/1-cross-event-recovery");
        IndividualRadiationDefectAccounting accounting = seeded(0.5, 8, 8);
        EventResult const result = run_event(accounting, 1.0, cells.size());
        require(!result.attempted.empty(), "entry-probe event attempted nothing");
        require(close(result.attempted.front(), entry_probe ? 1.0 : 0.5, 2e-15),
                "entry probe did not schedule the recovery interval first");
        require(result.acceptances == (entry_probe ? 1u : 2u),
                "entry probe did not change the number of radiation solves");
        require(result.rejections == 0, "recovery probe was rejected");
        require(close(result.accepted_fraction, 1.0, 2e-15),
                "accepted fractions do not sum to one event");
        require(close(accounting.cooldown_fraction_ceiling,
                      entry_probe ? 1.0 : 0.5, 2e-15),
                "accepted recovery probe did not commit the ceiling");
        if(entry_probe)
            require(accounting.cooldown_required_candidates == 8,
                    "accepted probe did not reset the cooldown");
    }

    // 2. Progressive recovery 1/4 -> 1/2: the probe doubles, it is not a jump
    //    to the full interval, and the committed ceiling follows the interval
    //    that was actually accepted.
    {
        TEST_MARKER("entry_probe/2-progressive-recovery");
        IndividualRadiationDefectAccounting accounting = seeded(0.25, 8, 8);
        EventResult const result = run_event(accounting, 1.0, cells.size());
        require(close(result.attempted.front(), entry_probe ? 0.5 : 0.25, 2e-15),
                "entry probe did not double the persisted ceiling");
        require(close(result.accepted_fraction, 1.0, 2e-15),
                "accepted fractions do not sum to one event");
        if(entry_probe)
            require(close(accounting.cooldown_fraction_ceiling, 0.5, 2e-15),
                    "progressive recovery committed the wrong ceiling");
    }

    // 3. Unearned recovery: the measured-success threshold is not met, so the
    //    controller must not probe in either mode.
    {
        TEST_MARKER("entry_probe/3-unearned-probe");
        IndividualRadiationDefectAccounting accounting = seeded(0.5, 7, 8);
        EventResult const result = run_event(accounting, 1.0, cells.size());
        require(close(result.attempted.front(), 0.5, 2e-15),
                "unearned entry probe fired");
        require(result.acceptances == 2,
                "unearned entry probe changed the candidate count");
        require(close(accounting.cooldown_fraction_ceiling, 0.5, 2e-15),
                "unearned entry probe moved the ceiling");
    }

    // 4. Persistent genuine failure: the probe is rejected, the event still
    //    completes at the safe interval, the ceiling is restored rather than
    //    ratcheted up, and the probe cooldown doubles.
    {
        TEST_MARKER("entry_probe/4-genuine-failure");
        IndividualRadiationDefectAccounting accounting = seeded(0.5, 8, 8);
        EventResult const result = run_event(accounting, 0.5, cells.size());
        require(result.rejections == (entry_probe ? 1u : 0u),
                "rejected entry probe accounting is wrong");
        require(close(result.accepted_fraction, 1.0, 2e-15),
                "accepted fractions do not sum to one event after a rejection");
        require(close(accounting.cooldown_fraction_ceiling, 0.5, 2e-15),
                "a rejected probe moved the persisted ceiling");
        require(accounting.cooldown_required_candidates ==
                    (entry_probe ? 16u : 8u),
                "a rejected probe did not double the cooldown");
    }

#ifdef RICH_MPI
    auto agreed = [](double const value, char const* const message)
    {
        double extrema[2] = {value, -value};
        MPI_Allreduce(MPI_IN_PLACE, extrema, 2, MPI_DOUBLE, MPI_MIN,
                      MPI_COMM_WORLD);
        require(extrema[0] == -extrema[1], message);
    };

    // 6a. Flag-off regression, common persisted state.  Legacy mode keeps the
    //     restored controller rank-local, so heterogeneous state is not a
    //     legal input there: ranks would split over different candidate counts
    //     and meet in different collectives.  With common state the legacy
    //     controller must still agree, and the 1/2 trap must still be closed
    //     (that is the defect the flag exists to fix, not a regression).
    if(!entry_probe) {
        TEST_MARKER("entry_probe/6a-legacy-common-state");
        IndividualRadiationDefectAccounting accounting = seeded(0.5, 8, 8);
        EventResult const result = run_event(accounting, 1.0, cells.size());
        agreed(result.attempted.front(),
               "legacy ranks attempted different first radiation fractions");
        agreed(static_cast<double>(result.attempted.size()),
               "legacy ranks attempted different numbers of radiation candidates");
        require(close(result.attempted.front(), 0.5, 2e-15),
                "legacy mode probed without the flag");
        require(close(accounting.cooldown_fraction_ceiling, 0.5, 2e-15),
                "legacy mode escaped the persisted ceiling");
    }

    // 6b. Heterogeneous persisted state, flag on.  Rank 0 restores a
    //     restricted controller; the other ranks come up genuinely fresh --
    //     no retry history at all, which is what a rank that never rejected
    //     looks like after a restart or a migration.  Only the unconditional
    //     restore/persist gates plus the reduction make those ranks agree, so
    //     reverting either gate fails here.  The accounting is then carried
    //     from event to event without reseeding: the measured-success counter
    //     must climb by exactly the number of accepted candidates until the
    //     unified controller earns its probe, which every rank must then take
    //     together.  Recovery is deferred, never vetoed.
    if(entry_probe) {
        TEST_MARKER("entry_probe/6b-fresh-rank-recovery");
        IndividualRadiationDefectAccounting accounting =
            rank == 0 ? seeded(0.5, 8, 8) :
            IndividualRadiationDefectAccounting();
        require(rank == 0 || (accounting.cooldown_accepted_candidates == 0 &&
                              accounting.defect_rejections == 0 &&
                              close(accounting.cooldown_fraction_ceiling, 1.0,
                                    2e-15)),
                "non-zero ranks were not started with fresh accounting");
        bool probed = false;
        std::size_t events = 0;
        std::uint64_t successes = 0;
        // The unified controller starts at min(successes) = 0 and needs one
        // cooldown worth of accepted candidates; two halves per event puts the
        // bound at a handful of events, so this terminates well inside it.
        for(; events < 16 && !probed; ++events) {
            std::uint64_t const previous_successes =
                accounting.cooldown_accepted_candidates;
            EventResult const result = run_event(accounting, 1.0, cells.size());
            agreed(result.attempted.front(),
                   "ranks attempted different first radiation fractions");
            agreed(static_cast<double>(result.attempted.size()),
                   "ranks attempted different numbers of radiation candidates");
            agreed(static_cast<double>(result.acceptances),
                   "ranks accepted different numbers of radiation candidates");
            agreed(accounting.cooldown_fraction_ceiling,
                   "ranks persisted different retry ceilings");
            agreed(static_cast<double>(
                       accounting.cooldown_accepted_candidates),
                   "ranks persisted different measured-success counts");
            agreed(static_cast<double>(
                       accounting.cooldown_required_candidates),
                   "ranks persisted different probe cooldowns");
            require(close(result.accepted_fraction, 1.0, 2e-15),
                   "accepted fractions do not sum to one event");
            probed = result.attempted.front() > 0.5 + 2e-15;
            if(!probed) {
                require(close(result.attempted.front(), 0.5, 2e-15),
                        "unequal restored state did not fall back to the "
                        "restrictive ceiling");
                // The first event is where the heterogeneous restored state is
                // reduced away, so rank 0's seeded counter legitimately drops
                // to the common minimum; from the second event on the state is
                // already unified and the counter must track the acceptances
                // exactly.
                require(events == 0 ||
                            accounting.cooldown_accepted_candidates ==
                                previous_successes + result.acceptances,
                        "measured-success counter did not follow the accepted "
                        "candidates");
                require(close(accounting.cooldown_fraction_ceiling, 0.5, 2e-15),
                        "a non-probing event moved the persisted ceiling");
                successes = accounting.cooldown_accepted_candidates;
            }
        }
        require(probed, "unified state never earned its recovery probe");
        require(successes > 0,
                "the measured-success counter never advanced before the probe");
        require(close(accounting.cooldown_fraction_ceiling, 1.0, 2e-15),
                "the accepted recovery probe did not commit the full interval");
        require(accounting.cooldown_required_candidates == 8,
                "the accepted recovery probe did not reset the cooldown");
    }
#endif
}

// Real grey diffusion driver that records the interval of every accepted
// candidate, for the entry-probe solver comparison below.
class RecordingGreyDiffusion final : public Diffusion
{
public:
    RecordingGreyDiffusion(OpacityCalculator const& opacity,
                           DiffusionBoundaryCalculator const& boundary,
                           EquationOfState const& eos) :
        Diffusion(opacity, boundary, eos, std::vector<std::string>(),
                  false, true, false, false)
    {}

    mutable std::vector<double> accepted_intervals;

    void PostCG(Tessellation3D const& tess,
                std::vector<Conserved3D>& extensives,
                double dt,
                std::vector<ComputationalCell3D>& cells,
                std::vector<double> const& result,
                std::vector<double> const& full_result) const override
    {
        Diffusion::PostCG(tess, extensives, dt, cells, result, full_result);
        accepted_intervals.push_back(dt);
    }
};

// RICH_INDIVIDUAL_RADIATION_ENTRY_PROBE against the real grey solver.  The
// controller cases above use a fake driver, so they prove the scheduling but
// not that an earned entry probe actually produces the reference solve.  Here
// one event runs twice over identical initial state: once with a fresh
// controller (the reference, a single full-interval solve) and once with a
// persisted controller pinned at a ceiling of 1/2 whose probe is earned.  With
// the flag on the probe must recover the reference solve exactly -- same
// accepted interval, same cells, same extensives, same energy-defect
// accounting -- and commit the recovered ceiling.  With the flag off the same
// persisted state must still split the event into the two halves that the
// 1/2 trap forces, which is the behaviour the flag is there to escape.
void testIndividualRadiationEntryProbeRealSolver()
{
    char const* const flag_value =
        std::getenv("RICH_INDIVIDUAL_RADIATION_ENTRY_PROBE");
    bool const entry_probe =
        flag_value != nullptr && std::string(flag_value) == "1";

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
        cell.ID = (std::size_t{1} << 32) + 9501 +
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

    auto make_context = [&](IndividualRadiationDefectAccounting& accounting)
    {
        IndividualStepContext context;
        context.previous_event_tick = 0;
        context.event_tick = 2;
        context.previous_event_time = 0.0;
        context.event_time = 1.0;
        context.time_quantum = 0.5;
        context.mesh_build_policy = IndividualMeshBuildPolicy::FullReference;
        context.active_indices.resize(initial_cells.size());
        std::iota(context.active_indices.begin(), context.active_indices.end(), 0);
        context.active_mask.assign(initial_cells.size(), 1);
        context.cell_time_steps.assign(initial_cells.size(), 1.0);
        context.primitive_ticks.assign(initial_cells.size(), 0);
        context.radiation_defect_accounting = &accounting;
        return context;
    };

    PowerLawOpacity opacity(1e-12, 0.0, 0.0, 1e-12, 0.0, 0.0);
    DiffusionClosedBox boundary;

    // Reference: an unrestricted controller solves the event in one go.
    std::vector<ComputationalCell3D> reference_cells = initial_cells;
    std::vector<Conserved3D> reference_extensives = initial_extensives;
    ProgressTracker reference_tracker;
    RecordingGreyDiffusion reference_driver(opacity, boundary, eos);
    IndividualRadiationDefectAccounting reference_accounting;
    RadiationStep reference_step(tess, reference_cells, reference_extensives,
        reference_tracker,
#ifdef RICH_MPI
        nullptr,
#endif
        reference_driver, false);
    reference_step.stepIndividual(make_context(reference_accounting));
    require(reference_driver.accepted_intervals.size() == 1 &&
            close(reference_driver.accepted_intervals.front(), 1.0, 2e-15),
            "the unrestricted reference did not solve the event in one candidate");
    require(reference_accounting.defect_rejections == 0,
            "the unrestricted reference rejected a candidate");

    // Probe run: the controller is pinned at a ceiling of 1/2 with its
    // measured-success quota already met, so the probe is earned.
    std::vector<ComputationalCell3D> probe_cells = initial_cells;
    std::vector<Conserved3D> probe_extensives = initial_extensives;
    ProgressTracker probe_tracker;
    RecordingGreyDiffusion probe_driver(opacity, boundary, eos);
    IndividualRadiationDefectAccounting probe_accounting;
    probe_accounting.defect_rejections = 1;
    probe_accounting.cooldown_fraction_ceiling = 0.5;
    probe_accounting.cooldown_accepted_candidates = 8;
    probe_accounting.cooldown_required_candidates = 8;
    RadiationStep probe_step(tess, probe_cells, probe_extensives, probe_tracker,
#ifdef RICH_MPI
        nullptr,
#endif
        probe_driver, false);
    probe_step.stepIndividual(make_context(probe_accounting));

    if(entry_probe) {
        require(probe_driver.accepted_intervals.size() == 1 &&
                close(probe_driver.accepted_intervals.front(), 1.0, 2e-15),
                "the earned entry probe did not solve the full interval");
        require(close(probe_accounting.cooldown_fraction_ceiling, 1.0, 2e-15),
                "the accepted entry probe did not commit the recovered ceiling");
        require(probe_accounting.defect_rejections == 1,
                "the entry probe was rejected by the real solver");
        // Same interval from the same state: the probe must reproduce the
        // reference solve, state and energy-defect accounting alike.
        for(std::size_t i = 0; i < probe_cells.size(); ++i)
            require(close(probe_cells[i].Erad, reference_cells[i].Erad, 2e-12) &&
                    close(probe_cells[i].temperature,
                          reference_cells[i].temperature, 2e-12) &&
                    close(probe_cells[i].internal_energy,
                          reference_cells[i].internal_energy, 2e-12) &&
                    close(probe_extensives[i].Erad,
                          reference_extensives[i].Erad, 2e-12) &&
                    close(probe_extensives[i].energy,
                          reference_extensives[i].energy, 2e-12),
                    "the recovered entry probe did not reproduce the reference solve");
        require(close(static_cast<double>(
                          probe_accounting.cumulative_signed_extent),
                      static_cast<double>(
                          reference_accounting.cumulative_signed_extent),
                      2e-12) &&
                close(static_cast<double>(
                          probe_accounting.cumulative_absolute_extent),
                      static_cast<double>(
                          reference_accounting.cumulative_absolute_extent),
                      2e-12),
                "the recovered entry probe did not reproduce the reference "
                "energy-defect accounting");
        require(probe_accounting.accepted_dirichlet_candidates ==
                    reference_accounting.accepted_dirichlet_candidates,
                "the recovered entry probe accepted a different number of "
                "candidates than the reference");
    }
    else {
        require(probe_driver.accepted_intervals.size() == 2,
                "legacy mode did not split the event at the persisted ceiling");
        for(double const interval : probe_driver.accepted_intervals)
            require(close(interval, 0.5, 2e-15),
                    "legacy mode used an interval other than the persisted half");
        require(close(probe_accounting.cooldown_fraction_ceiling, 0.5, 2e-15),
                "legacy mode escaped the persisted ceiling without the flag");
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
    // Most scheduler cases here were written for uncapped bins on the scheduler's own grid; run them with
    // the pre-2026-09-27 settings unless an invocation chooses (the default K = 2 runs as its own
    // invocation, RICH_INDIVIDUAL_MAX_BIN_SPREAD=2, where testBinSpreadCap is active).
    setenv("RICH_INDIVIDUAL_MAX_BIN_SPREAD", "-1", 0);
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
        char const* const active_hilbert_compatibility_test =
            std::getenv("RICH_TEST_ACTIVE_HILBERT_COMPATIBILITY");
        if(active_hilbert_compatibility_test != nullptr)
            testActiveHilbertSelectorCompatibility();
        else
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
            testActiveTimeBinMask();
            testHydroFaceIntervalContract();
			testRuntimeStepStdout();
			testIndividualThermodynamicSlopeCoupling();
				testIndividualNegativeMassDiagnostic();
				testHydroMaterialStateValidation();
#ifdef RICH_MPI
				testRestoreMinimumOccupiedBin();
				testChangeWakeFinalization();
				testAnchoredInitialization();
				testChangeWakeSpacingReference();
				testBinSpreadCap();
				testRestoreBinSpreadCap();
				testRemoteNeighborBinClosureMPI();
				testClosureReexpandPromotionMPI();
				testCompetingRemoteBinRequestsMPI();
				testHydroWakeFaceIntervalsMPI();
				testHydroRadiationSharedDonorMPI();
				testHydroMaterialUpdateMPI();
				testConditionExtensiveCollectiveFailureMPI();
				testIndividualBoundaryGhostPredictionMPI();
            testIndividualCellUpdateOnSparsePartialMeshMPI();
            testPermutationAwareAllActiveMapping();
            testDistributedActiveWideIdExchangeMPI();
            testDistributedActiveExchangeFailuresMPI();
            testDistributedActiveLargeCountHelpersMPI();
            testCanonicalGlobalMaximumNegativePolicyMPI();
            TEST_MARKER("retry_limiter_scope");
            testIndividualRadiationRetryLimiterScope();
            TEST_MARKER("entry_probe_controller");
            testIndividualRadiationEntryProbeController();
            TEST_MARKER("entry_probe_real_solver");
            testIndividualRadiationEntryProbeRealSolver();
            TEST_MARKER("grey_fractional_candidate_refresh");
            testGreyFractionalCandidateRefresh();
            TEST_MARKER("all_done");
#else
            testSchedulerAndAMR();
            testMixedDirichletDefectAcceptance();
            testSynchronizedScheduler();
            testRestoreMinimumOccupiedBin();
            testChangeWakeFinalization();
            testAnchoredInitialization();
            testChangeWakeSpacingReference();
            testBinSpreadCap();
            testRestoreBinSpreadCap();
            testInactiveWakeAlignment();
            testHydroWakeFaceIntervals();
            testForcedAllActiveEventOverlay();
			testDefaultCellUpdaterAllActiveCommit();
			testHydroRadiationPositivityLimiter();
			testRepairOrderingAroundComptonSubsteps();
            testPartialGeometry();
            testSimulationSynchronizedEventLifecycle();
            testExplicitQuantumAfterGlobalStep();
            testInitialPointVelocities();
            testAdaptiveControllerBackoff();
            testMonteCarloRejection();
            testGlobalRadiationFractionalRetry();
            testGlobalRadiationPersistentFractionalRetryBackoff();
            testIndividualRadiationRetryLimiterScope();
            testIndividualRadiationEntryProbeController();
            testIndividualRadiationEntryProbeRealSolver();
            testGreyFractionalCandidateRefresh();
#endif
        }
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
