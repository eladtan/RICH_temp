#include <limits>
#include <cmath>
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <filesystem>

#include "ComptonTableReader.hpp"
#include "source/Radiation/CMMC/src/units/units.hpp"
#include "source/Radiation/CMMC/src/planck_integral/planck_integral.hpp"

#include <boost/math/special_functions/pow.hpp>

namespace {

Vector read_column_file(std::string const& path) {
    std::ifstream ifs(path);
    if (!ifs.is_open()) {
        std::cerr << "ComptonTableReader: cannot open file " << path << std::endl;
        std::exit(1);
    }
    Vector vals;
    std::string line;
    while (std::getline(ifs, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream iss(line);
        double v;
        while (iss >> v) vals.push_back(v);
    }
    return vals;
}

Matrix read_matrix_file(std::string const& path, std::size_t expected_rows) {
    std::ifstream ifs(path);
    if (!ifs.is_open()) {
        std::cerr << "ComptonTableReader: cannot open file " << path << std::endl;
        std::exit(1);
    }
    Matrix mat(expected_rows, Vector(expected_rows, 0.0));
    std::string line;
    std::size_t row = 0;
    while (std::getline(ifs, line)) {
        if (line.empty() || line[0] == '#') continue;
        if (row >= expected_rows) {
            std::cerr << "ComptonTableReader: too many rows in " << path << std::endl;
            std::exit(1);
        }
        std::istringstream iss(line);
        for (std::size_t col = 0; col < expected_rows; ++col) {
            if (!(iss >> mat[row][col])) {
                std::cerr << "ComptonTableReader: parse error in " << path
                          << " row " << row << " col " << col << std::endl;
                std::exit(1);
            }
        }
        ++row;
    }
    if (row != expected_rows) {
        std::cerr << "ComptonTableReader: expected " << expected_rows
                  << " rows but got " << row << " in " << path << std::endl;
        std::exit(1);
    }
    return mat;
}

} // anonymous namespace

ComptonTableReader::ComptonTableReader(
    Vector const& energy_groups_centers_,
    Vector const& energy_groups_boundaries_,
    bool force_detailed_balance_)
    : energy_groups_centers(energy_groups_centers_),
      energy_groups_boundaries(energy_groups_boundaries_),
      num_energy_groups(energy_groups_centers_.size()),
      force_detailed_balance(force_detailed_balance_),
      n_eq(energy_groups_centers_.size(), 0.0),
      B(energy_groups_centers_.size(), 0.0)
{}

void ComptonTableReader::load_tables(std::string const& directory) {
    namespace fs = std::filesystem;

    if (!fs::is_directory(directory)) {
        std::cerr << "ComptonTableReader: table directory does not exist: "
                  << directory << std::endl;
        std::exit(1);
    }

    auto const energy_path = (fs::path(directory) / "energy_groups.txt").string();
    auto const temp_path   = (fs::path(directory) / "temperatures.txt").string();

    Vector table_boundaries = read_column_file(energy_path);
    validate_energy_grid(table_boundaries);

    temperature_grid = read_column_file(temp_path);
    if (temperature_grid.size() < 2) {
        std::cerr << "ComptonTableReader: temperature grid has fewer than 2 entries" << std::endl;
        std::exit(1);
    }
    for (std::size_t i = 1; i < temperature_grid.size(); ++i) {
        if (temperature_grid[i] <= temperature_grid[i - 1]) {
            std::cerr << "ComptonTableReader: temperature grid is not monotonically increasing" << std::endl;
            std::exit(1);
        }
    }

    std::size_t const n_temps = temperature_grid.size();
    S_log_tables.resize(n_temps, Matrix(num_energy_groups, Vector(num_energy_groups, 0.0)));

    for (std::size_t i = 0; i < n_temps; ++i) {
        auto const mat_path = (fs::path(directory) / (std::to_string(i) + ".txt")).string();
        if (!fs::exists(mat_path)) {
            std::cerr << "ComptonTableReader: missing table file " << mat_path << std::endl;
            std::exit(1);
        }
        Matrix S = read_matrix_file(mat_path, num_energy_groups);
        for (std::size_t g0 = 0; g0 < num_energy_groups; ++g0) {
            for (std::size_t g = 0; g < num_energy_groups; ++g) {
                double const val = S[g0][g];
                S_log_tables[i][g0][g] = std::log(std::max(val, std::numeric_limits<double>::min()));
            }
        }
    }

    compute_dSdUm_tables();

    std::cout << "ComptonTableReader: loaded " << n_temps
              << " temperature tables from " << directory << std::endl;
}

void ComptonTableReader::validate_energy_grid(Vector const& table_boundaries) const {
    if (table_boundaries.size() != num_energy_groups + 1) {
        std::cerr << "ComptonTableReader: energy_groups.txt has "
                  << table_boundaries.size() << " entries, expected "
                  << num_energy_groups + 1 << std::endl;
        std::exit(1);
    }
    for (std::size_t i = 0; i < table_boundaries.size(); ++i) {
        double const rel = std::abs(table_boundaries[i] - energy_groups_boundaries[i])
                         / std::max(std::abs(energy_groups_boundaries[i]), 1e-300);
        if (rel > 1e-12) {
            std::cerr << "ComptonTableReader: energy grid mismatch at index " << i
                      << ": table=" << table_boundaries[i]
                      << " expected=" << energy_groups_boundaries[i]
                      << " relative error=" << rel << std::endl;
            std::exit(1);
        }
    }
}

void ComptonTableReader::compute_dSdUm_tables() {
    std::size_t const n_temps = temperature_grid.size();
    dSdUm_tables.resize(n_temps, Matrix(num_energy_groups, Vector(num_energy_groups, 0.0)));

    for (std::size_t i = 0; i < n_temps; ++i) {
        std::size_t lower = i > 0 ? i - 1 : 0;
        std::size_t upper = i < n_temps - 1 ? i + 1 : n_temps - 1;
        while ((temperature_grid[upper] - temperature_grid[lower]) <
               std::max(1e5, temperature_grid[i] * 0.2)) {
            if (lower > 0) --lower;
            if (upper < n_temps - 1) ++upper;
            if (lower == 0 && upper == n_temps - 1) break;
        }
        double const dT = temperature_grid[upper] - temperature_grid[lower];
        for (std::size_t g0 = 0; g0 < num_energy_groups; ++g0) {
            for (std::size_t g = 0; g < num_energy_groups; ++g) {
                dSdUm_tables[i][g0][g] =
                    (std::exp(S_log_tables[upper][g0][g]) - std::exp(S_log_tables[lower][g0][g])) / dT;
            }
        }
    }
}

void ComptonTableReader::set_Bg_ng(double const temperature) const {
    using boost::math::pow;
    double constexpr fac = pow<3>(units::clight) / (8.0 * M_PI * units::planck_constant);
    for (std::size_t g = 0; g < num_energy_groups; ++g) {
        double const Bg = planck_integral::planck_energy_density_group_integral(
            energy_groups_boundaries[g], energy_groups_boundaries[g + 1], temperature);
        double const nu  = energy_groups_centers[g] / units::planck_constant;
        double const dnu = (energy_groups_boundaries[g + 1] - energy_groups_boundaries[g])
                         / units::planck_constant;
        n_eq[g] = fac * Bg / (pow<3>(nu) * dnu);
        B[g] = Bg;
    }
}

void ComptonTableReader::get_tau_matrix(
    double const temperature, double const density,
    double const A, double const Z,
    Matrix& tau, Matrix& dtau_dUm) const
{
    auto const tmp_iterator = std::lower_bound(
        temperature_grid.cbegin(), temperature_grid.cend(), temperature);
    auto const tmp_i = std::distance(temperature_grid.cbegin(), tmp_iterator) - 1;

    if (tmp_i + 1 == static_cast<int>(temperature_grid.size())) {
        printf("ComptonTableReader: temperature T=%gkev is too high (max=%gkev)\n",
               temperature / units::kev_kelvin, temperature_grid.back() / units::kev_kelvin);
        std::exit(1);
    }
    if (tmp_i == -1) {
        printf("ComptonTableReader: temperature T=%gkev is too low (min=%gkev)\n",
               temperature / units::kev_kelvin, temperature_grid[0] / units::kev_kelvin);
        std::exit(1);
    }

    if (force_detailed_balance) set_Bg_ng(temperature);

    double const x = (temperature - temperature_grid[tmp_i])
                   / (temperature_grid[tmp_i + 1] - temperature_grid[tmp_i]);

    for (std::size_t i = 0; i < num_energy_groups; ++i) {
        double const E_i = energy_groups_centers[i];
        for (std::size_t j = i; j < num_energy_groups; ++j) {
            tau[i][j] = std::exp(S_log_tables[tmp_i][i][j]) * (1.0 - x)
                      + std::exp(S_log_tables[tmp_i + 1][i][j]) * x;
            dtau_dUm[i][j] = dSdUm_tables[tmp_i][i][j] * (1.0 - x)
                            + dSdUm_tables[tmp_i + 1][i][j] * x;

            if (i == j) continue;

            tau[j][i] = std::exp(S_log_tables[tmp_i][j][i]) * (1.0 - x)
                      + std::exp(S_log_tables[tmp_i + 1][j][i]) * x;
            dtau_dUm[j][i] = dSdUm_tables[tmp_i][j][i] * (1.0 - x)
                            + dSdUm_tables[tmp_i + 1][j][i] * x;

            if (force_detailed_balance) {
                if (B[j] * E_i < std::numeric_limits<double>::min() * 1e40)
                    continue;
                double const E_j = energy_groups_centers[j];
                double const detailed_balance_factor =
                    (1.0 + n_eq[j]) * B[i] * E_j / ((1.0 + n_eq[i]) * B[j] * E_i);

                if (std::isnan(detailed_balance_factor)) continue;

                if (detailed_balance_factor < 1.0) {
                    tau[j][i] = tau[i][j] * detailed_balance_factor;
                } else {
                    tau[i][j] = tau[j][i] / detailed_balance_factor;
                }
            }
        }
    }

    double const Nelectron = density * units::Navogadro / A * Z;
    for (std::size_t i = 0; i < num_energy_groups; ++i) {
        for (std::size_t j = 0; j < num_energy_groups; ++j) {
            tau[i][j] *= Nelectron;
            dtau_dUm[i][j] *= Nelectron;
        }
    }
}
