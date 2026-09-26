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
#include "source/Radiation/conj_grad_solve.hpp"

#include <boost/math/special_functions/pow.hpp>

using boost::math::pow;

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
    Vector const& energy_groups_boundaries_)
    : energy_groups_centers(energy_groups_centers_),
      energy_groups_boundaries(energy_groups_boundaries_),
      num_energy_groups(energy_groups_centers_.size()),
      sigma_s_buf(energy_groups_centers_.size(), Vector(energy_groups_centers_.size(), 0.0)),
      dsigma_s_buf(energy_groups_centers_.size(), Vector(energy_groups_centers_.size(), 0.0)),
      B_eq_buf(energy_groups_centers_.size(), 0.0),
      n_eq_buf(energy_groups_centers_.size(), 0.0),
      dBdT_buf(energy_groups_centers_.size(), 0.0),
      n_buf(energy_groups_centers_.size(), 0.0)
{}

void ComptonTableReader::validate_table_version(std::string const& path) const {
    std::ifstream ifs(path);
    if (!ifs.is_open()) {
        std::cerr << "ComptonTableReader: cannot open " << path << std::endl;
        std::exit(1);
    }
    std::string first_line;
    std::getline(ifs, first_line);
    if (first_line.find("version:") == std::string::npos) {
        std::cerr << "ComptonTableReader: no version marker in " << path
                  << ". Expected version " << EXPECTED_TABLE_VERSION << std::endl;
        std::exit(1);
    }
    auto pos = first_line.find("version:") + 8;
    auto end_pos = first_line.find("(", pos);
    int version = std::stoi(first_line.substr(pos, end_pos - pos));
    if (version != EXPECTED_TABLE_VERSION) {
        std::cerr << "ComptonTableReader: table version " << version
                  << " is not supported. Expected version "
                  << EXPECTED_TABLE_VERSION << std::endl;
        std::exit(1);
    }
}

void ComptonTableReader::load_tables(std::string const& directory) {
    namespace fs = std::filesystem;

    if (!fs::is_directory(directory)) {
        std::cerr << "ComptonTableReader: table directory does not exist: "
                  << directory << std::endl;
        std::exit(1);
    }

    auto const energy_path = (fs::path(directory) / "energy_groups.txt").string();
    auto const temp_path   = (fs::path(directory) / "temperatures.txt").string();

    validate_table_version(energy_path);

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

    sigma_s_tables.resize(n_temps);
    dsigma_s_dT_tables.resize(n_temps);

    for (std::size_t i = 0; i < n_temps; ++i) {
        auto const sigma_s_path = (fs::path(directory) / (std::to_string(i) + ".txt")).string();
        if (!fs::exists(sigma_s_path)) {
            std::cerr << "ComptonTableReader: missing table file " << sigma_s_path << std::endl;
            std::exit(1);
        }
        sigma_s_tables[i] = read_matrix_file(sigma_s_path, num_energy_groups);

        auto const dsigma_s_path = (fs::path(directory) / ("dSdT_" + std::to_string(i) + ".txt")).string();
        if (!fs::exists(dsigma_s_path)) {
            std::cerr << "ComptonTableReader: missing derivative table file " << dsigma_s_path << std::endl;
            std::exit(1);
        }
        dsigma_s_dT_tables[i] = read_matrix_file(dsigma_s_path, num_energy_groups);
    }

    std::cout << "ComptonTableReader: loaded " << n_temps
              << " sigma_s tables (version " << EXPECTED_TABLE_VERSION
              << ") from " << directory << std::endl;
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

void ComptonTableReader::get_S_and_dSdUm(
    double const T, double const density,
    double const A, double const Z,
    Vector const& E_g, bool const calculate_n,
    Matrix& S, Matrix& dSdUm) const
{
    // 1. Interpolate sigma_s and dsigma_s/dT at temperature T
    auto const tmp_iterator = std::lower_bound(
        temperature_grid.cbegin(), temperature_grid.cend(), T);
    auto const tmp_i = std::distance(temperature_grid.cbegin(), tmp_iterator) - 1;

    if (tmp_i + 1 == static_cast<int>(temperature_grid.size())) {
        printf("ComptonTableReader: temperature T=%gkev is too high (max=%gkev)\n",
               T / units::kev_kelvin, temperature_grid.back() / units::kev_kelvin);
        std::exit(1);
    }
    if (tmp_i == -1) {
        printf("ComptonTableReader: temperature T=%gkev is too low (min=%gkev)\n",
               T / units::kev_kelvin, temperature_grid[0] / units::kev_kelvin);
        std::exit(1);
    }

    double const x = (T - temperature_grid[tmp_i])
                   / (temperature_grid[tmp_i + 1] - temperature_grid[tmp_i]);

    double const omx = 1.0 - x;
    for (std::size_t i = 0; i < num_energy_groups; ++i) {
        for (std::size_t j = 0; j < num_energy_groups; ++j) {
            sigma_s_buf[i][j]  = sigma_s_tables[tmp_i][i][j] * omx + sigma_s_tables[tmp_i + 1][i][j] * x;
            dsigma_s_buf[i][j] = dsigma_s_dT_tables[tmp_i][i][j] * omx + dsigma_s_dT_tables[tmp_i + 1][i][j] * x;
        }
    }

    // 2. Scale by N_e, and clip negative off-diagonal coefficients from quadrature noise
    double const Nelectron = density * units::Navogadro / A * Z;
    for (std::size_t i = 0; i < num_energy_groups; ++i) {
        for (std::size_t j = 0; j < num_energy_groups; ++j) {
            sigma_s_buf[i][j]  *= Nelectron;
            dsigma_s_buf[i][j] *= Nelectron;
            if (i != j && sigma_s_buf[i][j] < 0.0) {
                sigma_s_buf[i][j] = 0.0;
                dsigma_s_buf[i][j] = 0.0;
            }
        }
    }

    // 3. Compute equilibrium B_g(T), n^B_g(T), dB_g/dT
    double constexpr fac = pow<3>(units::clight) / (8.0 * M_PI * units::planck_constant);

    for (std::size_t g = 0; g < num_energy_groups; ++g) {
        B_eq_buf[g] = planck_integral::planck_energy_density_group_integral(
            energy_groups_boundaries[g], energy_groups_boundaries[g + 1], T);
        double const nu  = energy_groups_centers[g] / units::planck_constant;
        double const dnu = (energy_groups_boundaries[g + 1] - energy_groups_boundaries[g])
                         / units::planck_constant;
        n_eq_buf[g] = fac * B_eq_buf[g] / (pow<3>(nu) * dnu);
    }

    for (std::size_t g = 0; g < num_energy_groups; ++g) {
        dBdT_buf[g] = planck_integral::planck_energy_density_dBdT_group(
            energy_groups_boundaries[g], energy_groups_boundaries[g + 1], T);
    }

    // 4. Compute nonequilibrium occupation numbers from E_g
    for (std::size_t g = 0; g < num_energy_groups; ++g) {
        if (calculate_n) {
            double const nu  = energy_groups_centers[g] / units::planck_constant;
            double const dnu = (energy_groups_boundaries[g + 1] - energy_groups_boundaries[g])
                             / units::planck_constant;
            n_buf[g] = std::min(100.0, fac * E_g[g] / (pow<3>(nu) * dnu));
        } else {
            n_buf[g] = 0.0;
        }
    }

    // 5. Always-shrink DB enforcement + analytical derivatives
    double constexpr tiny_thresh = std::numeric_limits<double>::min() * 1e40;
    double constexpr occupation_cap = 100.0;

    // Use capped equilibrium occupancies in detailed-balance factors:
    //   n_eq_db = min(occupation_cap, n_eq_raw)
    // and piecewise-consistent derivatives:
    //   dn_eq_db/dT = dn_eq_raw/dT when uncapped, else 0.
    //
    // For DB derivative formulas we need:
    //   d/dT ln(B/(1+n_eq_db)) = dlnB - dln(1+n_eq_db)
    Vector n_eq_db(num_energy_groups, 0.0);
    Vector log_db_factor_deriv(num_energy_groups, 0.0);
    for (std::size_t g = 0; g < num_energy_groups; ++g) {
        n_eq_db[g] = std::min(occupation_cap, n_eq_buf[g]);

        double dn_eq_raw_dT = 0.0;
        if (B_eq_buf[g] > tiny_thresh) {
            dn_eq_raw_dT = n_eq_buf[g] * dBdT_buf[g] / B_eq_buf[g];
        }
        double const dn_eq_db_dT = (n_eq_buf[g] < occupation_cap) ? dn_eq_raw_dT : 0.0;

        if (B_eq_buf[g] > tiny_thresh) {
            double const dlnB = dBdT_buf[g] / B_eq_buf[g];
            double const dln_n1 = dn_eq_db_dT / (1.0 + n_eq_db[g]);
            log_db_factor_deriv[g] = dlnB - dln_n1;
        } else {
            log_db_factor_deriv[g] = 0.0;
        }
    }

    // Photon-number detailed balance on each unordered pair g < gp:
    //   sigma_s[g][gp] (1+n_gp) B_g/nu_g = sigma_s[gp][g] (1+n_g) B_gp/nu_gp,
    // i.e. the balanced reverse coefficient is Q * sigma_s[g][gp]. The larger of
    // the two coefficients is shrunk to its balanced value.
    for (std::size_t g = 0; g < num_energy_groups; ++g) {
        for (std::size_t gp = g + 1; gp < num_energy_groups; ++gp) {
            if (B_eq_buf[g] < tiny_thresh || B_eq_buf[gp] < tiny_thresh) continue;
            double const forward = sigma_s_buf[g][gp];
            double const reverse = sigma_s_buf[gp][g];
            if (forward < tiny_thresh && reverse < tiny_thresh) continue;

            double const Q = ((1.0 + n_eq_db[gp]) * B_eq_buf[g])
                           / ((1.0 + n_eq_db[g]) * B_eq_buf[gp])
                           * (energy_groups_centers[gp] / energy_groups_centers[g]);
            if (!std::isfinite(Q) || Q <= tiny_thresh) continue;
            double const dlnQ = log_db_factor_deriv[g] - log_db_factor_deriv[gp];

            double const balanced_reverse = Q * forward;
            if (reverse > balanced_reverse) {
                sigma_s_buf[gp][g] = balanced_reverse;
                dsigma_s_buf[gp][g] = Q * (dsigma_s_buf[g][gp] + forward * dlnQ);
            } else if (reverse < balanced_reverse) {
                sigma_s_buf[g][gp] = reverse / Q;
                dsigma_s_buf[g][gp] = (dsigma_s_buf[gp][g] - reverse * dlnQ) / Q;
            }
        }
    }

    // 6. Assemble S and dS/dT. A transfer g -> gp removes sigma_s (1+n_gp) E_g
    //    from g and deposits sigma_s (nu_gp/nu_g) (1+n_gp) E_g into gp.
    for (std::size_t i = 0; i < num_energy_groups; ++i) {
        for (std::size_t j = 0; j < num_energy_groups; ++j) {
            S[i][j] = 0.0;
            dSdUm[i][j] = 0.0;
        }
    }

    Matrix dS_dT(num_energy_groups, Vector(num_energy_groups, 0.0));

    for (std::size_t g = 0; g < num_energy_groups; ++g) {
        for (std::size_t gp = 0; gp < num_energy_groups; ++gp) {
            if (gp == g) continue;
            double const n1 = 1.0 + n_buf[gp];
            double const ratio = energy_groups_centers[gp] / energy_groups_centers[g];

            S[g][g]      -= sigma_s_buf[g][gp] * n1;
            dS_dT[g][g]  -= dsigma_s_buf[g][gp] * n1;

            S[g][gp]     += sigma_s_buf[g][gp] * ratio * n1;
            dS_dT[g][gp] += dsigma_s_buf[g][gp] * ratio * n1;
        }
    }

    // 7. Convert dS/dT to dS/dU_m
    double const Um_factor = 1.0 / (4.0 * CG::radiation_constant * pow<3>(T));
    for (std::size_t i = 0; i < num_energy_groups; ++i) {
        for (std::size_t j = 0; j < num_energy_groups; ++j) {
            dSdUm[i][j] = dS_dT[i][j] * Um_factor;
        }
    }
}
