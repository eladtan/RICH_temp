#ifndef COMPTON_TABLE_READER_HPP
#define COMPTON_TABLE_READER_HPP

#include <vector>
#include <string>

using Vector = std::vector<double>;
using Matrix = std::vector<std::vector<double>>;

class ComptonTableReader {
public:
    ComptonTableReader(
        Vector const& energy_groups_centers_,
        Vector const& energy_groups_boundaries_);

    void load_tables(std::string const& directory);

    void get_S_and_dSdUm(
        double T, double density, double A, double Z,
        Vector const& E_g, bool calculate_n,
        Matrix& S, Matrix& dSdUm) const;

    double get_maximum_temperature_grid() const { return temperature_grid.back(); }

private:
    void validate_energy_grid(Vector const& table_boundaries) const;
    void validate_table_version(std::string const& path) const;

    Vector const energy_groups_centers;
    Vector const energy_groups_boundaries;
    std::size_t const num_energy_groups;

    static constexpr int EXPECTED_TABLE_VERSION = 3;

    Vector temperature_grid;

    std::vector<Matrix> sigma_out_tables;
    std::vector<Matrix> dsigma_out_dT_tables;
    std::vector<Matrix> sigma_in_tables;
    std::vector<Matrix> dsigma_in_dT_tables;

    mutable Matrix sigma_out_buf;
    mutable Matrix dsigma_out_buf;
    mutable Matrix sigma_in_buf;
    mutable Matrix dsigma_in_buf;
    mutable Vector B_eq_buf;
    mutable Vector n_eq_buf;
    mutable Vector dBdT_buf;
    mutable Vector n_buf;
};

#endif
