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
        Vector const& energy_groups_boundaries_,
        bool force_detailed_balance_);

    void load_tables(std::string const& directory);

    void get_tau_matrix(double temperature, double density, double A, double Z,
                        Matrix& tau, Matrix& dtau_dUm) const;

    double get_maximum_temperature_grid() const { return temperature_grid.back(); }

private:
    void validate_energy_grid(Vector const& table_boundaries) const;
    void set_Bg_ng(double temperature) const;

    Vector const energy_groups_centers;
    Vector const energy_groups_boundaries;
    std::size_t const num_energy_groups;
    bool const force_detailed_balance;

    Vector temperature_grid;
    std::vector<Matrix> S_log_tables;
    std::vector<Matrix> dSdT_tables;

    mutable Vector n_eq;
    mutable Vector B;
};

#endif
