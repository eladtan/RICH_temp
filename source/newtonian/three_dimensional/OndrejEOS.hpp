#include "../common/equation_of_state.hpp"

#ifndef ONDREJEOS_HPP
#define ONDREJEOS_HPP 1


/**
 * @brief Class for the EOS described in "Radiation Magnetohydrodynamic Simulations of Protostellar Collapse: Protostellar Core Formation" Tomida et al 2013
 * the density table is from 1e-22 g/cc to 911 g/cc and the temperature is from 500K to 1e8 K
 * 
 */
class OndrejEOS : public EquationOfState
{
private:
    double mind_, maxd_, dd_, lscale_, mscale_, tscale_;
    std::vector<double> P_, cs_, S_, U_, T_, CV_;
    size_t Nt_;
    mutable std::vector<double> tvec_, dvec_;

    double InterpData2(double density, double other, std::vector<double> const& otherv, std::vector<double> const& data) const;

public:
    explicit OndrejEOS(std::string const& density_file, std::string const &Pfile, std::string const &csfile,
                       std::string const &Sfile, std::string const &Ufile, std::string const &Tfile, std::string const &CVfile, 
                       double lscale, double mscale, double tscale);

    double dp2e(double d, double p, tvector const &tracers = tvector(), vector<string> const &tracernames = std::vector<std::string>()) const;

    double dp2T(double d, double p, tvector const &tracers = tvector(), vector<string> const &tracernames = std::vector<std::string>()) const;

    double dT2p(double d, double T, tvector const &tracers = tvector(), vector<string> const &tracernames = std::vector<std::string>()) const;

    double dT2e(double d, double T, tvector const &tracers = tvector(), vector<string> const &tracernames = std::vector<std::string>()) const;

    double de2p(double d, double e, tvector const &tracers = tvector(), vector<string> const &tracernames = std::vector<std::string>()) const;

    double dp2c(double d, double p, tvector const &tracers = tvector(), vector<string> const &tracernames = std::vector<std::string>()) const;

    double dp2cv(double d, double p, tvector const &tracers = tvector(), vector<string> const &tracernames = std::vector<std::string>()) const;

    double de2c(double d, double e, tvector const &tracers = tvector(), vector<string> const &tracernames = std::vector<std::string>()) const;

    double dp2s(double d, double p, tvector const &tracers = tvector(), vector<string> const &tracernames = std::vector<std::string>()) const;

    double sd2p(double s, double d, tvector const &tracers = tvector(), vector<string> const &tracernames = std::vector<std::string>()) const;

    double de2T(double const d, double const e, tvector const& tracers = tvector(), vector<string> const& tracernames = vector<string>()) const;

    double dT2cv(double const d, double const T, tvector const& tracers = tvector(), vector<string> const& tracernames = vector<string>()) const;
};
#endif