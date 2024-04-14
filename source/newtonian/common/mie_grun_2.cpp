#include "mie_grun_2.hpp"
#include "source/misc/universal_error.hpp"
#include <cmath>
#include <limits>

MieGrun2::MieGrun2(double const rho0, double const a0, double const gamma, double const s) : rho0_(rho0), gamma_(gamma), a0_(a0), s_(s) {}

double MieGrun2::dp2e(double d, double p, tvector const& tracers, vector<string> const& tracernames) const
{
    double const eta = d/rho0_;
    double const g = (eta-1)*(eta-0.5*gamma_*(eta-1))/((eta-s_*(eta-1))*(eta-s_*(eta-1)));
    return (p-rho0_*a0_*a0_*g)/(gamma_*rho0_);
}

double MieGrun2::de2p(double d, double e, tvector const& tracers, vector<string> const& tracernames) const
{
    double const eta = d/rho0_;
    double const g = (eta-1)*(eta-0.5*gamma_*(eta-1))/((eta-s_*(eta-1))*(eta-s_*(eta-1)));
    return rho0_*a0_*a0_*g + gamma_ * rho0_*e;
}

double MieGrun2::dp2c(double d, double p, tvector const& tracers, vector<string> const& tracernames) const
{
    double const eta = d/rho0_;
    double const g = (eta-1)*(eta-0.5*gamma_*(eta-1))/((eta-s_*(eta-1))*(eta-s_*(eta-1)));
    double const dgdeta = -(gamma_*(eta-1)-s_*eta+s_-eta)/((eta-s_*(eta-1))*(eta-s_*(eta-1))*(eta-s_*(eta-1)));
    return a0_;
    return std::sqrt(dgdeta*a0_*a0_ + rho0_*gamma_*p/(d*d));
}

double MieGrun2::de2c(double d, double e, tvector const& tracers, vector<string> const& tracernames) const
{
    return dp2c(d, de2p(d, e, tracers, tracernames), tracers, tracernames);
}


double MieGrun2::dp2s(double d, double p, tvector const& tracers, vector<string> const& tracernames) const
{
    throw UniversalError("dp2s not implemented in MieGrun2");
    return 0;
}

double MieGrun2::sd2p(double s, double d, tvector const& tracers, vector<string> const& tracernames) const
{
    throw UniversalError("sd2p not implemented in MieGrun2");
    return 0;
}

double MieGrun2::dT2cv(double d, double T, tvector const& tracers, vector<string> const& tracernames) const
{
    throw UniversalError("dT2cv not implemented in MieGrun2");
    return 0;
}

double MieGrun2::de2T(double d, double e, tvector const& tracers, vector<string> const& tracernames) const
{
    throw UniversalError("de2T not implemented in MieGrun2");
    return 0;
}

double MieGrun2::dT2e(double d, double T, tvector const& tracers, vector<string> const& tracernames) const
{
    throw UniversalError("dT2e not implemented in MieGrun2");
    return 0;
}