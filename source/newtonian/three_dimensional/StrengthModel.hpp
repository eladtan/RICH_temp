/*! \file StrengthModel.hpp
\brief Abstract class for different constitutive models
\author Omri Reved
*/

#ifndef STRENGTHMODEL_HPP
#define STRENGTHMODEL_HPP 1
#include "computational_cell.hpp"

class StrengthModel
{
public:
    virtual double getG(ComputationalCell3D const& cell, size_t j) const = 0;
    virtual double getY(ComputationalCell3D const& cell, size_t j) const = 0;
};


//! \brief constant strength
class ConstStrength : public StrengthModel
{
public:
    ConstStrength(double G, double Y) : G_(G), Y_(Y) {};
    inline double getG(ComputationalCell3D const& cell, size_t j) const override
    {
        return G_;
    }
    inline double getY(ComputationalCell3D const& cell, size_t j) const override
    {
        return Y_;
    }
private:
    const double G_;
    const double Y_;
};

//! \brief Steinberg Gunian strength model
class SteinbergStrength : public StrengthModel
{
public:
    SteinbergStrength(double G, double Y, double GP, double GT, double Ymax, double rho0, double beta, double n, double eps0, double T0=300.) : G0_(G), Y0_(Y), GP_(GP), GT_(GT), Ymax_(Ymax), rho0_(rho0), beta_(beta), n_(n), eps0_(eps0), T0_(T0) {};
    inline double getG(ComputationalCell3D const& cell, size_t j) const override
    {
        double eta = cell.tracers[j]*cell.density/rho0_; //// BAD APPROXIMATION
        return G0_*(1.+GP_ * cell.pressure/std::pow(eta, 1./3.) + GT_*(cell.temperature - T0_));
    }
    inline double getY(ComputationalCell3D const& cell, size_t j) const override
    {
        double eta = cell.tracers[j]*cell.density/rho0_; //// BAD APPROXIMATION
        return std::min(Ymax_, Y0_*std::pow(1+beta_*(cell.strain_pl+eps0_),n_))*(1.+GP_ * cell.pressure/pow(eta, 1./3.) + GT_*(cell.temperature - T0_));
    }
private:
    const double G0_;
    const double Y0_;
    const double GP_;
    const double GT_;
    const double Ymax_;
    const double beta_;
    const double eps0_;
    const double T0_;
    const double rho0_;
    const double n_;
};

//! \brief Johnson Cook strength model
class JohnsonStrength : public StrengthModel
{
public:
    JohnsonStrength(double G, double A, double B, double C, double Tmelt, double Ttrans, double n, double m, double eps0dot=1.) : G0_(G), A_(A), B_(B), C_(C), Tmelt_(Tmelt), Ttrans_(Ttrans), n_(n), m_(m), eps0dot_(eps0dot) {};
    inline double getG(ComputationalCell3D const& cell, size_t j) const override
    {
        return G0_;
    }
    inline double getY(ComputationalCell3D const& cell, size_t j) const override
    {
        double theta = (cell.temperature - Ttrans_)/(Tmelt_-Ttrans_);
        theta = theta > 0. ? std::min(theta, 1.) : 0;
        return (A_+B_*pow(cell.strain_pl, n_))*std::max(1., 1.+C_*log(cell.strain_pl_dot/eps0dot_))*(1-pow(theta, m_));
    }
private:
    const double G0_;
    const double A_;
    const double B_;
    const double C_;
    const double Tmelt_;
    const double Ttrans_;
    const double eps0dot_;
    const double n_;
    const double m_;
};

#endif