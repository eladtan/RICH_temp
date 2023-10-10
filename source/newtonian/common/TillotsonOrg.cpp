#include "TillotsonOrg.hpp"
#include <assert.h>
#include <boost/math/tools/roots.hpp>
#include <iostream>
#include "../../misc/universal_error.hpp"
#include "../../misc/utils.hpp"

TillotsonOrg::TillotsonOrg(double a, double b, double A, double B, double rho0, double E0, double EIV, double ECV, double
	alpha, double beta):
	a_(a), b_(b), A_(A), B_(B), rho0_(rho0), E0_(E0), EIV_(EIV), ECV_(ECV), alpha_(alpha), beta_(beta),
	temp_d_(0), temp_p_(0) {}

double TillotsonOrg::de2pI(double d, double e)const
{
	double eta = d / rho0_;
	double c = E0_ * eta*eta;
	double AB = (A_ - 2 * B_)*eta + (B_ - A_) + B_ * eta*eta;
	double res = 0;
	res = std::max((a_ + b_ / (e / c + 1))*d*e + AB, -d * e * 1e-2);
	return res;
}

double TillotsonOrg::de2pII(double d, double e)const
{
	double P2 = de2pI(d, e);
	double P3 = de2pIV(d, e);
	return ((e - EIV_)*P3 + (ECV_ - e)*P2) / (ECV_ - EIV_);
}

double TillotsonOrg::de2pIV(double d, double e)const
{
	double eta = d / rho0_;
	if (alpha_ > 100 * eta*eta)
		return a_ * d*e;
	double mu = eta - 1;
	double c = E0_ * eta*eta;
	double A = A_ * mu;
	double eta2 = alpha_ * (rho0_ / d - 1) * (rho0_ / d - 1);
	double exp_alpha = (eta2 > 100) ? 0 : std::exp(-eta2);
	double eta3 = beta_ * (rho0_ / d - 1);
	double exp_beta = (eta3 > 100) ? 0 : A * std::exp(-eta3);
	return std::max(a_*d*e + exp_alpha * (b_*d*e / (e / c + 1) + exp_beta), a_ *d*e*1e-5);
}

double TillotsonOrg::dep2cI(double d, double e) const
{
	double eta = d / rho0_;
	double c = E0_ * eta*eta;
	double w0 = e / c + 1;
	double gamma = a_ + b_ / w0;
	double const p = de2pI(d, e);
	double res = (gamma + 1)*p / d + (A_ + B_ * (eta*eta - 1)) / d + b_ * (w0 - 1)*(2 * e - p / d) / (w0*w0);
	res = std::max(res, 1e-10*E0_);
	return res;
}

double TillotsonOrg::dep2cIV(double d, double e) const
{
	double const p = de2pIV(d, e);
	double eta = d / rho0_;
	double w0 = e / (E0_*eta*eta) + 1;
	double z = 1.0 / eta - 1.0;
	double afactor = (alpha_*z*z > 100) ? 0 : std::exp(-alpha_ * z*z);
	double res0 = p * (a_ + b_ * afactor / w0 + 1) / d;
	double bfactor = (beta_*z > 100) ? 0 : std::exp(-beta_ * z);
	double res1 = A_ * bfactor*afactor*(1 + (eta - 1)*(beta_ + 2 * alpha_*z - eta) / (eta*eta)) / rho0_;
	res1 += b_ * d*e*afactor*(2 * alpha_*z*w0 / rho0_ + (p / d - 2 * e) / (E0_*d)) / (w0*w0*eta*eta);
	double res = std::max(res0 + res1, 1e-10*E0_);
	return res;
}


double TillotsonOrg::dp2e(double d, double p, tvector const & /*tracers*/, vector<string> const & /*tracernames*/) const
{
	throw UniversalError("TillotsonOrg::dp2e not implemented");
	return 0;
}

double TillotsonOrg::de2p(double d, double e, tvector const& /*tracers*/, vector<string> const& /*tracernames*/) const
{
	if (d >= rho0_)
	{
		return de2pI(d, e);
	}
	else
	{
		if (e <= EIV_ )
			return de2pI(d, e);
		if (e >= ECV_)
			return de2pIV(d, e);
		return de2pII(d, e);
	}
}

double TillotsonOrg::de2c(double d, double e, tvector const & tracers, vector<string> const & tracernames) const
{
	if (d >= rho0_  || e < EIV_)
	{
		double res = dep2cI(d, e);
		return std::sqrt(res);
	}
	else
	{
		if (e >= ECV_ )
		{
			double res = dep2cIV(d, e);
			assert(res > 0);
			return std::sqrt(res);
		}
		else
		{
			double c1 = dep2cI(d, e);
			double c4 = dep2cIV(d, e);
			double res = std::sqrt((c1*(ECV_ - e) + c4 * (e - EIV_)) / (ECV_ - EIV_));
			assert(res > 0);
			return res;
		}
	}
}

double TillotsonOrg::dp2c(double d, double p, tvector const & tracers, vector<string> const & tracernames) const
{
	throw UniversalError("TillotsonOrg::dp2c not implemented");
	return 0;
}

double TillotsonOrg::dp2s(double /*d*/, double /*p*/, tvector const & /*tracers*/, vector<string> const & /*tracernames*/) const
{
	throw UniversalError("TillotsonOrg::dp2s not implemented");
	return 0;
}

double TillotsonOrg::sd2p(double /*s*/, double /*d*/, tvector const & /*tracers*/, vector<string> const & /*tracernames*/) const
{
	throw UniversalError("TillotsonOrg::sd2p not implemented");
	return 0;
}

TillotsonOrg::~TillotsonOrg(void)
{}

