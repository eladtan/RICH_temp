#include <cmath>
#include "ideal_gas.hpp"
#include "../../misc/universal_error.hpp"
#include "utils/Portable.hpp"
#include <limits>

IdealGas::IdealGas(double const AdiabaticIndex) : g_(AdiabaticIndex), f_(std::numeric_limits<double>::signaling_NaN()), 
	beta_(std::numeric_limits<double>::signaling_NaN()), mu_(std::numeric_limits<double>::signaling_NaN()) {}

IdealGas::IdealGas(double const AdiabaticIndex, double const f, double const beta, double const mu) :
	g_(AdiabaticIndex), f_(f), beta_(beta), mu_(mu) {}

double IdealGas::getAdiabaticIndex(void) const
{
	return g_;
}

double IdealGas::dp2e(double d, double p, tvector const& /*tracers*/, vector<string> const& /*tracernames*/) const
{
#ifdef RICH_DEBUG
	if (!(g_ > 0) || !(p > 0) || !(d > 0))
	{
		UniversalError eo("Negative quantity in ideal gas dp2e");
		eo.addEntry("Density", d);
		eo.addEntry("Pressure", p);
		eo.addEntry("Gamma index", g_);
		throw eo;
	}
#endif //RICH_DEBUG

	double e = 0;
	ideal_gas::dp2e(g_, d, p, e);
	return e;
}

double IdealGas::de2p(double d, double e, tvector const& /*tracers*/, vector<string> const& /*tracernames*/) const
{
#ifdef RICH_DEBUG
	if (!(g_ > 0) || !(e > 0) || !(d > 0))
	{
		UniversalError eo("Negative quantity in ideal gas de2p");
		eo.addEntry("Density", d);
		eo.addEntry("Energy", e);
		eo.addEntry("Gamma index", g_);
		throw eo;
	}
#endif // RICH_DEBUG

	double p = 0;
	if (!ideal_gas::de2p(g_, d, e, p))
		throw UniversalError("Negative thermal energy");
	return p;
}

namespace {
	/*
	UniversalError imaginary_speed_of_sound(double g,
						double p,
						double d)
	{
	  UniversalError res("Speed of sound came out imaginary");
	  res.addEntry("adiabatic index",g);
	  res.addEntry("density",d);
	  res.addEntry("pressure",p);
	  return res;
	}
	*/
}

double IdealGas::dp2c(double d, double p, tvector const& /*tracers*/, vector<string> const& /*tracernames*/) const
{
#ifdef RICH_DEBUG
	if (!(g_ > 0) || !(p > 0) || !(d > 0))
	{
		UniversalError eo("Negative quantity in ideal gas");
		eo.addEntry("Density", d);
		eo.addEntry("Pressure", p);
		eo.addEntry("Gamma index", g_);
		throw eo;
	}
#endif //RICH_DEBUG
	assert(!(p < 0));
	assert(!(d < 0));
	assert(!(g_ < 0));
	double c = 0;
	ideal_gas::dp2c(g_, d, p, c);
	return c;
}

double IdealGas::de2c(double d, double e, tvector const& tracers, vector<string> const& tracernames) const
{
	double p = de2p(d, e, tracers, tracernames);
	double c = 0;
	ideal_gas::dp2c(g_, d, p, c);
	return c;
}

namespace
{
	// Applies a formula to every item: a functor, so that it can run on the device
	template<typename Formula>
	struct ForEachItem
	{
		static constexpr bool onDevice = true;
		EosArrays arrays;
		double g = 0;

		RICH_INLINE_FUNCTION void operator()(std::size_t i) const
		{
			if(this->arrays.status != nullptr and this->arrays.status[i] != 0)
			{
				return;
			}
			const std::size_t k = i * this->arrays.stride;
			if(not Formula::Apply(this->g, this->arrays.density[k], this->arrays.second[k], this->arrays.result[i * this->arrays.resultStride])
				and this->arrays.status != nullptr)
			{
				this->arrays.status[i] = this->arrays.failure;
			}
		}
	};

	struct Dp2e
	{
		RICH_INLINE_FUNCTION static bool Apply(double g, double d, double p, double &e) { return ideal_gas::dp2e(g, d, p, e); }
	};

	struct De2p
	{
		RICH_INLINE_FUNCTION static bool Apply(double g, double d, double e, double &p) { return ideal_gas::de2p(g, d, e, p); }
	};

	struct Dp2c
	{
		RICH_INLINE_FUNCTION static bool Apply(double g, double d, double p, double &c) { return ideal_gas::dp2c(g, d, p, c); }
	};

	struct De2c
	{
		RICH_INLINE_FUNCTION static bool Apply(double g, double d, double e, double &c) { return ideal_gas::de2c(g, d, e, c); }
	};
}

void IdealGas::dp2eArrays(EosArrays const& arrays, portable::Where where) const
{
	portable::ParallelFor(where, "IdealGas::dp2e", arrays.n, ForEachItem<Dp2e>{arrays, g_});
}

void IdealGas::de2pArrays(EosArrays const& arrays, portable::Where where) const
{
	portable::ParallelFor(where, "IdealGas::de2p", arrays.n, ForEachItem<De2p>{arrays, g_});
}

void IdealGas::dp2cArrays(EosArrays const& arrays, portable::Where where) const
{
	portable::ParallelFor(where, "IdealGas::dp2c", arrays.n, ForEachItem<Dp2c>{arrays, g_});
}

void IdealGas::de2cArrays(EosArrays const& arrays, portable::Where where) const
{
	portable::ParallelFor(where, "IdealGas::de2c", arrays.n, ForEachItem<De2c>{arrays, g_});
}

bool IdealGas::RunsOnDevice(void) const
{
	return true;
}

bool IdealGas::ThreadSafe(void) const
{
	return true;
}

double IdealGas::dp2s(double d, double p, tvector const& /*tracers*/, vector<string> const& /*tracernames*/) const
{
	return p * pow(d, -g_);
}

double IdealGas::sd2p(double s, double d, tvector const& /*tracers*/, vector<string> const& /*tracernames*/) const
{
	if ((!(s > 0) || (!(d > 0))))
	{
		UniversalError eo("Negative quantaties in entropy calc");
		eo.addEntry("entropy", s);
		eo.addEntry("density", d);
		throw eo;
	}
	assert(s > 0 && d > 0);
	return s * pow(d, g_);
}

double IdealGas::dT2cv(double const d, double const T, tvector const& /*tracers*/, vector<string> const& /*tracernames*/) const
{
	return f_ * beta_ * std::pow(T, beta_ - 1) * std::pow(d, 1 - mu_);
}

double IdealGas::de2T(double const d, double const e, tvector const& /*tracers*/, vector<string> const& /*tracernames*/) const
{
	return std::pow(std::pow(d, mu_) * e / f_, 1.0 / beta_);
}

double IdealGas::dT2e(double const d, double const T, tvector const& /*tracers*/, vector<string> const& /*tracernames*/) const
{
	return f_ * std::pow(T, beta_) * std::pow(d, -mu_);
}