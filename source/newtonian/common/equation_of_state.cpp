#include "equation_of_state.hpp"
#include "source/misc/universal_error.hpp"

namespace
{
	/* Calls a scalar form for each item in turn, on the host. A throwing item is marked failed when there is a
	   status array; otherwise the exception propagates. */
	template<typename Scalar>
	void ForEachItem(EosArrays const& arrays, portable::Where where, Scalar const& scalar)
	{
		if(where == portable::Where::Device)
		{
			throw UniversalError("EquationOfState: this equation of state does not run on the device");
		}
		const vector<string> noNames;
		const vector<string> &names = arrays.tracerNames != nullptr ? *arrays.tracerNames : noNames;
		for(std::size_t i = 0; i < arrays.n; ++i)
		{
			if(arrays.status != nullptr and arrays.status[i] != 0)
			{
				continue;
			}
			const std::size_t k = i * arrays.stride;
			tvector tracers{};
			for(std::size_t j = 0; j < arrays.tracersNum; ++j)
			{
				tracers[j] = arrays.tracers[k + j];
			}
			try
			{
				arrays.result[i * arrays.resultStride] = scalar(arrays.density[k], arrays.second[k], tracers, names);
			}
			catch(...)
			{
				if(arrays.status == nullptr)
				{
					throw;
				}
				arrays.status[i] = arrays.failure;
			}
		}
	}
}

EquationOfState::~EquationOfState(void) {}

void EquationOfState::dp2eArrays(EosArrays const& arrays, portable::Where where) const
{
	ForEachItem(arrays, where, [this](double d, double p, tvector const& tracers, vector<string> const& names)
		{ return this->dp2e(d, p, tracers, names); });
}

void EquationOfState::de2pArrays(EosArrays const& arrays, portable::Where where) const
{
	ForEachItem(arrays, where, [this](double d, double e, tvector const& tracers, vector<string> const& names)
		{ return this->de2p(d, e, tracers, names); });
}

void EquationOfState::dp2cArrays(EosArrays const& arrays, portable::Where where) const
{
	ForEachItem(arrays, where, [this](double d, double p, tvector const& tracers, vector<string> const& names)
		{ return this->dp2c(d, p, tracers, names); });
}

void EquationOfState::de2cArrays(EosArrays const& arrays, portable::Where where) const
{
	ForEachItem(arrays, where, [this](double d, double e, tvector const& tracers, vector<string> const& names)
		{ return this->de2c(d, e, tracers, names); });
}

double EquationOfState::dT2cv(double const /*d*/, double const /*T*/,
	  tvector const& /*tracers*/, vector<string> const& /*tracernames*/) const
{
    throw UniversalError("dT2cv not implemented");
    return 0;
}

double EquationOfState::de2T(double const /*d*/, double const /*e*/,
	  tvector const& /*tracers*/, vector<string> const& /*tracernames*/) const
{
    throw UniversalError("de2T not implemented");
    return 0;
}

double EquationOfState::dT2e(double const /*d*/, double const /*T*/,
	  tvector const& /*tracers*/, vector<string> const& /*tracernames*/) const
{
   throw UniversalError("dT2e not implemented");
   return 0;
}