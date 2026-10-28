/*! \file PortableFunction.hpp
  \brief Marks functions that run on the host and, when compiled by a device compiler, on the device, and says
  where a loop runs. It does not include Kokkos, so plain headers (an equation of state) can use it.
*/
#ifndef RICH_PORTABLE_FUNCTION_HPP
#define RICH_PORTABLE_FUNCTION_HPP

#include <cmath>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define RICH_INLINE_FUNCTION __host__ __device__ inline
#else
#define RICH_INLINE_FUNCTION inline
#endif

namespace portable
{
//! \brief Where a loop runs: the host's threads, or Kokkos' default execution space (e.g. a GPU)
enum class Where
{
	Host,
	Device
};

// The same results as std::min and std::max, NaN and signed zeros included
RICH_INLINE_FUNCTION double Min(double a, double b)
{
	return (b < a) ? b : a;
}

RICH_INLINE_FUNCTION double Max(double a, double b)
{
	return (a < b) ? b : a;
}

RICH_INLINE_FUNCTION double Abs(double x)
{
	return ::fabs(x);
}

RICH_INLINE_FUNCTION double Sqrt(double x)
{
	return ::sqrt(x);
}

RICH_INLINE_FUNCTION double CopySign(double x, double sign)
{
	return ::copysign(x, sign);
}

// std::isfinite: x - x is 0 for finite x and NaN for infinities and NaN
RICH_INLINE_FUNCTION bool IsFinite(double x)
{
	return x - x == 0.0;
}
} // namespace portable

#endif // RICH_PORTABLE_FUNCTION_HPP
