/*! \file Portable.hpp
  \brief Arrays and loops shared by serial, host-thread and device runs. The only file that knows whether
  RICH is built with Kokkos: without it, loops run one after another on the host.
*/
#ifndef RICH_PORTABLE_HPP
#define RICH_PORTABLE_HPP

#include <cstddef>
#include <exception>
#include <mutex>
#include <utility>
#include <vector>
#include "utils/PortableFunction.hpp"
#ifdef RICH_WITH_KOKKOS
#include <Kokkos_Core.hpp>
#include "utils/KokkosInitializer.hpp"
#endif // RICH_WITH_KOKKOS

namespace portable
{
//! \brief Whether the device has memory of its own (otherwise it runs on the host's arrays)
#ifdef RICH_WITH_KOKKOS
inline constexpr bool separateDevice = not Kokkos::SpaceAccessibility<Kokkos::HostSpace,
	Kokkos::DefaultExecutionSpace::memory_space>::accessible;
#else
inline constexpr bool separateDevice = false;
#endif // RICH_WITH_KOKKOS

/*! \brief An array on the host, with a copy on the device when the device has memory of its own. The host
  array is a std::vector, so host code (MPI exchanges) uses it directly. T must be trivially copyable.
*/
template<typename T>
class Buffer
{
public:
	std::vector<T> &Host(void)
	{
		return this->host_;
	}

	const std::vector<T> &Host(void) const
	{
		return this->host_;
	}

	//! \brief The array a loop running at `where` uses
	T *Data(Where where)
	{
#ifdef RICH_WITH_KOKKOS
		if constexpr(separateDevice)
		{
			if(where == Where::Device)
			{
				return this->device_.data();
			}
		}
#endif // RICH_WITH_KOKKOS
		(void)where;
		return this->host_.data();
	}

	//! \brief Copies the host array to the device's
	void ToDevice(void)
	{
#ifdef RICH_WITH_KOKKOS
		if constexpr(separateDevice)
		{
			this->AllocateDevice();
			Kokkos::deep_copy(this->DeviceArray(), HostView(this->host_.data(), this->host_.size()));
		}
#endif // RICH_WITH_KOKKOS
	}

	//! \brief Copies the device array to the host's
	void ToHost(void)
	{
#ifdef RICH_WITH_KOKKOS
		if constexpr(separateDevice)
		{
			this->host_.resize(this->deviceSize_);
			Kokkos::deep_copy(HostView(this->host_.data(), this->host_.size()), this->DeviceArray());
		}
#endif // RICH_WITH_KOKKOS
	}

	//! \brief Copies host elements [first, first + count) to the device's (sized as the host's)
	void ToDevice(std::size_t first, std::size_t count)
	{
#ifdef RICH_WITH_KOKKOS
		if constexpr(separateDevice)
		{
			this->AllocateDevice();
			const std::pair<std::size_t, std::size_t> range(first, first + count);
			Kokkos::deep_copy(Kokkos::subview(this->device_, range), HostView(this->host_.data() + first, count));
		}
#else
		(void)first;
		(void)count;
#endif // RICH_WITH_KOKKOS
	}

	//! \brief Copies device elements [first, first + count) to the host's
	void ToHost(std::size_t first, std::size_t count)
	{
#ifdef RICH_WITH_KOKKOS
		if constexpr(separateDevice)
		{
			const std::pair<std::size_t, std::size_t> range(first, first + count);
			Kokkos::deep_copy(HostView(this->host_.data() + first, count), Kokkos::subview(this->device_, range));
		}
#else
		(void)first;
		(void)count;
#endif // RICH_WITH_KOKKOS
	}

	/*! \brief Sizes the device array as the host's, without copying (for a loop that only writes it). The device
	  memory only grows, with some headroom, so that arrays whose size changes a little each step are not reallocated.
	*/
	void AllocateDevice(void)
	{
#ifdef RICH_WITH_KOKKOS
		if constexpr(separateDevice)
		{
			const std::size_t n = this->host_.size();
			if(this->device_.extent(0) < n)
			{
				this->device_ = DeviceView();
				this->device_ = DeviceView(Kokkos::view_alloc(Kokkos::WithoutInitializing, "rich_buffer"), n + n / 8);
			}
			this->deviceSize_ = n;
		}
#endif // RICH_WITH_KOKKOS
	}

	void Release(void)
	{
		std::vector<T>().swap(this->host_);
#ifdef RICH_WITH_KOKKOS
		this->device_ = DeviceView();
		this->deviceSize_ = 0;
#endif // RICH_WITH_KOKKOS
	}

private:
	std::vector<T> host_;
#ifdef RICH_WITH_KOKKOS
	using DeviceView = Kokkos::View<T*, Kokkos::DefaultExecutionSpace::memory_space>;
	using HostView = Kokkos::View<T*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
	DeviceView device_;
	std::size_t deviceSize_ = 0;

	// The part of the device array in use
	auto DeviceArray(void) const
	{
		return Kokkos::subview(this->device_, std::make_pair(std::size_t(0), this->deviceSize_));
	}
#endif // RICH_WITH_KOKKOS
};

/*! \brief Runs f(i) for i in [0, n) at `where`, and returns when all are done. F::onDevice says whether f can
  run on the device; a functor that cannot always runs on the host's threads.
*/
template<typename F>
void ParallelFor(Where where, const char *name, std::size_t n, const F &f)
{
#ifdef RICH_WITH_KOKKOS
	KokkosInitializer::Instance();
	if constexpr(F::onDevice)
	{
		if(where == Where::Device)
		{
			Kokkos::parallel_for(name, Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, n), f);
			Kokkos::DefaultExecutionSpace().fence();
			return;
		}
	}
	Kokkos::parallel_for(name, Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, n), f);
	Kokkos::DefaultHostExecutionSpace().fence();
#else
	(void)where;
	(void)name;
	for(std::size_t i = 0; i < n; ++i)
	{
		f(i);
	}
#endif // RICH_WITH_KOKKOS
}

//! \brief The number of threads the host's loops run on (1 without Kokkos)
inline std::size_t HostThreads(void)
{
#ifdef RICH_WITH_KOKKOS
	KokkosInitializer::Instance();
	return static_cast<std::size_t>(Kokkos::DefaultHostExecutionSpace().concurrency());
#else
	return 1;
#endif // RICH_WITH_KOKKOS
}

//! \brief Runs f(i) for i in [0, n) on the host's threads (one after another without Kokkos), for host-only loops
template<typename F>
void HostParallelFor(std::size_t n, const F &f)
{
#ifdef RICH_WITH_KOKKOS
	KokkosInitializer::Instance();
	Kokkos::parallel_for("rich::HostParallelFor", Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, n),
		[&f](const std::size_t i) { f(i); });
	Kokkos::DefaultHostExecutionSpace().fence();
#else
	for(std::size_t i = 0; i < n; ++i)
	{
		f(i);
	}
#endif // RICH_WITH_KOKKOS
}

/*! \brief HostParallelFor for a loop whose items may throw: rethrows what the item with the lowest index threw, the
  error one item after another would have stopped at (the items after it may have run too). With parallel false, or
  without Kokkos, the items run one after another.
*/
template<typename F>
void HostParallelForOrdered(std::size_t n, const F &f, bool parallel = true)
{
#ifdef RICH_WITH_KOKKOS
	if(parallel)
	{
		std::size_t firstFailed = n;
		std::exception_ptr error;
		std::mutex lock;
		HostParallelFor(n, [&](std::size_t i)
		{
			try
			{
				f(i);
			}
			catch(...)
			{
				const std::lock_guard<std::mutex> guard(lock);
				if(i < firstFailed)
				{
					firstFailed = i;
					error = std::current_exception();
				}
			}
		});
		if(error)
		{
			std::rethrow_exception(error);
		}
		return;
	}
#else
	(void)parallel;
#endif // RICH_WITH_KOKKOS
	for(std::size_t i = 0; i < n; ++i)
	{
		f(i);
	}
}
} // namespace portable

#endif // RICH_PORTABLE_HPP
