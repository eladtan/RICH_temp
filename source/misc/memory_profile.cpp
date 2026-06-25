#include "memory_profile.hpp"

#ifdef MEMORY_PROFILE

#include <atomic>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <malloc.h>
#include <new>
#include <string>

namespace
{
    struct Counters
    {
        std::atomic<std::size_t> alloc_count{0};
        std::atomic<std::size_t> alloc_bytes{0};
        std::atomic<std::size_t> free_count{0};
        std::atomic<std::size_t> free_bytes{0};
        std::atomic<std::size_t> alloc_bins[8];
        std::atomic<std::size_t> free_bins[8];

        Counters()
        {
            for(std::size_t i = 0; i < 8; ++i)
            {
                alloc_bins[i].store(0, std::memory_order_relaxed);
                free_bins[i].store(0, std::memory_order_relaxed);
            }
        }
    };

    Counters &counters()
    {
        static Counters c;
        return c;
    }

    thread_local bool inside_allocator = false;

    std::size_t bin_for(std::size_t bytes)
    {
        if(bytes <= 64) return 0;
        if(bytes <= 256) return 1;
        if(bytes <= 1024) return 2;
        if(bytes <= 4096) return 3;
        if(bytes <= 16384) return 4;
        if(bytes <= 65536) return 5;
        if(bytes <= 1048576) return 6;
        return 7;
    }

    const char *bin_label(std::size_t bin)
    {
        static const char *labels[] = {"<=64", "<=256", "<=1K", "<=4K", "<=16K", "<=64K", "<=1M", ">1M"};
        return labels[bin];
    }

    void record_alloc(std::size_t bytes)
    {
        Counters &c = counters();
        c.alloc_count.fetch_add(1, std::memory_order_relaxed);
        c.alloc_bytes.fetch_add(bytes, std::memory_order_relaxed);
        c.alloc_bins[bin_for(bytes)].fetch_add(1, std::memory_order_relaxed);
    }

    void record_free(std::size_t bytes)
    {
        Counters &c = counters();
        c.free_count.fetch_add(1, std::memory_order_relaxed);
        c.free_bytes.fetch_add(bytes, std::memory_order_relaxed);
        c.free_bins[bin_for(bytes)].fetch_add(1, std::memory_order_relaxed);
    }

    double rss_gb()
    {
        std::ifstream status("/proc/self/status");
        std::string line;
        while(std::getline(status, line))
        {
            if(line.compare(0, 6, "VmRSS:") == 0)
            {
                std::size_t pos = line.find_first_of("0123456789");
                if(pos != std::string::npos)
                    return std::stod(line.substr(pos)) / (1024.0 * 1024.0);
            }
        }
        return 0.0;
    }

    void *allocate_raw(std::size_t size)
    {
        if(size == 0)
            size = 1;
        void *ptr = std::malloc(size);
        if(ptr == nullptr)
            throw std::bad_alloc();
        return ptr;
    }

    void free_raw(void *ptr) noexcept
    {
        if(ptr == nullptr)
            return;
        std::free(ptr);
    }
}

namespace memory_profile
{
    Snapshot snapshot()
    {
        Snapshot s;
        Counters &c = counters();
        s.alloc_count = c.alloc_count.load(std::memory_order_relaxed);
        s.alloc_bytes = c.alloc_bytes.load(std::memory_order_relaxed);
        s.free_count = c.free_count.load(std::memory_order_relaxed);
        s.free_bytes = c.free_bytes.load(std::memory_order_relaxed);
        for(std::size_t i = 0; i < 8; ++i)
        {
            s.alloc_bins[i] = c.alloc_bins[i].load(std::memory_order_relaxed);
            s.free_bins[i] = c.free_bins[i].load(std::memory_order_relaxed);
        }
        return s;
    }

    void print_scope(const char *label, const Snapshot &before, double seconds)
    {
        Snapshot after = snapshot();
        std::cerr << "[MEMORY_PROFILE] " << label
                  << ": time=" << std::fixed << std::setprecision(6) << seconds << "s"
                  << ", allocs=" << (after.alloc_count - before.alloc_count)
                  << ", alloc_bytes=" << (after.alloc_bytes - before.alloc_bytes)
                  << ", frees=" << (after.free_count - before.free_count)
                  << ", free_bytes=" << (after.free_bytes - before.free_bytes)
                  << ", rss=" << std::setprecision(3) << rss_gb() << " GB";
        std::cerr << ", alloc_bins={";
        bool first = true;
        for(std::size_t i = 0; i < 8; ++i)
        {
            std::size_t delta = after.alloc_bins[i] - before.alloc_bins[i];
            if(delta == 0)
                continue;
            if(!first)
                std::cerr << ",";
            std::cerr << bin_label(i) << ":" << delta;
            first = false;
        }
        std::cerr << "}" << std::endl;
    }

    Scope::Scope(const char *label)
        : label_(label),
          before_(snapshot()),
          start_(std::chrono::high_resolution_clock::now())
    {}

    Scope::~Scope()
    {
        const double seconds = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start_).count();
        print_scope(label_, before_, seconds);
    }
}

void *operator new(std::size_t size)
{
    if(inside_allocator)
        return allocate_raw(size);
    inside_allocator = true;
    void *ptr = allocate_raw(size);
    record_alloc(size);
    inside_allocator = false;
    return ptr;
}

void *operator new[](std::size_t size)
{
    return operator new(size);
}

void operator delete(void *ptr) noexcept
{
    if(ptr == nullptr)
        return;
    if(!inside_allocator)
    {
        inside_allocator = true;
        record_free(malloc_usable_size(ptr));
        inside_allocator = false;
    }
    free_raw(ptr);
}

void operator delete[](void *ptr) noexcept
{
    operator delete(ptr);
}

void operator delete(void *ptr, std::size_t) noexcept
{
    operator delete(ptr);
}

void operator delete[](void *ptr, std::size_t) noexcept
{
    operator delete(ptr);
}

void *operator new(std::size_t size, std::align_val_t alignment)
{
    if(size == 0)
        size = 1;
    void *ptr = nullptr;
    const std::size_t align = static_cast<std::size_t>(alignment);
    if(posix_memalign(&ptr, align, size) != 0)
        throw std::bad_alloc();
    if(!inside_allocator)
    {
        inside_allocator = true;
        record_alloc(size);
        inside_allocator = false;
    }
    return ptr;
}

void *operator new[](std::size_t size, std::align_val_t alignment)
{
    return operator new(size, alignment);
}

void operator delete(void *ptr, std::align_val_t) noexcept
{
    operator delete(ptr);
}

void operator delete[](void *ptr, std::align_val_t) noexcept
{
    operator delete(ptr);
}

void operator delete(void *ptr, std::size_t, std::align_val_t) noexcept
{
    operator delete(ptr);
}

void operator delete[](void *ptr, std::size_t, std::align_val_t) noexcept
{
    operator delete(ptr);
}

#endif // MEMORY_PROFILE
