#ifndef MEMORY_PROFILE_HPP
#define MEMORY_PROFILE_HPP

#include <cstddef>

#ifdef MEMORY_PROFILE
#include <chrono>

namespace memory_profile
{
    struct Snapshot
    {
        std::size_t alloc_count = 0;
        std::size_t alloc_bytes = 0;
        std::size_t free_count = 0;
        std::size_t free_bytes = 0;
        std::size_t alloc_bins[8] = {};
        std::size_t free_bins[8] = {};
    };

    Snapshot snapshot();
    void print_scope(const char *label, const Snapshot &before, double seconds);

    class Scope
    {
    public:
        explicit Scope(const char *label);
        ~Scope();

    private:
        const char *label_;
        Snapshot before_;
        std::chrono::high_resolution_clock::time_point start_;
    };
}

#define MEMORY_PROFILE_CONCAT_IMPL(a, b) a##b
#define MEMORY_PROFILE_CONCAT(a, b) MEMORY_PROFILE_CONCAT_IMPL(a, b)
#define MEMORY_PROFILE_SCOPE(label) memory_profile::Scope MEMORY_PROFILE_CONCAT(memory_profile_scope_, __LINE__)(label)

#else

#define MEMORY_PROFILE_SCOPE(label) ((void)0)

#endif // MEMORY_PROFILE

#endif // MEMORY_PROFILE_HPP
