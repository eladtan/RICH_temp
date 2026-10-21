#ifndef KOKKOS_INITIALIZER_HPP
#define KOKKOS_INITIALIZER_HPP

#include <Kokkos_Core.hpp>

// RICH's Kokkos initializer: initializes Kokkos when RICH first needs it (the
// Simulation constructor, in builds with RICH_WITH_KOKKOS) and finalizes it
// automatically at program exit, so drivers never call Kokkos::initialize/finalize.
//
// Libraries that use Kokkos (e.g. STORM::gpu::KokkosInitializer) have their own
// initializer. Whichever runs first starts Kokkos and owns it; the others find it
// already initialized and never finalize it. Static objects are destroyed in reverse
// order of construction, so the owner, created first, is destroyed after every other
// initializer.
//
// The single instance is a function-local static created on first use:
//  - Kokkos registers its execution spaces from its own static initializers. A
//    namespace-scope global could run Kokkos::initialize before those, since the
//    order of static initialization across translation units is unspecified.
//  - Static objects and atexit handlers are destroyed in reverse order of
//    construction. Kokkos::initialize starts the device runtime (CUDA/HIP), which
//    registers its exit handlers while this constructor runs, so this destructor
//    runs before them, while the device runtime is still usable. Objects created
//    in main are destroyed before this, when main returns.
//  - A local static of an inline function is one object for the whole program; a
//    variable in an anonymous namespace would be duplicated in every file that
//    includes this header.
class KokkosInitializer
{
public:
    static KokkosInitializer &Instance()
    {
        static KokkosInitializer instance;
        return instance;
    }

    // Whether Instance() has been created, without creating it.
    static bool Exists()
    {
        return Created();
    }

    KokkosInitializer(const KokkosInitializer &) = delete;
    KokkosInitializer &operator=(const KokkosInitializer &) = delete;

    ~KokkosInitializer()
    {
        this->Finalize();
    }

    // Finalize before program exit, for setups that need Kokkos torn down earlier
    // (e.g. before MPI_Finalize when HIP shares resources with an OFI context).
    // Kokkos cannot be initialized again afterwards. Safe to call repeatedly.
    void Finalize()
    {
        if(this->initializedHere_ and Kokkos::is_initialized() and not Kokkos::is_finalized())
        {
            Kokkos::finalize();
        }
        this->initializedHere_ = false;
    }

private:
    KokkosInitializer()
    {
        Created() = true;
        if(not Kokkos::is_initialized() and not Kokkos::is_finalized())
        {
            Kokkos::initialize();
            this->initializedHere_ = true;
        }
    }

    static bool &Created()
    {
        static bool created = false;
        return created;
    }

    bool initializedHere_ = false;
};

#endif // KOKKOS_INITIALIZER_HPP
