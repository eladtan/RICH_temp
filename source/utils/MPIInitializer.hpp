#ifndef MPI_INITIALIZER_HPP
#define MPI_INITIALIZER_HPP

#if defined(RICH_MPI) || defined(STORM_WITH_MPI)
#include <mpi.h>
#endif

// Finalizes MPI automatically at program exit. Unlike KokkosInitializer it never
// initializes MPI: the driver still calls MPI_Init, and the destructor calls
// MPI_Finalize only if MPI was initialized and has not been finalized yet (so an
// explicit MPI_Finalize in the driver remains valid).
//
// The single instance is a function-local static created by the first call to
// Instance() (see KokkosInitializer.hpp for why it is not a namespace-scope
// global or a variable in an anonymous namespace). Static objects are destroyed
// in reverse order of construction, so the object whose Instance() is first called
// earlier is destroyed later. Objects that own MPI resources (communicators,
// windows, requests) must be destroyed before this; objects created in main are
// destroyed when main returns, before it.
//
// Without MPI (neither RICH_MPI nor STORM_WITH_MPI) this is a no-op.
class MPIInitializer
{
public:
    static MPIInitializer &Instance()
    {
        static MPIInitializer instance;
        return instance;
    }

    MPIInitializer(const MPIInitializer &) = delete;
    MPIInitializer &operator=(const MPIInitializer &) = delete;

    ~MPIInitializer()
    {
#if defined(RICH_MPI) || defined(STORM_WITH_MPI)
        int initialized = 0;
        int finalized = 0;
        MPI_Initialized(&initialized);
        MPI_Finalized(&finalized);
        if(initialized and not finalized)
        {
            MPI_Finalize();
        }
#endif
    }

private:
    MPIInitializer() = default;
};

#endif // MPI_INITIALIZER_HPP
