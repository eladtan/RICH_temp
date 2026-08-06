#include <fstream>
#include <iostream>

#ifdef RICH_MPI
#include <mpi.h>
#endif

#include "source/newtonian/three_dimensional/FastMultipoleAcceleration3D.hpp"

int main(int argc, char** argv)
{
#ifndef RICH_MPI
    (void) argc;
    (void) argv;
    return 1;
#else
    MPI_Init(&argc, &argv);
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int accepted = 0;
    int potentialRejected = 0;
    {
        try
        {
            FastMultipoleAcceleration3D acceleration;
            accepted = 1;
        }
        catch(...)
        {
            accepted = 0;
        }
    }
    try
    {
        FmmGravityOptions unsupported;
        unsupported.computePotential = true;
        FastMultipoleAcceleration3D acceleration(unsupported);
        (void) acceleration;
    }
    catch(...)
    {
        potentialRejected = 1;
    }

    int globallyAccepted = 0;
    MPI_Allreduce(&accepted, &globallyAccepted, 1, MPI_INT, MPI_LAND,
                  MPI_COMM_WORLD);
    int globallyRejected = 0;
    MPI_Allreduce(&potentialRejected, &globallyRejected, 1, MPI_INT, MPI_LAND,
                  MPI_COMM_WORLD);
    const int passed = globallyAccepted && globallyRejected;
    if(rank == 0)
    {
        std::ofstream output("fmm_gravity_mpi_guard_metrics.txt");
        output << "rich_mpi 1\n";
        output << "constructor_accepted " << globallyAccepted << "\n";
        output << "potential_option_rejected " << globallyRejected << "\n";
        output << "pass " << passed << "\n";
        std::cout << "fmm_gravity_mpi_guard constructor_accepted="
                  << globallyAccepted << " potential_option_rejected="
                  << globallyRejected << std::endl;
    }
    MPI_Finalize();
    return passed ? 0 : 1;
#endif
}
