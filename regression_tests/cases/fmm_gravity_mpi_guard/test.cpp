#include <fstream>
#include <iostream>
#include <string>

#include "source/misc/universal_error.hpp"
#include "source/newtonian/three_dimensional/FastMultipoleAcceleration3D.hpp"

int main()
{
    bool richMpi = false;
    bool constructorRejected = false;
    bool messageMatched = false;

#ifdef RICH_MPI
    richMpi = true;
    try
    {
        FastMultipoleAcceleration3D acceleration;
        (void) acceleration;
    }
    catch(UniversalError const& error)
    {
        constructorRejected = true;
        messageMatched = error.getErrorMessage().find("disabled in MPI builds") !=
            std::string::npos;
    }
#endif

    const bool passed = richMpi && constructorRejected && messageMatched;
    std::ofstream output("fmm_gravity_mpi_guard_metrics.txt");
    output << "rich_mpi " << (richMpi ? 1 : 0) << "\n";
    output << "constructor_rejected " << (constructorRejected ? 1 : 0) << "\n";
    output << "message_matched " << (messageMatched ? 1 : 0) << "\n";
    output << "pass " << (passed ? 1 : 0) << "\n";

    std::cout << "fmm_gravity_mpi_guard rich_mpi=" << richMpi
              << " constructor_rejected=" << constructorRejected
              << " message_matched=" << messageMatched
              << " pass=" << passed << std::endl;
    return passed ? 0 : 1;
}
