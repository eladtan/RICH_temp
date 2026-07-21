#include "vtune.h"

void vtune_pause(void)
{
    #ifdef WITH_VTUNE
        __itt_pause();
    #else // WITH_VTUNE
        throw UniversalError("Compiled without Vtune");
    #endif // WITH_VTUNE
}

void vtune_resume(void)
{
    #ifdef WITH_VTUNE
        __itt_resume();
    #else // WITH_VTUNE
        throw UniversalError("Compiled without Vtune");
    #endif // WITH_VTUNE
}

void vtune_stop(void)
{
    vtune_pause();
}

void vtune_start(void)
{
    vtune_resume();
}
