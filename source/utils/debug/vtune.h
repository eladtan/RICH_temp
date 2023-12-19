#ifdef WITH_VTUNE
    #include <ittnotify.h>
#endif // WITH_VTUNE
#include "misc/universal_error.hpp"

inline void vtune_pause(void)
{
    #ifdef WITH_VTUNE
        __itt_pause();
    #else // WITH_VTUNE
        throw UniversalError("Compiled without Vtune");
    #endif // WITH_VTUNE
}

inline void vtune_resume(void)
{
    #ifdef WITH_VTUNE
        __itt_resume();
    #else // WITH_VTUNE
        throw UniversalError("Compiled without Vtune");
    #endif // WITH_VTUNE
}

inline void vtune_stop(void)
{
    vtune_pause();
}

inline void vtune_start(void)
{
    vtune_resume();
}
