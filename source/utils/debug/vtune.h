#ifdef WITH_VTUNE
    #include <ittnotify.h>
#endif // WITH_VTUNE
#include "misc/universal_error.hpp"

void vtune_pause(void);

void vtune_resume(void);

void vtune_stop(void);

void vtune_start(void);