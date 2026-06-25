set(CMAKE_CXX_STANDARD 17)

if(DEFINED MPI)
    add_definitions("-DRICH_MPI")
    add_definitions("-D__WITH_MPI")
    message(STATUS "Defined 'RICH_MPI' and '__WITH_MPI'")
endif()

set(CMAKE_CXX_FLAGS "")
set(CMAKE_Fortran_FLAGS "")

set(CXX_WARNING_FLAGS "")
list(APPEND CXX_WARNING_FLAGS
    "-Wextra"
    "-Wshadow"
    "-Wunused-value"
    "-Wunused-variable"
    "-Wunused-function"
    "-Wunused-macros"
    "-Werror=return-type")

list(APPEND CMAKE_CXX_FLAGS ${CXX_WARNING_FLAGS})
list(APPEND CMAKE_CXX_FLAGS
    "-g3"
    "-fno-omit-frame-pointer"
    "-fno-optimize-sibling-calls"
    "-fno-common"
    "-fstack-protector-all")

# energy groups
if(NOT DEFINED ENERGY_GROUPS_NUM)
    set(ENERGY_GROUPS_NUM 1)
endif()

add_definitions("-DENERGY_GROUPS_NUM=${ENERGY_GROUPS_NUM}")

if(DEFINED HIGH_RES)
    message(STATUS "High-resolution mode enabled")
    add_definitions("-DHIGH_RES")
endif()

if(DEFINED ASAN)
    message(STATUS "Address Sanitizer Enabled")
    list(APPEND CMAKE_CXX_FLAGS "-fsanitize=address,undefined,bounds" "-shared-libasan")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=address,undefined,bounds -shared-libasan")
endif()

if(DEFINED MC_DEBUG)
    # status message
    message(STATUS "Monte Carlo Debug Mode Enabled")
    add_definitions("-DSTORM_DEBUG")
    message(STATUS "Defined 'STORM_DEBUG'")
endif()

if(DEFINED MC_TRACE_DEBUG)
    message(STATUS "Monte Carlo Trace Debug Enabled with history size ${MC_TRACE_DEBUG}")
    add_definitions("-DSTORM_WITH_TRACING_HISTORY=${MC_TRACE_DEBUG}")
endif()

if(DEFINED MEMORY_DEBUG)
    message(STATUS "Memory Debug Mode Enabled")
    add_definitions("-DMEMORY_DEBUG")
endif()

if(DEFINED TIMING)
    message(STATUS "Timing Mode Enabled")
    add_definitions("-DTIMING")
endif()

# if build is Debug
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_definitions("-DDEBUG")
    message(STATUS "Build Type: Debug")
else()
    if(NOT DEFINED FORCE_ASSERT)
        add_definitions("-DNDEBUG")
    else()
        message(STATUS "Assertions forced on (FORCE_ASSERT)")
    endif()
    add_definitions("-DOMPI_SKIP_MPICXX")
endif()

list(APPEND CMAKE_CXX_FLAGS_RELEASE "-O2")
list(APPEND CMAKE_CXX_FLAGS_DEBUG
        "-O0"
        "-gdwarf-3"
        "-D_GLIBCXX_ASSERTIONS" # "-D_GLIBCXX_DEBUG"
        "-DDEBUG")

if(DEFINED GNU)
    list(APPEND CMAKE_CXX_FLAGS
            "-static-libgcc"
            "-static-libstdc++"
            "-Wl,-Bstatic -lgomp -Wl,-Bdynamic"
            "-Wdouble-promotion"
            "-rdynamic"
            "-fstrict-aliasing"
            "-Wno-deprecated-copy"
            "-Wno-double-promotion"
            "-Wno-shadow")

    list(APPEND CMAKE_Fortran_FLAGS
            "-fall-intrinsics"
            "-std=f2018"
            "-fdec-static"
            "-finit-local-zero"
            "-finit-integer=-2147483647"
            "-finit-real=snan"
            "-finit-logical=True"
            "-finit-derived"
            "-ffpe-trap=invalid,zero,overflow"
            "-ffree-line-length-none"
            "-cpp"
            "-fdefault-real-8"
            "-fdefault-double-8"
            "-fbacktrace"
            "-g"
            "-Wall"
            "-Wextra"
            "-Wsurprising"
            "-Wpedantic"
            "-Wno-uninitialized"
            "-mcmodel=medium")
       
    list(APPEND CMAKE_Fortran_FLAGS_DEBUG
            "-O0"
            "-fcheck=all"
            "-Wno-maybe-uninitialized"
            "-Wno-tabs"
            "-Wno-conversion")
    list(APPEND CMAKE_Fortran_FLAGS_RELEASE "-O2")
endif()

if(DEFINED INTEL)
    list(APPEND CMAKE_Fortran_FLAGS
            "-init=arrays,zero,minus_huge,snan"
            "-fp-speculation=safe"
            "-r8"
            "-assume no2underscores"
            "-lstdc++"
            "-lrt"
            "-traceback"
            "-fpe0"
            "-gen-interfaces"
            "-warn all"
            "-warn errors"
            "-mcmodel=medium")

    list(APPEND CMAKE_Fortran_FLAGS_DEBUG
            "-O0"
            "-fp-model precise"
            "-fp-model source"
            "-fimf-arch-consistency=true"
            "-fp-stack-check"
            "-debug"
            "-check bounds"
            "-check format"
            "-check output_conversion"
            "-check pointers"
            "-check uninit"
            "-check stack"
            "-check shape"
            "-mcmodel=medium"
            "-shared-intel")
        
        list(APPEND CMAKE_Fortran_FLAGS_RELEASE "-O2")

        list(APPEND CMAKE_CXX_FLAGS
                "-diag-remark=13397,13401,15552,2196"
                "-Wall"
                "-qopenmp"
                "-ansi-alias"
                "-fimf-arch-consistency=true")

        list(APPEND CMAKE_CXX_FLAGS_DEBUG
                "-fp-model consistent"
                "-diag-disable=openmp"
                "-Wno-unknown-pragmas")
        
        list(APPEND CMAKE_CXX_FLAGS_RELEASE
                "-fp-model precise"
                "-march=core-avx2"
                "-qopenmp")
endif()

list(JOIN CMAKE_CXX_FLAGS_RELEASE " " CMAKE_CXX_FLAGS_RELEASE)
list(JOIN CMAKE_CXX_FLAGS_DEBUG " " CMAKE_CXX_FLAGS_DEBUG)
list(JOIN CMAKE_CXX_FLAGS " " CMAKE_CXX_FLAGS)
list(JOIN CMAKE_Fortran_FLAGS_RELEASE " " CMAKE_Fortran_FLAGS_RELEASE)
list(JOIN CMAKE_Fortran_FLAGS_DEBUG " " CMAKE_Fortran_FLAGS_DEBUG)
list(JOIN CMAKE_Fortran_FLAGS " " CMAKE_Fortran_FLAGS)