# VCL is x86-only (SSE/AVX intrinsics); other architectures use the scalar fallbacks
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    set(VCL_INCLUDE "${CMAKE_SOURCE_DIR}/opt/vcl")
endif()

# vcl
if(VCL_INCLUDE)
    message(STATUS "VCL directory ${VCL_INCLUDE}")
    include_directories("${VCL_INCLUDE}")
    add_compile_definitions("USE_VCL_VECTORIZATION")
else()
    list(FILTER RICH_SRC EXCLUDE REGEX "^${CMAKE_SOURCE_DIR}/opt/vcl/")
endif()

set(CLIPPER_INCLUDE "${CMAKE_SOURCE_DIR}/opt/clipper")

# clipper
if(CLIPPER_INCLUDE)
    message(STATUS "CLIPPER directory ${CLIPPER_INCLUDE}")
    include_directories("${CLIPPER_INCLUDE}")
    add_compile_definitions("USE_CLIPPER")
endif()
