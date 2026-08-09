set(VCL_INCLUDE "${CMAKE_SOURCE_DIR}/opt/vcl")

# vcl
if(VCL_INCLUDE)
    message(STATUS "VCL directory ${VCL_INCLUDE}")
    include_directories("${VCL_INCLUDE}")
    add_compile_definitions("USE_VCL_VECTORIZATION")
endif()

set(CLIPPER_INCLUDE "${CMAKE_SOURCE_DIR}/opt/clipper")

# clipper
if(CLIPPER_INCLUDE)
    message(STATUS "CLIPPER directory ${CLIPPER_INCLUDE}")
    include_directories("${CLIPPER_INCLUDE}")
    add_compile_definitions("USE_CLIPPER")
endif()
