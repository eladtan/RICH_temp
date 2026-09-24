cmake_minimum_required(VERSION 3.14)
project(RichMPIRMARegression LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(MPI 3.0 REQUIRED COMPONENTS CXX)
get_filename_component(RICH_ROOT "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)

add_executable(mpi_rma_agent agent.cpp)
target_compile_definitions(mpi_rma_agent PRIVATE __WITH_MPI OMPI_SKIP_MPICXX)
target_include_directories(mpi_rma_agent PRIVATE "${RICH_ROOT}/source/utils")
target_link_libraries(mpi_rma_agent PRIVATE MPI::MPI_CXX)

# Build the existing benchmark without writing generated files or binaries
# into STORM. Regression defaults keep native backends disabled.
option(BUILD_CROOKED_PIPE "Build STORM's unchanged Crooked Pipe benchmark" ON)
option(ENABLE_OFI_BACKEND "Enable OFI for backend timing comparisons" OFF)
option(AUDIT_MPI_RMA "Count MPI RMA calls through PMPI (disable for timing)" ON)
option(BUILD_MARSHAK "Build the external parallel Marshak comparison driver" OFF)
if(BUILD_CROOKED_PIPE)
  set(STORM_WITH_MPI ON CACHE BOOL "" FORCE)
  set(STORM_NO_IBV ON CACHE BOOL "" FORCE)
  if(ENABLE_OFI_BACKEND)
    set(STORM_NO_OFI OFF CACHE BOOL "" FORCE)
  else()
    set(STORM_NO_OFI ON CACHE BOOL "" FORCE)
  endif()
  set(STORM_WITH_COMPTON OFF CACHE BOOL "" FORCE)
  set(STORM_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
  set(STORM_DEPS_DIR "${RICH_ROOT}/source/utils")
  set(STORM_MADVORO_DIR "${RICH_ROOT}/source/3D/tessellation/voronoi")
  set(STORM_MADCART_DIR "${RICH_ROOT}/source/3D/tessellation/cartesian")
  set(STORM_MESH_DECOMPOSER_DIR "${RICH_ROOT}/source/3D/tessellation/MeshDecomposer3D")
  add_subdirectory("${RICH_ROOT}/source/monte" storm)
  get_target_property(storm_definitions storm INTERFACE_COMPILE_DEFINITIONS)
  if(ENABLE_OFI_BACKEND AND NOT "__WITH_OFI" IN_LIST storm_definitions)
    message(FATAL_ERROR "The requested OFI backend was not found")
  endif()
  file(GLOB exception_sources "${STORM_MADVORO_DIR}/exception/*.cpp")
  add_executable(crooked_pipe_mpi_rma
    "${RICH_ROOT}/source/monte/examples/crooked_pipe/main.cpp"
    ${exception_sources})
  if(AUDIT_MPI_RMA)
    target_sources(crooked_pipe_mpi_rma PRIVATE mpi_audit.cpp)
  endif()
  target_include_directories(crooked_pipe_mpi_rma PRIVATE
    "${RICH_ROOT}/source/utils"
    "${RICH_ROOT}/source/monte/deps/CMMC/src")
  target_compile_definitions(crooked_pipe_mpi_rma PRIVATE OMPI_SKIP_MPICXX)
  target_link_libraries(crooked_pipe_mpi_rma PRIVATE storm)

  if(BUILD_MARSHAK)
    add_executable(marshak_mpi_rma marshak/main.cpp ${exception_sources})
    target_include_directories(marshak_mpi_rma PRIVATE
      "${RICH_ROOT}/source/utils" "${RICH_ROOT}/source/monte/deps/CMMC/src")
    target_compile_definitions(marshak_mpi_rma PRIVATE OMPI_SKIP_MPICXX
      MARSHAK_REFERENCE="${RICH_ROOT}/source/monte/examples/marshak_wave_2/reference.txt")
    target_link_libraries(marshak_mpi_rma PRIVATE storm)
  endif()

  add_executable(reallocation_cleanup reallocation_cleanup.cpp)
  target_include_directories(reallocation_cleanup PRIVATE "${RICH_ROOT}/source/utils")
  target_link_libraries(reallocation_cleanup PRIVATE storm)
endif()

enable_testing()
add_test(NAME mpi_rma_agent
  COMMAND "${MPIEXEC_EXECUTABLE}" ${MPIEXEC_NUMPROC_FLAG} 2
    ${MPIEXEC_PREFLAGS} $<TARGET_FILE:mpi_rma_agent> ${MPIEXEC_POSTFLAGS})
set_tests_properties(mpi_rma_agent PROPERTIES TIMEOUT 60 PROCESSORS 2)
if(BUILD_CROOKED_PIPE)
  add_test(NAME reallocation_cleanup
    COMMAND "${MPIEXEC_EXECUTABLE}" ${MPIEXEC_NUMPROC_FLAG} 2
      ${MPIEXEC_PREFLAGS} $<TARGET_FILE:reallocation_cleanup> ${MPIEXEC_POSTFLAGS})
  set_tests_properties(reallocation_cleanup PROPERTIES TIMEOUT 60 PROCESSORS 2)
endif()
