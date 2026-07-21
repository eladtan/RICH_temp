# Get Catch2 from github
include(FetchContent)
FetchContent_Declare(
catch2
GIT_REPOSITORY https://github.com/catchorg/Catch2.git
GIT_TAG v3.11.0
)
FetchContent_MakeAvailable(catch2)

message(STATUS "Fetching Catch2")
message(STATUS "Catch2 at ${catch2_SOURCE_DIR}")

set(TESTS_DIR "${PROJECT_SOURCE_DIR}/../tests_catch2")
message(STATUS "TESTS_DIR = ${TESTS_DIR}")
file(GLOB_RECURSE TESTS_SOURCES "${TESTS_DIR}/*.cpp")

message(STATUS "Test sources:")
foreach(file IN LISTS TESTS_SOURCES)
    message(STATUS "  ${file}")
endforeach()

if (TESTS_SOURCES)
    add_executable(rich_tests ${TESTS_SOURCES})
    target_link_libraries(rich_tests PRIVATE rich_core Catch2::Catch2)
    target_include_directories(rich_tests PRIVATE "${PROJECT_SOURCE_DIR}" "${PROJECT_SOURCE_DIR}/..")

    include(Catch)
    set(CMAKE_CATCH_DISCOVER_TESTS_DISCOVERY_MODE PRE_TEST)
    catch_discover_tests(rich_tests)
else()
    message(STATUS "No test sources found")
endif()
