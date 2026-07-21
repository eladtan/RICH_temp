#ifndef MPI_TYPES_H
#define MPI_TYPES_H

#include <type_traits> // for std::is_convertible
#include <vector> // for std::vector

using rank_t = int;

#ifdef RICH_MPI

template<typename Test, template<typename...> class Ref>
struct is_specialization : std::false_type {};

template<template<typename...> class Ref, typename... Args>
struct is_specialization<Ref<Args...>, Ref>: std::true_type {};

#endif // RICH_MPI

#endif // MPI_TYPES_H