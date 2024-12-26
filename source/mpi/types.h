#ifndef MPI_TYPES_H
#define MPI_TYPES_H

#ifdef RICH_MPI

using rank_t = int;

template<typename Test, template<typename...> class Ref>
struct is_specialization : std::false_type {};

template<template<typename...> class Ref, typename... Args>
struct is_specialization<Ref<Args...>, Ref>: std::true_type {};

template<typename T>
using is_vector = is_specialization<T, std::vector>;

#endif // RICH_MPI

#endif // MPI_TYPES_H