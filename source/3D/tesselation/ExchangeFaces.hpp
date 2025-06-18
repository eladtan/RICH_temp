#ifndef EXCHANGE_FACES_HPP
#define EXCHANGE_FACES_HPP

#ifdef RICH_MPI

#include "ExchangeGhosts.hpp"

boost::container::flat_map<size_t, std::pair<rank_t, size_t>> ExchangeFaces(const Tessellation3D &tess, const std::vector<size_t> &facesList);

#endif // RICH_MPI

#endif // EXCHANGE_FACES_HPP