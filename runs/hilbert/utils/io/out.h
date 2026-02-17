#ifndef _OUT_H
#define _OUT_H

#ifdef RICH_MPI

#include <iostream>
#include <mpi.h>
#include "3D/tessellation/voronoi/Voronoi3D.hpp"
#include "3D/output/write3D.hpp"

void printToHDF5(const Voronoi3D &voronoi, const std::string &fileName);

#endif // RICH_MPI

#endif // _OUT_H