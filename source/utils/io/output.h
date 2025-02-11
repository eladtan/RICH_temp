#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <iomanip>

#ifdef RICH_MPI
    #include <mpi.h>
    #include "mpi/Synchronize.h"
    #include "3D/elementary/Vector3D.hpp"
#endif // RICH_MPI

void printArray(int rank, std::vector<int> &data);

void printArraySize(int rank, std::vector<int> &data);

void printToFile(std::string fileName, std::vector<int> &vector);

#ifdef RICH_MPI
    void printAllArrays(std::vector<int> &myData);

    void printAllArraysSizes(std::vector<int> &myData);

    void printPointsDirectory(std::string dir_path, const std::vector<Vector3D> &points);
#endif // RICH_MPI