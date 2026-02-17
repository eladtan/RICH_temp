#ifndef _DUPLICATIONS_H
#define _DUPLICATIONS_H

#ifdef RICH_MPI
#include <mpi.h>
#endif // RICH_MPI

#include <vector>

#define DUPLICATION_ERROR_CODE 2000

template<typename T>
void reportDuplications(const std::vector<T> &vector)
{
    for(size_t i = 0; i < vector.size(); i++)
    {
        for(size_t j = 0; j < vector.size(); j++)
        {
            if(i == j) continue;
            if(vector[i] == vector[j])
            {
                UniversalError eo("Duplication found");
                eo.addEntry("i", i);
                eo.addEntry("j", j);
                eo.addEntry("value", vector[i]);
                throw eo;
            }
        }
    }
}

#endif // _DUPLICATIONS_H