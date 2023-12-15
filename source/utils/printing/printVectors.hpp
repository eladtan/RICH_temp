#ifndef PRINT_VECTORS_HPP
#define PRINT_VECTORS_HPP

#include <ostream>
#include <vector>

#include "printTuples.hpp"

#define ELEMENTS_TO_PRINT_VECTOR 3

template<typename T>
std::ostream &operator<<(std::ostream &stream, const std::vector<T> &vector)
{
    if(vector.empty())
    {
        return stream << "{}";
    }

    stream << "{";
    size_t firstElementsToShow = std::min<size_t>(vector.size(), ELEMENTS_TO_PRINT_VECTOR);
    for(size_t i = 0; i < firstElementsToShow - 1; i++)
    {
        stream << vector[i] << ", ";
    }
    stream << vector[firstElementsToShow - 1];
    if(firstElementsToShow < vector.size())
    {
        stream << ", ... ,";
        size_t lastElementsToShow = std::min<size_t>(vector.size() - firstElementsToShow, ELEMENTS_TO_PRINT_VECTOR);
        for(size_t i = vector.size() - lastElementsToShow; i < vector.size() - 1; i++)
        {
            stream << vector[i] << ", ";
        }
        stream << vector[vector.size() - 1];
    }
    return stream << "}";
}

#endif // PRINT_VECTORS_HPP