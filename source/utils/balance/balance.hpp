/**
 * \author Maor Mizrachi
 * \brief This file implements MPI balance function: each rank sends a list of elements (and a comparator function, determining the requested order
 * between the elements), and the procedure returns a list of "borders", which are elements that should be used to divide all the elements, so
 * that after sorting according to the borders, the elements are roughly equally distributed.
*/
#ifndef _RICH_BALANCE2_HPP
#define _RICH_BALANCE2_HPP

#include <vector>
#include <algorithm>
#include <functional>
#include <assert.h>
#include <mpi.h>

namespace
{
    template<typename T, typename Comparator = std::function<bool(const T&, const T&)>>
    std::pair<T, int> getMedianOfMedians(bool hasValue, const T &value, const Comparator &comp)
    {
        int rank, size;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);

        std::vector<T> medians(size);
        MPI_Allgather(&value, sizeof(T), MPI_BYTE, &medians[0], sizeof(T), MPI_BYTE, MPI_COMM_WORLD);

        std::vector<int> haveValues(size);
        int hasValueToSend = (hasValue)? 1 : 0;
        MPI_Allgather(&hasValueToSend, 1, MPI_INT, &haveValues[0], 1, MPI_INT, MPI_COMM_WORLD);

        std::vector<std::pair<T, int>> mediansByRanks;

        for(int i = 0; i < size; i++)
        {
            if(haveValues[i])
            {
                mediansByRanks.push_back(std::make_pair(medians[i], i));
            }
        }

        if(mediansByRanks.empty())
        {
            throw UniversalError("The requested stat orders are too high");
        }

        auto newComp = [comp](const std::pair<T, int> &lhs, const std::pair<T, int> &rhs)
                        {
                            if(lhs.first == rhs.first) return lhs.second < rhs.second;
                            else return comp(lhs.first, rhs.first);
                        }; 

        std::sort(mediansByRanks.begin(), mediansByRanks.end(), newComp);
        return mediansByRanks[mediansByRanks.size() / 2];
    }

    template<typename T, typename Comparator = std::function<bool(const T&, const T&)>>
    void findOrderStatistics(const typename std::vector<T>::iterator &vectorFirst, const typename std::vector<T>::iterator &vectorLast, const typename std::vector<size_t>::iterator &statsFirst, const typename std::vector<size_t>::iterator &statsLast, std::vector<T> &result, const Comparator &comp, size_t statOffset)
    {
        // MPI_Barrier(MPI_COMM_WORLD);

        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        if(statsFirst == statsLast)
        {
            return;
        }
        typename std::vector<size_t>::difference_type statMidLen = (statsLast - statsFirst) / 2;
        typename std::vector<T>::difference_type vecMidLen = (vectorLast - vectorFirst) / 2;

        // Consider the statistical order requested (the mid of the stats array)
        // We also consider the median of medians of the vectors, and calculate its stat order, call it M (that is, the median of medians is the M-th element in size, taking into account all the vectors).
        // if S = M, then the median of medians is the S stat order.
        // if S > M, then the M'th smallest element is on the left (when 'left' means the part of the array, which is below the median of medians).
        // continue looking it, and all the requested orders below M, in the left sides.
        // if S < M, then the M'th smallest element is on the right sides. But, we have to normalize the statistic orders (because there are elements which are smaller, and ignored)
        // pay attention that the 'left' and 'right' sides should be carefully calculated

        size_t statRequested = *(statsFirst + statMidLen) - statOffset;
        std::pair<T, int> medianOfMedians;
        if(vectorFirst == vectorLast)
        {
            // has no values...
            medianOfMedians = getMedianOfMedians(false, T(), comp);
        }
        else
        {
            medianOfMedians = getMedianOfMedians(true, *(vectorFirst + vecMidLen), comp);
        }
        auto newComp = [comp, rank, medianOfMedians](const T &lhs, const T &rhs)
                    {
                        if(lhs == rhs) return (medianOfMedians.second != rank);
                        return comp(lhs, rhs);
                    }; 
        typename std::vector<T>::iterator vectorBeforeMid = std::lower_bound(vectorFirst, vectorLast, medianOfMedians.first, newComp);
        if((vectorLast - vectorFirst > 1) and (medianOfMedians.second == rank) and (vectorBeforeMid == vectorFirst))
        {
            // edge case
            if(vectorLast - vectorFirst == 2)
            {
                vectorBeforeMid = vectorFirst + 1; // ONE BEFORE LAST (otherwise we stay in the same vectorFirst, vectorLast)
            }
            else
            {
                vectorBeforeMid = vectorFirst + vecMidLen; // MIDDLE
            }
        }
        size_t numBeforeMid = static_cast<size_t>(vectorBeforeMid - vectorFirst);
        size_t statOrderOfMedOfMed;
        MPI_Allreduce(&numBeforeMid, &statOrderOfMedOfMed, 1, MPI_UNSIGNED_LONG, MPI_SUM, MPI_COMM_WORLD);
        std::vector<size_t>::iterator statBeforeMid = std::lower_bound(statsFirst, statsLast, statOffset + statOrderOfMedOfMed);

        // we found a requested order!
        if(statOrderOfMedOfMed == statRequested)
        {
            result.push_back(medianOfMedians.first);
        }   
        // now, if the left side contains a correct stat-order, and it belongs to me, cut the right part so it won't include it:
        // if we have just found a stat-order, 'remove' it from the requested to find
        std::vector<size_t>::iterator rightStatsBegin = (statOrderOfMedOfMed == statRequested)? statBeforeMid + 1 : statBeforeMid;
        int rightNewOffset = statOffset + statOrderOfMedOfMed + (rightStatsBegin - statBeforeMid);
        // recurse to right:
        findOrderStatistics(vectorBeforeMid, vectorLast, rightStatsBegin, statsLast, result, comp, rightNewOffset);
        // recurse to left:
        findOrderStatistics(vectorFirst, vectorBeforeMid, statsFirst, statBeforeMid, result, comp, statOffset);
    }

    template<typename T, typename Comparator = std::function<bool(const T&, const T&)>>
    std::vector<T> getStatOrders(std::vector<T> &input, std::vector<size_t> &orders, const Comparator &comp = [](const T &a, const T &b){return a < b;})
    {
        std::sort(input.begin(), input.end(), comp);
        std::sort(orders.begin(), orders.end());
        std::vector<T> results;
        findOrderStatistics(input.begin(), input.end(), orders.begin(), orders.end(), results, comp, 0);
        std::sort(results.begin(), results.end());
        return results;
    }
}

/**
 * gets an input vector, and a comparator (an optional argument). Returns a borders vector, which represents the elements that should be put as separators,
 * in order that each one of the ranks will have a equally-sized input after rearranging.
*/
template<typename T, typename Comparator = std::function<bool(const T&, const T&)>>
std::vector<T> getBorders(std::vector<T> &input, const Comparator &comp = [](const T &a, const T &b){return a < b;})
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // stage 0: calculate the order statistics of the necessary bounds

    std::vector<size_t> currLengths(size);
    size_t mySize = input.size();
    size_t totalSize = 0;
    MPI_Allreduce(&mySize, &totalSize, 1, MPI_UNSIGNED_LONG, MPI_SUM, MPI_COMM_WORLD);

    if(static_cast<size_t>(size) > totalSize)
    {
        throw UniversalError("Too many ranks were given compared to the number of points");
    }
    std::vector<size_t> stats(size);
    for(int i = 0; i < size - 1; i++)
    {
        stats[i] = (totalSize / size) * (i + 1);
    }
    stats[size - 1] = totalSize - 1;

    return getStatOrders(input, stats, comp);
}


#endif // _RICH_BALANCE2_HPP
