#ifndef WEIGHTED_BALANCE3_HPP
#define WEIGHTED_BALANCE3_HPP

#ifdef RICH_MPI

#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <numeric>
#include <type_traits>
#include <vector>
#include <mpi.h>
#include "mpi/mpi_commands.hpp"
#include "misc/universal_error.hpp"

template<typename T>
struct WeightedBalance3Pair
{
    T value;
    double weight;
};

template<typename T>
struct WeightedBalance3OrderInfo
{
    int hasValue;
    T minValue;
    T maxValue;
};

template<typename T>
struct WeightedBalance3Candidate
{
    int order;
    T value;
};

template<typename T, typename Comparator>
std::vector<WeightedBalance3Pair<T>> makeWeightedBalance3Pairs(const std::vector<T> &values, const std::vector<double> &weights, const Comparator &comp)
{
    static_assert(std::is_trivially_copyable<T>::value, "weightedBalance3 requires trivially copyable values");

    if(!weights.empty() && values.size() != weights.size())
    {
        UniversalError eo("getWeightedBorders3: values and weights must have the same size");
        eo.addEntry("Values size", values.size());
        eo.addEntry("Weights size", weights.size());
        throw eo;
    }

    std::vector<WeightedBalance3Pair<T>> pairs;
    pairs.reserve(values.size());
    for(size_t i = 0; i < values.size(); i++)
    {
        pairs.push_back({values[i], weights.empty() ? 1.0 : weights[i]});
    }

    std::sort(pairs.begin(), pairs.end(), [&comp](const WeightedBalance3Pair<T> &lhs, const WeightedBalance3Pair<T> &rhs)
    {
        return comp(lhs.value, rhs.value);
    });
    return pairs;
}

template<typename T, typename Comparator>
bool weightedBalance3IsGloballyOrdered(const std::vector<WeightedBalance3Pair<T>> &localPairs, const Comparator &comp, const MPI_Comm &comm)
{
    rank_t size;
    MPI_Comm_size(comm, &size);

    WeightedBalance3OrderInfo<T> localInfo{};
    localInfo.hasValue = localPairs.empty() ? 0 : 1;
    if(localInfo.hasValue)
    {
        localInfo.minValue = localPairs.front().value;
        localInfo.maxValue = localPairs.back().value;
    }

    std::vector<WeightedBalance3OrderInfo<T>> allInfo(static_cast<size_t>(size));
    MPI_Allgather(&localInfo, sizeof(localInfo), MPI_BYTE,
                  allInfo.data(), sizeof(localInfo), MPI_BYTE, comm);

    bool seenValue = false;
    T previousMax{};
    for(const WeightedBalance3OrderInfo<T> &info : allInfo)
    {
        if(!info.hasValue)
        {
            continue;
        }

        if(seenValue && comp(info.minValue, previousMax))
        {
            return false;
        }
        previousMax = info.maxValue;
        seenValue = true;
    }
    return true;
}

template<typename T>
double weightedBalance3LocalWeight(const std::vector<WeightedBalance3Pair<T>> &pairs)
{
    return std::accumulate(pairs.cbegin(), pairs.cend(), 0.0,
        [](double acc, const WeightedBalance3Pair<T> &pair)
        {
            return acc + pair.weight;
        });
}

template<typename T, typename Comparator>
std::vector<T> weightedBalance3RootGather(std::vector<WeightedBalance3Pair<T>> localPairs, const Comparator &comp, const MPI_Comm &comm)
{
    rank_t rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    if(size <= 1)
    {
        return {};
    }

    int localBytes = static_cast<int>(localPairs.size() * sizeof(WeightedBalance3Pair<T>));
    std::vector<int> recvBytes, displacements;
    if(rank == 0)
    {
        recvBytes.resize(static_cast<size_t>(size));
        displacements.resize(static_cast<size_t>(size), 0);
    }

    MPI_Gather(&localBytes, 1, MPI_INT,
               rank == 0 ? recvBytes.data() : nullptr, 1, MPI_INT, 0, comm);

    std::vector<WeightedBalance3Pair<T>> allPairs;
    if(rank == 0)
    {
        int totalBytes = 0;
        for(rank_t r = 0; r < size; r++)
        {
            displacements[static_cast<size_t>(r)] = totalBytes;
            totalBytes += recvBytes[static_cast<size_t>(r)];
        }
        allPairs.resize(static_cast<size_t>(totalBytes) / sizeof(WeightedBalance3Pair<T>));
    }

    MPI_Gatherv(localPairs.data(), localBytes, MPI_BYTE,
                rank == 0 ? allPairs.data() : nullptr,
                rank == 0 ? recvBytes.data() : nullptr,
                rank == 0 ? displacements.data() : nullptr,
                MPI_BYTE, 0, comm);

    std::vector<T> borders(static_cast<size_t>(size - 1));
    if(rank == 0)
    {
        std::sort(allPairs.begin(), allPairs.end(), [&comp](const WeightedBalance3Pair<T> &lhs, const WeightedBalance3Pair<T> &rhs)
        {
            return comp(lhs.value, rhs.value);
        });

        double totalWeight = weightedBalance3LocalWeight(allPairs);
        if(totalWeight <= 0)
        {
            for(WeightedBalance3Pair<T> &pair : allPairs)
            {
                pair.weight = 1.0;
            }
            totalWeight = static_cast<double>(allPairs.size());
        }

        double currentWeight = 0;
        size_t pointIndex = 0;
        for(rank_t targetRank = 1; targetRank < size; targetRank++)
        {
            const double targetWeight = totalWeight * static_cast<double>(targetRank) / static_cast<double>(size);
            while(pointIndex < allPairs.size() && currentWeight + allPairs[pointIndex].weight <= targetWeight)
            {
                currentWeight += allPairs[pointIndex].weight;
                pointIndex++;
            }

            if(allPairs.empty())
            {
                borders[static_cast<size_t>(targetRank - 1)] = T{};
            }
            else if(pointIndex < allPairs.size())
            {
                borders[static_cast<size_t>(targetRank - 1)] = allPairs[pointIndex].value;
            }
            else
            {
                borders[static_cast<size_t>(targetRank - 1)] = allPairs.back().value + static_cast<T>(1);
            }
        }

        std::cout << "Weighted borders determined in 1 pass (root gather fallback)." << std::endl;
    }

    MPI_Bcast(borders.data(), static_cast<int>(borders.size() * sizeof(T)), MPI_BYTE, 0, comm);
    if(!borders.empty())
    {
        borders.push_back(borders.back() + static_cast<T>(1));
    }
    return borders;
}

template<typename T, typename Comparator>
std::vector<T> weightedBalance3Direct(std::vector<WeightedBalance3Pair<T>> localPairs, const Comparator &comp, const MPI_Comm &comm)
{
    (void) comp;

    rank_t rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    if(size <= 1)
    {
        return {};
    }

    double localWeight = weightedBalance3LocalWeight(localPairs);
    double totalWeight = 0;
    MPI_Allreduce(&localWeight, &totalWeight, 1, MPI_DOUBLE, MPI_SUM, comm);

    if(totalWeight <= 0)
    {
        for(WeightedBalance3Pair<T> &pair : localPairs)
        {
            pair.weight = 1.0;
        }
        localWeight = static_cast<double>(localPairs.size());
        MPI_Allreduce(&localWeight, &totalWeight, 1, MPI_DOUBLE, MPI_SUM, comm);
    }

    if(totalWeight <= 0)
    {
        return {};
    }

    double weightOffset = 0;
    MPI_Exscan(&localWeight, &weightOffset, 1, MPI_DOUBLE, MPI_SUM, comm);
    if(rank == 0)
    {
        weightOffset = 0;
    }
    const double localEndWeight = weightOffset + localWeight;

    std::vector<double> prefixWeights(localPairs.size());
    double currentWeight = 0;
    for(size_t i = 0; i < localPairs.size(); i++)
    {
        currentWeight += localPairs[i].weight;
        prefixWeights[i] = currentWeight;
    }

    std::vector<WeightedBalance3Candidate<T>> localCandidates;
    for(rank_t targetRank = 1; targetRank < size; targetRank++)
    {
        const double targetWeight = totalWeight * static_cast<double>(targetRank) / static_cast<double>(size);
        if(targetWeight < weightOffset || targetWeight >= localEndWeight || localPairs.empty())
        {
            continue;
        }

        const double localTarget = targetWeight - weightOffset;
        auto it = std::upper_bound(prefixWeights.cbegin(), prefixWeights.cend(), localTarget);
        const size_t index = std::min<size_t>(std::distance(prefixWeights.cbegin(), it), localPairs.size() - 1);
        localCandidates.push_back({targetRank, localPairs[index].value});
    }

    int sendBytes = static_cast<int>(localCandidates.size() * sizeof(WeightedBalance3Candidate<T>));
    std::vector<int> recvBytes(static_cast<size_t>(size));
    MPI_Allgather(&sendBytes, 1, MPI_INT, recvBytes.data(), 1, MPI_INT, comm);

    std::vector<int> displacements(static_cast<size_t>(size), 0);
    int totalBytes = 0;
    for(rank_t r = 0; r < size; r++)
    {
        displacements[static_cast<size_t>(r)] = totalBytes;
        totalBytes += recvBytes[static_cast<size_t>(r)];
    }

    std::vector<WeightedBalance3Candidate<T>> allCandidates(static_cast<size_t>(totalBytes) / sizeof(WeightedBalance3Candidate<T>));
    MPI_Allgatherv(localCandidates.data(), sendBytes, MPI_BYTE,
                   allCandidates.data(), recvBytes.data(), displacements.data(),
                   MPI_BYTE, comm);

    std::vector<T> borders(static_cast<size_t>(size - 1));
    std::vector<char> hasBorder(static_cast<size_t>(size - 1), 0);
    for(const WeightedBalance3Candidate<T> &candidate : allCandidates)
    {
        if(candidate.order <= 0 || candidate.order >= size)
        {
            continue;
        }

        const size_t index = static_cast<size_t>(candidate.order - 1);
        borders[index] = candidate.value;
        hasBorder[index] = 1;
    }

    const bool complete = std::all_of(hasBorder.cbegin(), hasBorder.cend(), [](char value){ return value != 0; });
    if(!complete)
    {
        return weightedBalance3RootGather(std::move(localPairs), comp, comm);
    }

    if(rank == 0)
    {
        std::cout << "Weighted borders determined in 1 pass." << std::endl;
    }

    if(!borders.empty())
    {
        borders.push_back(borders.back() + static_cast<T>(1));
    }
    return borders;
}

template<typename T, typename Comparator = std::function<bool(const T&, const T&)>>
std::vector<T> getWeightedBorders3(const std::vector<T> &values, const std::vector<double> &weights, const Comparator &comp = std::less<T>{}, const MPI_Comm &comm = MPI_COMM_WORLD)
{
    std::vector<WeightedBalance3Pair<T>> localPairs = makeWeightedBalance3Pairs(values, weights, comp);
    if(weightedBalance3IsGloballyOrdered(localPairs, comp, comm))
    {
        return weightedBalance3Direct(std::move(localPairs), comp, comm);
    }
    return weightedBalance3RootGather(std::move(localPairs), comp, comm);
}

template<typename T, typename Comparator = std::function<bool(const T&, const T&)>>
std::vector<T> getBorders3(const std::vector<T> &values, const Comparator &comp = std::less<T>{}, const MPI_Comm &comm = MPI_COMM_WORLD)
{
    return getWeightedBorders3(values, std::vector<double>(), comp, comm);
}

#endif // RICH_MPI

#endif // WEIGHTED_BALANCE3_HPP
