// #ifndef TESSELLATION3D_MPI_EXCHANGE_HPP
// #define TESSELLATION3D_MPI_EXCHANGE_HPP

// #ifdef RICH_MPI

// #include "mpi/mpi_exchange_commands.hpp"
// #include "3D/tessellation/Tessellation3D.hpp"

// template<typename T>
// void MPI_Tessellation_movement_exchange_data(const Tessellation3D &tess, std::vector<T> &data)
// {
//     // TODO: exchange all
//     const Tessellation3D::AllPointsMap &indexTranslationMap = tess.GetIndicesInAllPoints();
//     auto indexTranslation = [&indexTranslationMap](const size_t &index){return indexTranslationMap.at(index);};
//     std::vector<T> dataToMe;
//     for(const size_t &index : tess.GetSelfIndex())
//     {
//         dataToMe.push_back(data[indexTranslation(index)]);
//     }
//     data = std::move(MPI_Exchange_data_indices<T, size_t>(data, tess.GetSentProcs(), tess.GetSentPoints(), indexTranslation));
//     dataToMe.insert(dataToMe.end(), data.begin(), data.end());
// }

// template<typename T>
// void MPI_Tessellation_ghost_exchange_data(const Tessellation3D &tess, std::vector<T> &data, const T &defaultValue = T())
// {
//     const Tessellation3D::AllPointsMap &indexTranslationMap = tess.GetIndicesInAllPoints();
//     auto indexTranslation = [&indexTranslationMap](const size_t &index){return indexTranslationMap.at(index);};
//     std::vector<std::vector<T>> incoming = MPI_Exchange_data_seperate_indices<T, size_t>(data, tess.GetDuplicatedProcs(), tess.GetDuplicatedPoints(), indexTranslation);
//     data.resize(tess.GetTotalPointNumber(), defaultValue);
//     size_t size = incoming.size();
//     const std::vector<std::vector<size_t>> &ghost_indices = tess.GetGhostIndeces();
//     for(size_t i = 0; i < size; i++)
//     {
//         const std::vector<size_t> &indices = ghost_indices[i];
//         size_t incomingSize = indices.size();
//         for(size_t j = 0; j < incomingSize; j++)
//         {
//             data[indices[j]] = incoming[i][j];
//         }
//     }
// }

// #endif // RICH_MPI

// #endif // TESSELLATION3D_MPI_EXCHANGE_HPP