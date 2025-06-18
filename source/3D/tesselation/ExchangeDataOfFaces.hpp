#ifndef EXCHANGE_DATA_OF_FACES_HPP
#define EXCHANGE_DATA_OF_FACES_HPP

#ifdef RICH_MPI

#include "ExchangeFaces.hpp"

template<typename T>
boost::container::flat_map<size_t, T> ExchangeDataOfFaces(const Tessellation3D &tess, const std::vector<size_t> &facesList, const std::vector<T> &facesData)
{
    boost::container::flat_map<size_t, std::pair<rank_t, size_t>> faces_map = ExchangeFaces(tess, facesList);

    struct FaceData : public Serializable
    {
        size_t faceIdx;
        T data;

		force_inline size_t dump(Serializer *serializer) const override
		{
			size_t bytes = 0;
			bytes += serializer->insert(this->faceIdx);
			bytes += serializer->insert(this->data);
			return bytes;
		}

		force_inline size_t load(const Serializer *serializer, size_t byteOffset) override
		{
			size_t bytes = 0;
			bytes += serializer->extract(this->faceIdx, byteOffset);
			bytes += serializer->extract(this->data, byteOffset + bytes);
			return bytes;
		}
    };

    rank_t size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::vector<std::vector<FaceData>> outcomingFacesData(size);

    for(const auto &[faceIdx, faceInfo] : faces_map)
    {
        rank_t otherRank = faceInfo.first;
        std::vector<FaceData> &rankData = outcomingFacesData[otherRank];

        size_t faceIdxInRank = faceInfo.second;
        
        rankData.emplace_back();
        FaceData &faceData = rankData.back();
        faceData.faceIdx = faceIdxInRank;
        faceData.data = facesData[faceIdx];
    }

    std::vector<std::vector<FaceData>> incomingFacesData = MPI_Exchange_all_to_all(outcomingFacesData, MPI_COMM_WORLD);
    for(rank_t otherRank = 0; otherRank < size; otherRank++)
    {
        for(const FaceData &faceData : incomingFacesData[otherRank])
        {
            if(faces_data.find(faceData.faceIdx) != faces_data.end())
            {
                UniversalError eo("Duplicate face data received");
                eo.addEntry("Face Index", faceData.faceIdx);
                eo.addEntry("Rank", otherRank);
                throw eo;
            }
            faces_data.insert({faceData.faceIdx, {faceData.data}});
        }
    }
    return faces_data;
}

#endif // RICH_MPI

#endif // EXCHANGE_DATA_OF_FACES_HPP