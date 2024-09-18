#ifndef TIME_REQUEST_DATA_HPP
#define TIME_REQUEST_DATA_HPP

#include "misc/serializable.hpp"
#include "TimingTree.hpp"

#ifdef RICH_MPI

template<typename T>
class TimeRequestData : public Serializable2
{
public:
    using NodeData = TimingTree<Vector3D>::NodeData;

    explicit inline TimeRequestData(const BoundingBox<T> &boundingBox_ = BoundingBox<T>(), size_t cellID_ = std::numeric_limits<size_t>::max(), const NodeData &value_ = NodeData()):
        boundingBox(boundingBox_), cellID(cellID_), value(value_)
    {}

    size_t dump(Serializer *serializer) const override;
    
    size_t load(Serializer *serializer, size_t byteOffset) override;

    BoundingBox<T> boundingBox;
    size_t cellID;
    NodeData value;
};

template<typename T>
size_t TimeRequestData<T>::dump(Serializer *serializer) const
{
    size_t bytes = 0;
    bytes += this->boundingBox.dump(serializer);
    bytes += serializer->insert(this->cellID);
    bytes += this->value.dump(serializer);
    return bytes;
}

template<typename T>
inline size_t TimeRequestData<T>::load(Serializer *serializer, size_t byteOffset)
{
    size_t bytesRead = 0;
    bytesRead += this->boundingBox.load(serializer, byteOffset);
    bytesRead += serializer->extract(this->cellID, byteOffset + bytesRead);
    bytesRead += this->value.load(serializer, byteOffset + bytesRead);
    return bytesRead;
}
    
#endif // RICH_MPI

#endif // TIME_REQUEST_DATA_HPP