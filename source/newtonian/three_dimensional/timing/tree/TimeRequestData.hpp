#ifndef TIME_REQUEST_DATA_HPP
#define TIME_REQUEST_DATA_HPP

#include "misc/serializable.hpp"
#include "TimingTree.hpp"

#ifdef RICH_MPI

template<typename T>
class TimeRequestData : public Serializable
{
public:
    using NodeData = TimingTree<Vector3D>::NodeData;

    explicit inline TimeRequestData(const BoundingBox<T> &boundingBox_ = BoundingBox<T>(), size_t cellID_ = std::numeric_limits<size_t>::max(), const NodeData &value_ = NodeData()):
        boundingBox(boundingBox_), cellID(cellID_), value(value_)
    {}

    size_t getChunkSize() const override;
    std::vector<double> serialize() const override;
    void unserialize(const std::vector<double> &data) override;

    BoundingBox<T> boundingBox;
    size_t cellID;
    NodeData value;
};

template<typename T>
size_t TimeRequestData<T>::getChunkSize() const
{
    return this->boundingBox.getChunkSize() + 1 + this->value.getChunkSize();
}

template<typename T>
std::vector<double> TimeRequestData<T>::serialize() const
{
    std::vector<double> serialized;
    std::vector<double> serData = this->boundingBox.serialize();
    serialized.insert(serialized.end(), serData.cbegin(), serData.cend());
    serialized.push_back(static_cast<double>(this->cellID));
    serData = this->value.serialize();
    serialized.insert(serialized.end(), serData.cbegin(), serData.cend());
    return serialized;
}

template<typename T>
void TimeRequestData<T>::unserialize(const std::vector<double> &data)
{
    size_t BBChunkSize = this->boundingBox.getChunkSize();
    std::vector<double> serData(data.cbegin(), data.cbegin() + BBChunkSize);
    this->boundingBox.unserialize(serData);
    this->cellID = static_cast<size_t>(data[BBChunkSize]);
    serData = std::vector<double>(data.cbegin() + BBChunkSize + 1, data.cend());
    this->value.unserialize(serData);
}

#endif // RICH_MPI

#endif // TIME_REQUEST_DATA_HPP