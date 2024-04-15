#ifndef RANGE_QUERY_DATA
#define RANGE_QUERY_DATA

#include "utils/point/3DPoint.hpp"

struct RangeQueryData 
                    #ifdef RICH_MPI
                        : public Serializable
                    #endif // RICH_MPI
{
    size_t pointIdx;
    _3DPoint center;
    typename _3DPoint::coord_type radius;

    #ifdef RICH_MPI

        size_t getChunkSize(void) const override
        {
            return 1 + 3 + 1;
        }

        std::vector<double> serialize(void) const override
        {
            std::vector<double> data;
            data.push_back(static_cast<double>(this->pointIdx));
            data.push_back(this->center.x);
            data.push_back(this->center.y);
            data.push_back(this->center.z);
            data.push_back(static_cast<double>(this->radius));
            return data;
        }

        void unserialize(const std::vector<double> &data) override
        {
            size_t index = 0;
            this->pointIdx = static_cast<size_t>(data[index]);
            index++;
            this->center.x = data[index];
            index++;
            this->center.y = data[index];
            index++;
            this->center.z = data[index];
            index++;
            this->radius = static_cast<typename _3DPoint::coord_type>(data[index]);
        }

    #endif // RICH_MPI
};

#endif // RANGE_QUERY_DATA