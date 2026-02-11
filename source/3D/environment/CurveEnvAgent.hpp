#ifndef CURVE_ENVIRONMENT_AGENT
#define CURVE_ENVIRONMENT_AGENT

#include <vector>
#include "EnvironmentAgent.h"

template<typename curve_index_t = size_t>
class CurveEnvironmentAgent : public EnvironmentAgent
{
public:
    inline CurveEnvironmentAgent(const Vector3D &ll, const Vector3D &ur, const std::vector<curve_index_t> &ranges, const MPI_Comm &comm = MPI_COMM_WORLD):
        EnvironmentAgent(ll, ur, comm), range(ranges)
    {}

    virtual ~CurveEnvironmentAgent() = default;

    virtual inline int getCellOwner(curve_index_t d) const
    {
        size_t index = static_cast<size_t>(std::distance(this->range.begin(), std::upper_bound(this->range.begin(), this->range.end(), d)));
        return std::min<size_t>(index, this->size - 1);
    };

    virtual void updatePoints(const std::vector<Vector3D> &newPoints)
    {}

    virtual inline void updateBorders(const std::vector<curve_index_t> &newRange)
    {
        this->range = newRange;
    }

protected:
    std::vector<curve_index_t> range;
};

#endif // CURVE_ENVIRONMENT_AGENT