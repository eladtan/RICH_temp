#ifndef HILBERT_CURVE_ENVAGENT_HPP
#define HILBERT_CURVE_ENVAGENT_HPP

#include "../CurveEnvAgent.hpp"
#include "3D/hilbert/HilbertConvertor3D.hpp"

class HilbertCurveEnvironmentAgent : public CurveEnvironmentAgent<hilbert_index_t>
{
public:
    using DistancesVector = std::vector<std::pair<typename Vector3D::coord_type, typename Vector3D::coord_type>>;

    inline HilbertCurveEnvironmentAgent(const Vector3D &ll, const Vector3D &ur, const std::vector<hilbert_index_t> &ranges, HilbertConvertor3D *convertor, const MPI_Comm &comm = MPI_COMM_WORLD):
        CurveEnvironmentAgent(ll, ur, ranges, comm), convertor(convertor)
    {
        this->order = this->convertor->getOrder();
    };

    virtual ~HilbertCurveEnvironmentAgent() = default;

    virtual inline int getOwner(const Vector3D &point) const override
    {
        // TODO: that's wrong
        return this->getCellOwner(this->convertor->xyz2d(point));
    };

    virtual void updatePoints(const std::vector<Vector3D> &newPoints) override
    {}

    virtual inline void updateBorders(const std::vector<hilbert_index_t> &newRange, int newOrder)
    {
        this->CurveEnvironmentAgent::updateBorders(newRange);
        if(this->convertor != nullptr)
        {
            this->convertor->changeOrder(newOrder);
        }
    }

protected:
    HilbertConvertor3D *convertor = nullptr;
    int order;
};

#endif // HILBERT_CURVE_ENVAGENT_HPP
