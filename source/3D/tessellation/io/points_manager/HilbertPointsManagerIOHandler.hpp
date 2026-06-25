#ifndef HILBERT_POINTS_MANAGER_IO_HANDLER_HPP
#define HILBERT_POINTS_MANAGER_IO_HANDLER_HPP

#ifdef RICH_MPI

#include "PointsManagerIOHandler.hpp"

class HilbertPointsManagerIOHandler : public PointsManagerIOHandler
{
public:
    void dump(HDF5Writer &writer, const std::string &group, const PointsManager<Vector3D, MadVoro::VoronoiPayload> &pm) const override;

    std::shared_ptr<PointsManager<Vector3D, MadVoro::VoronoiPayload>> load(const HDF5Reader &reader, const std::string &group, const Vector3D &ll, const Vector3D &ur) const override;
};

#endif // RICH_MPI

#endif // HILBERT_POINTS_MANAGER_IO_HANDLER_HPP
