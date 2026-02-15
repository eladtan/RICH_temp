#ifndef HILBERT_LOAD_BALANCER_HPP
#define HILBERT_LOAD_BALANCER_HPP

#ifdef RICH_MPI

#include "CurveLoadBalancer.hpp"
#include "3D/hilbert/HilbertConvertor3D.hpp"
#include "utils/balance/balance.hpp"
#include "utils/balance/weightedBalance.hpp"
#include "utils/balance/weightedBalance2.hpp"

class HilbertLoadBalancer : public CurveLoadBalancer
{
public:
    HilbertLoadBalancer(const std::shared_ptr<HilbertConvertor3D> convertor, const std::shared_ptr<const Kernelization3D::IndexingKernel3D> indexing, const std::vector<curve_index_t> &boundaries = std::vector<curve_index_t>());

    ~HilbertLoadBalancer() override = default;

    void rebalance(const std::vector<Vector3D> &points, const std::vector<double> &weights) override;

    std::shared_ptr<LoadBalancer> clone(const std::shared_ptr<HilbertConvertor3D> newConvertor, const std::shared_ptr<const Kernelization3D::IndexingKernel3D> newIndexing) const;

    std::shared_ptr<HilbertConvertor3D> convertor;
    std::shared_ptr<const Kernelization3D::IndexingKernel3D> indexing;
};

#endif // RICH_MPI

#endif // HILBERT_LOAD_BALANCER_HPP