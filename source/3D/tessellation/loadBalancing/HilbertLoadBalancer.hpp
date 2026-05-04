#ifndef HILBERT_LOAD_BALANCER_HPP
#define HILBERT_LOAD_BALANCER_HPP

#ifdef RICH_MPI

#include "CurveLoadBalancer.hpp"
#include "3D/hilbert/HilbertConvertor3D.hpp"
#include "3D/environment/kernels/Identity.hpp"
#include "utils/balance/balance.hpp"
#include "utils/balance/weightedBalance.hpp"
#include "utils/balance/weightedBalance2.hpp"
#include "utils/balance/weightedBalance3.hpp"
#include "ds/OctTree/OctTree.hpp"
#include "3D/hilbert/rectangular/HilbertRectangularConvertor3D.hpp"

#define SPACE_FACTOR 1e-5

class HilbertLoadBalancer : public CurveLoadBalancer
{
public:
    static constexpr const char *type_name = "hilbert";

    HilbertLoadBalancer(const Vector3D &ll, const Vector3D &ur, const std::vector<Vector3D> &points,
                        const std::shared_ptr<const Kernelization3D::IndexingKernel3D> indexing = std::make_shared<const Kernelization3D::Identity>(),
                        const std::vector<curve_index_t> &boundaries = std::vector<curve_index_t>());

    HilbertLoadBalancer(std::shared_ptr<HilbertConvertor3D> convertor,
                        std::shared_ptr<const Kernelization3D::IndexingKernel3D> indexing,
                        const std::vector<curve_index_t> &boundaries);

    ~HilbertLoadBalancer() override = default;

    std::string getTypeName() const override { return type_name; }

    void rebalance(const std::vector<Vector3D> &points, const std::vector<double> &weights) override;

    std::shared_ptr<HilbertLoadBalancer> clone(void) const;

    curve_index_t getCurveIndex(const Vector3D &point) const override;

    void setIndexing(const std::shared_ptr<const Kernelization3D::IndexingKernel3D> indexing);

    inline std::shared_ptr<const Kernelization3D::IndexingKernel3D> getIndexing() const { return this->indexing; }

    void rescale(const Vector3D &ll, const Vector3D &ur, const std::vector<Vector3D> &points);

    void changeBox(const std::pair<Vector3D, Vector3D> &newBox) override;

    void printInfo(void) override;

    inline const std::shared_ptr<HilbertConvertor3D> &getConvertor(void) const {return this->convertor;}

    inline const std::vector<curve_index_t> &getBoundaries(void) const {return this->boundaries;}

    inline size_t getOrder(void) const {return this->convertor->getOrder();}

private:
    std::shared_ptr<HilbertConvertor3D> convertor;
    std::shared_ptr<const Kernelization3D::IndexingKernel3D> indexing;

    void initializeConvertor(const Vector3D &ll, const Vector3D &ur, const std::vector<Vector3D> &points);
};

#endif // RICH_MPI

#endif // HILBERT_LOAD_BALANCER_HPP
