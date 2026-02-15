#ifdef RICH_MPI

#include "HilbertLoadBalancer.hpp"

HilbertLoadBalancer::HilbertLoadBalancer(const std::shared_ptr<HilbertConvertor3D> convertor, const std::shared_ptr<const Kernelization3D::IndexingKernel3D> indexing, const std::vector<curve_index_t> &boundaries)
    : CurveLoadBalancer(boundaries), convertor(convertor), indexing(indexing)
{}

void HilbertLoadBalancer::rebalance(const std::vector<Vector3D> &points, const std::vector<double> &weights)
{
    if(this->convertor == nullptr)
    {
        throw UniversalError("HilbertLoadBalancer::rebalance: convertor was not initialized yet");
    }

    std::vector<curve_index_t> indices;
    for(const Vector3D &point : points)
    {
        indices.push_back(this->convertor->xyz2d((*this->indexing)(point)));
    }

    int dont_do_weights = (weights.empty() and std::all_of(weights.cbegin(), weights.cend(), [&weights](const double &x){return x == weights[0];}))? 1 : 0;
    MPI_Allreduce(MPI_IN_PLACE, &dont_do_weights, 1, MPI_INT, MPI_MAX, this->comm);
    if(this->rank == 0)
    {
        std::cout << "Running rebalancing" << std::endl;
    }
    if(dont_do_weights)
    {
        // responsibilityRange = getBorders(indices);
        this->boundaries = getWeightedBorders2(indices, std::vector<double>(points.size(), 1.0));
    }
    else
    {
        // responsibilityRange = getWeightedBorders(indices, weights);
        this->boundaries = getWeightedBorders2(indices, weights);
    }
}

std::shared_ptr<LoadBalancer> HilbertLoadBalancer::clone(const std::shared_ptr<HilbertConvertor3D> newConvertor, const std::shared_ptr<const Kernelization3D::IndexingKernel3D> newIndexing) const
{
    std::shared_ptr<HilbertLoadBalancer> clone = std::make_shared<HilbertLoadBalancer>(newConvertor, newIndexing, this->boundaries);
    return clone;
}

#endif // RICH_MPI