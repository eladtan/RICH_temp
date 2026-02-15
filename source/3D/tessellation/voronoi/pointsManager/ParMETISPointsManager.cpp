// #include "ParMETISPointManager.hpp"

// #ifdef RICH_MPI
// #ifdef WITH_PARMETIS

// std::pair<double, std::vector<double>> NormalizeWeights(const std::vector<double> &weights)
// {
//     double normalizeFactor = 1.0;
//     std::vector<double> normalizedWeights = weights;
//     double epsilon = EPSILON * *std::max_element(weights.cbegin(), weights.cend());
//     size_t N = normalizedWeights.size();
//     while(true)
//     {
//         bool normalize = false;
//         for(size_t i = 0; i < N; i++)
//         {
//             if(std::fabs(normalizedWeights[i] - normalizedWeights[i]) > epsilon)
//             {
//                 normalize = true;
//                 break;
//             }
//         }
//         if(not normalize)
//         {
//             break;
//         }
//         normalizeFactor *= 10.0;
//         for(double &w : normalizedWeights)
//         {
//             w *= 10.0;
//         }
//     }
//     std::cout << "Got " << weights << ", returning factor " << normalizeFactor << ", new weights " << normalizedWeights << std::endl;
//     double maxFactor;
//     MPI_Allreduce(&normalizeFactor, &maxFactor, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
//     double diffFactor = maxFactor / normalizeFactor;
//     for(double &w : normalizedWeights)
//     {
//         w *= diffFactor;
//     }
//     return {maxFactor, normalizedWeights};
// }

// ParMETISPointManager::ParMETISPointManager(const Tessellation3D &tess, const MPI_Comm &comm)
//     : PointsManager(tess.GetBoxCoordinates().first, tess.GetBoxCoordinates().second, comm), tess(tess)
// {
//     this->envAgent = nullptr;
// }

// ParMETISPointManager::~ParMETISPointManager()
// {
//     delete this->envAgent;
// }

// PointsExchangeResult ParMETISPointManager::exchange(const std::vector<Vector3D> &allPoints, const std::vector<double> &allWeights, const std::vector<size_t> &indicesToWorkWith, const std::vector<double> &radiuses, const std::vector<Vector3D> &previous_CM, bool noExchange)
// {
//     if(this->partition.empty())
//     {
//         PointsExchangeResult result;
//         result.newPoints = allPoints;
//         result.newWeights = allWeights;
//         result.newRadiuses = radiuses;
//         result.indicesToSelf = indicesToWorkWith;
//         result.newCMs = previous_CM;
//         result.participatingIndices = std::vector<bool>(allPoints.size(), true);
//         if(this->envAgent == nullptr)
//         {
//             this->envAgent = new PlainDistributedOctEnvironmentAgent(this->ll, this->ur, allPoints, this->comm);
//         }
//         this->envAgent->updatePoints(result.newPoints);
//         return result;
//     }
//     auto determineFunc = [this](const _3DPointData &point)
//     {
//         size_t localIndex = point.indexInAllPoints;
//         return this->partition.at(localIndex);
//     };
//     PointsExchangeResult result = this->pointsExchange(determineFunc, allPoints, allWeights, indicesToWorkWith, radiuses, previous_CM);
//     this->envAgent->updatePoints(result.newPoints);
//     this->totalWeight = std::accumulate(result.newWeights.cbegin(), result.newWeights.cend(), 0.0);
//     this->partition.clear();
//     return result;
// }

// void ParMETISPointManager::rebalance(const std::vector<Vector3D> &points, const std::vector<double> &weights)
// {    
//     boost::container::flat_map<size_t, std::pair<rank_t, size_t>> ghosts = ExchangeGhosts(this->tess);
//     auto [normalizeFactor, normalizedWeights] = NormalizeWeights(weights);
//     std::vector<idx_t> vtxdist(this->size + 1, 0);
//     size_t myNumPoints = points.size();
//     size_t myOffset;
//     MPI_Exscan(&myNumPoints, &myOffset, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, this->comm);
//     if(this->rank == 0)
//     {
//         myOffset = 0;
//     }
//     MPI_Allreduce(&myNumPoints, &vtxdist.back(), 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, this->comm);
//     MPI_Allgather(&myOffset, 1, IDX_T, vtxdist.data(), 1, IDX_T, this->comm);
//     std::vector<idx_t> xadj(myNumPoints);
//     std::vector<idx_t> adjncy;
//     std::vector<idx_t> vwgt;
//     std::vector<idx_t> adjwgt;
//     size_t N = this->tess.GetPointNo();
//     for(size_t i = 0; i < myNumPoints; i++)
//     {
//         xadj[i] = adjncy.size();
//         vwgt.push_back(std::ceil(normalizedWeights.size() > i? normalizedWeights[i] : normalizeFactor));
//         for(size_t neighbor : this->tess.GetNeighbors(i))
//         {
//             if(neighbor < N)
//             {
//                 adjncy.push_back(neighbor + myOffset);
//                 adjwgt.push_back(std::ceil(normalizeFactor));
//             }
//             else
//             {
//                 auto it = ghosts.find(neighbor);
//                 if(it == ghosts.cend())
//                 {
//                     // mirrored point
//                     continue;
//                 }
//                 else
//                 {

//                 }
//                 auto [otherRank, indexInOtherRank] = it->second;
//                 idx_t otherRankOffset = vtxdist[otherRank];
//                 adjncy.push_back(indexInOtherRank + otherRankOffset);
//                 adjwgt.push_back(std::ceil(normalizeFactor));
//             }
//         }
//     }
//     xadj.push_back(adjncy.size());
//     idx_t wgtflag = 3; // both vertex and edge weights
//     int numflag = 0; // C-style numbering
//     idx_t ncon = 1; // number of vertices' weights
//     idx_t nparts = this->size; // number of parts to divide to
//     std::vector<real_t> tpwgts(nparts * ncon, 1.0 / nparts);
//     std::vector<real_t> ubvec(ncon, 1.01); // imbalance tolerance
//     std::vector<idx_t> options = {0, 0, 0}; // use default options
    
//     // output parameters
//     idx_t edgecut;
//     this->partition = std::vector<idx_t>(myNumPoints);
//     idx_t dummy;
//     real_t dummy_real;
//     ParMETIS_V3_PartKway(vtxdist.data(), xadj.data(), adjncy.empty()? &dummy : adjncy.data(),
//                             vwgt.data(), adjwgt.data(), &wgtflag, &numflag, &ncon, &nparts,
//                             tpwgts.data(), ubvec.empty()? &dummy_real : ubvec.data(), options.data(),
//                             &edgecut, this->partition.empty()? &dummy : this->partition.data(), &this->comm);
// }

// const EnvironmentAgent *ParMETISPointManager::getEnvironmentAgent() const
// {
//     return this->envAgent;
// }

// void ParMETISPointManager::setLoadBalancer(std::shared_ptr<LoadBalancer> loadBalancer)
// {
//     LoadBalancer *plb = dynamic_cast<ParMETISLoadBalancer*>(loadBalancer.get());
//     if(plb == nullptr)
//     {
//         UniversalError eo("ParMETISPointManager::setLoadBalancer: given load balancer is not a ParMETISLoadBalancer");
//         throw eo;
//     }
//     this->loadBalancer = loadBalancer;
// }

// std::shared_ptr<LoadBalancer> ParMETISPointManager::getLoadBalancer(void)
// {
//     return this->loadBalancer;
// }

// #endif // WITH_PARMETIS
// #endif // RICH_MPI
