// #ifndef PARMETIS_POINT_MANAGER_HPP
// #define PARMETIS_POINT_MANAGER_HPP

// #ifdef RICH_MPI
// #ifdef WITH_PARMETIS

// #include <parmetis.h>
// #include "PointsManager.hpp"
// #include "3D/tessellation/Tessellation3D.hpp"
// #include "3D/tessellation/loadBalancing/ParMETISLoadBalancer.hpp"
// #include "3D/environment/PlainDistributedOctEnvAgent.hpp"
// #include "3D/tessellation/ExchangeGhosts.hpp"

// class ParMETISPointManager : public PointsManager
// {
// public:
//     ParMETISPointManager(const Tessellation3D &tess, const MPI_Comm &comm = MPI_COMM_WORLD);

//     ~ParMETISPointManager();

//     PointsExchangeResult exchange(const std::vector<Vector3D> &allPoints, const std::vector<double> &allWeights, const std::vector<size_t> &indicesToWorkWith, const std::vector<double> &radiuses, const std::vector<Vector3D> &previous_CM, bool noExchange);

//     void rebalance(const std::vector<Vector3D> &points, const std::vector<double> &weights = std::vector<double>());

//     const EnvironmentAgent *getEnvironmentAgent() const;

//     void setLoadBalancer(std::shared_ptr<LoadBalancer> loadBalancer);

//     std::shared_ptr<LoadBalancer> getLoadBalancer(void);

// private:
//     const Tessellation3D &tess;
//     std::shared_ptr<LoadBalancer> loadBalancer;
//     PlainDistributedOctEnvironmentAgent *envAgent;
//     std::vector<idx_t> partition;
// };

// #endif // WITH_PARMETIS
// #endif // RICH_MPI

// #endif // PARMETIS_POINT_MANAGER_HPP