#ifndef PARMETIS_LOAD_BALANCER_HPP
#define PARMETIS_LOAD_BALANCER_HPP

#ifdef RICH_MPI
#ifdef WITH_PARMETIS

#include "LoadBalancer.hpp"

class ParMETISLoadBalancer : public LoadBalancer
{
};

#endif // WITH_PARMETIS
#endif // RICH_MPI

#endif // PARMETIS_LOAD_BALANCER_HPP