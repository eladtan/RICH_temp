#include "manager/MonteCarloTracker.hpp"
#include <iostream>
struct Packet {std::size_t id, steps; int rank;};
int main() {
 STORM::MonteCarloTracker<Packet> tracker;
 Packet a{0,1,0}, b{0,2,1};
 tracker.ReportParticle(a); tracker.ReportParticle(b);
 auto route=tracker.GetLocalTrackParticleRoute(0);
 std::cout << "route size=" << route.size() << " origins=" << route[0].rank << "," << route[1].rank << '\n';
 return route.size()==2 && route[0].rank!=route[1].rank ? 0 : 1;
}
