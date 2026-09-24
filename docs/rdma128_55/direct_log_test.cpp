#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>
#include "gpu/GreyIMCKernel.hpp"
#include "reference_event.hpp"
struct V { double x=0,y=0,z=0; };
struct P { V location,velocity; STORM::cell_index_t cellIndex=0; double timeLeft,weight,initialWeight,frequency; uint64_t rngKey,rngCounter; };
double maxDifference=0;
bool same(double a,double b,double scale=1.) {double d=std::abs(a-b)/std::max({scale,std::abs(a),std::abs(b),1e-300});maxDifference=std::max(maxDifference,d);return d<=2e-14;}
bool same(V a,V b,double scale=1.) {return same(a.x,b.x,scale) && same(a.y,b.y,scale) && same(a.z,b.z,scale);}
int main() {
 std::mt19937_64 rng(92231); std::uniform_real_distribution<double> u(0,1);
 size_t offsets[]={0,6}; V normals[]={{1,0,0},{-1,0,0},{0,1,0},{0,-1,0},{0,0,1},{0,0,-1}};
 double planes[]={-1,-1,-1,-1,-1,-1}; STORM::cell_index_t next[]={1,2,3,4,5,6}; uint8_t boundary[]={1,1,1,1,1,1}; V center[1];
 size_t events[5]={};
 for(size_t i=0;i<300000;++i) {
  double absorption=std::pow(10.,-12+16*u(rng)), scattering=std::pow(10.,-12+16*u(rng)),fleck=u(rng);
  V cellVelocity=i%3==2 ? V{.1*u(rng),.1*u(rng),.1*u(rng)}:V{};
  double m[2]={},r[2]={};V momentum[2];
  STORM::gpu::GreyIMCViews<V> view;
  view.grid.cellFaceOffsets=offsets;view.grid.cellCount=1;view.grid.normals=normals;view.grid.facePlaneOffsets=planes;
  view.grid.nextCellIndices=next;view.grid.boundaryCrossings=boundary;view.grid.cellCenters=center;
  view.absorptionOpacities=&absorption;view.scatteringOpacities=&scattering;view.fleckFactors=&fleck;view.cellVelocities=&cellVelocity;
  view.speedOfLight=3;view.depositMomentum=1;view.comovingTransport=i%3!=0;view.staticScatterers=i%2;view.depositMaterialEnergy=i%5!=0;
  P original;original.location={1.8*u(rng)-.9,1.8*u(rng)-.9,1.8*u(rng)-.9};original.velocity={u(rng)-.5,u(rng)-.5,u(rng)-.5};
  auto &v=original.velocity;double norm=std::sqrt(v.x*v.x+v.y*v.y+v.z*v.z);v={3*v.x/norm,3*v.y/norm,3*v.z/norm};
  original.timeLeft=std::pow(10.,-12+13*u(rng));original.weight=std::pow(10.,-6+12*u(rng));original.initialWeight=original.weight;
  original.frequency=1;original.rngKey=rng();original.rngCounter=rng();P a=original,b=original;
  view.pendingMaterialEnergy=m;view.pendingRadiationEnergy=r;view.pendingMomentum=momentum;
  auto x=STORM::transport::ReferenceAdvanceIMC(a,view,STORM::transport::GreyOpacityPolicy{});
  view.pendingMaterialEnergy=m+1;view.pendingRadiationEnergy=r+1;view.pendingMomentum=momentum+1;
  auto y=STORM::transport::AdvanceIMC(b,view,STORM::transport::GreyOpacityPolicy{});
  if(x.error!=y.error || x.step.change!=y.step.change || x.step.nextCellIndex!=y.step.nextCellIndex || x.step.boundaryCrossing!=y.step.boundaryCrossing || x.directedFace!=y.directedFace ||
     !same(a.location,b.location)||!same(a.velocity,b.velocity)||!same(a.timeLeft,b.timeLeft,original.timeLeft)||!same(a.weight,b.weight,original.weight)||a.frequency!=b.frequency||a.rngCounter!=b.rngCounter||
     !same(m[0],m[1],original.weight)||!same(r[0],r[1],original.weight*original.timeLeft)||!same(momentum[0],momentum[1],original.weight)) throw std::runtime_error("differential mismatch at "+std::to_string(i));
  ++events[static_cast<int>(y.step.change)];
 }
 std::cout<<"max normalized state/tally difference="<<maxDifference<<"\n";
 std::cout<<"PASS: 300000 complete IMC events match event choices and RNG; floating state/tallies within 2e-14 relative to input scales; moving/zero-velocity frames\n";
}
