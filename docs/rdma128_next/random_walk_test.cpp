#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>
#include "gpu/GreyIMCKernel.hpp"
#include "reference_rw.hpp"
struct V { double x=0,y=0,z=0; };
struct P { V location,velocity; STORM::cell_index_t cellIndex=0; double timeLeft,weight,initialWeight,frequency; uint64_t rngKey,rngCounter; };
bool same(V a,V b) {return a.x==b.x && a.y==b.y && a.z==b.z;}
int main() {
 STORM::RandomWalk tables;
 std::mt19937_64 rng(994010); std::uniform_real_distribution<double> u(0,1);
 size_t offsets[]={0,6}; V normals[]={{1,0,0},{-1,0,0},{0,1,0},{0,-1,0},{0,0,1},{0,0,-1}};
 double nx[]={1,-1,0,0,0,0},ny[]={0,0,1,-1,0,0},nz[]={0,0,0,0,1,-1};
 double planes[]={-1,-1,-1,-1,-1,-1}; V center[1]; uint8_t eligible=1;
 double bounds[]={0,1,2},cdf[]={0,.4,1}; size_t taken=0,rejected=0;
 for(size_t i=0;i<300000;++i) {
  double total=std::pow(10.,-1+6*u(rng)),absorption=i%4==0?0:total*std::pow(10.,-8+8*u(rng)),fleck=u(rng);
  STORM::PGRWCellData spectral; spectral.sigmaT_bar=total;spectral.sigmaA_bar=absorption;spectral.D=3e10/(3*total);spectral.gamma=.7;spectral.groupCutoff=1;
  double m[2]={},r[2]={},group[2][2]={{},{}};size_t steps[2]={};
  STORM::gpu::GreyIMCViews<V> view;
  view.grid.cellFaceOffsets=offsets;view.grid.cellCount=1;view.grid.normals=normals;view.grid.facePlaneOffsets=planes;view.grid.cellCenters=center;
  if(i%2) {view.grid.normalX=nx;view.grid.normalY=ny;view.grid.normalZ=nz;}
  view.absorptionOpacities=&absorption;view.fleckFactors=&fleck;view.speedOfLight=3e10;view.depositMaterialEnergy=i%5!=0;
  view.energyBoundaries=bounds;view.thermalEmissionCdf=cdf;view.groupCount=2;
  auto &rw=view.randomWalk;rw.enabled=1;rw.cellEligible=&eligible;rw.cellTotalOpacity=&total;rw.minimumParticleOpticalDepth=3;rw.pgrwCells=&spectral;rw.spectralEnabled=i%3==0;
  rw.tables.tau=tables.GetTauTable().data();rw.tables.logTau=tables.GetLogTauTable().data();rw.tables.survival=tables.GetSurvivalTable().data();rw.tables.radius=tables.GetRadiusTable().data();rw.tables.tableSize=tables.GetTauTable().size();rw.tables.radiusTableSize=tables.GetRadiusTableSize();rw.tables.tauMin=tables.GetMinimumTau();rw.tables.tauMax=tables.GetMaximumTau();
  P original;original.location={2*u(rng)-1,2*u(rng)-1,2*u(rng)-1};original.velocity={3e10,0,0};
  original.timeLeft=std::pow(10.,-14+10*u(rng));original.weight=std::pow(10.,-6+12*u(rng));original.initialWeight=original.weight;
  original.frequency=2*u(rng);original.rngKey=rng();original.rngCounter=rng();P a=original,b=original;
  view.pendingMaterialEnergy=m;view.pendingRadiationEnergy=r;view.pendingGroupRadiationEnergy=group[0];rw.stepCounter=steps;
  auto x=STORM::transport::ReferenceRandomWalk(a,view);
  view.pendingMaterialEnergy=m+1;view.pendingRadiationEnergy=r+1;view.pendingGroupRadiationEnergy=group[1];rw.stepCounter=steps+1;
  auto y=STORM::transport::TryAdvanceRandomWalk(b,view);
  if(x.taken!=y.taken || x.invalid!=y.invalid || x.step.change!=y.step.change || x.step.nextCellIndex!=y.step.nextCellIndex || x.step.boundaryCrossing!=y.step.boundaryCrossing ||
     !same(a.location,b.location)||!same(a.velocity,b.velocity)||a.timeLeft!=b.timeLeft||a.weight!=b.weight||a.frequency!=b.frequency||a.rngCounter!=b.rngCounter||
     m[0]!=m[1]||r[0]!=r[1]||group[0][0]!=group[1][0]||group[0][1]!=group[1][1]||steps[0]!=steps[1]) throw std::runtime_error("differential mismatch at "+std::to_string(i));
  if(y.taken)++taken;else ++rejected;
 }
 std::cout<<"PASS: 300000 complete random-walk attempts match exactly; "<<taken<<" taken, "<<rejected<<" rejected; grey/spectral, AoS/SoA, particle/RNG/tallies\n";
}
