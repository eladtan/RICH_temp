#include <algorithm>
#include <iomanip>
#include <mpi.h>
#include "regression_tests/lib/voronoi_test_common.hpp"
int main(int argc,char**argv) {
  MPI_Init(&argc,&argv);
  int rank=0,p=1; MPI_Comm_rank(MPI_COMM_WORLD,&rank); MPI_Comm_size(MPI_COMM_WORLD,&p);
  const size_t n=argc>1?std::stoull(argv[1]):4096;
  const int builds=argc>2?std::stoi(argv[2]):4;
  const bool periodic=argc>3?std::stoi(argv[3])!=0:false;
  const Vector3D ll(0,0,0),ur(1,1,1);
  auto points=rank==0?MadVoro::regression_tests::RandRectangular(n,ll,ur,424242ULL):std::vector<Vector3D>{};
  points=MadVoro::regression_tests::SpreadPointsFromRoot(points,rank);
  MadVoro::regression_tests::VoronoiGrid grid(ll,ur);
  if(periodic) grid.SetPeriodicBoundaries(true,true,true);
  int failed=0;
  for(int k=0;k<builds;++k) {
    MPI_Barrier(MPI_COMM_WORLD);
    double start=MPI_Wtime();
    points=grid.BuildParallel(points);
    double elapsed=MPI_Wtime()-start,maximum=0;
    MPI_Reduce(&elapsed,&maximum,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
    double volume=MadVoro::regression_tests::ReduceGlobalVolume(MadVoro::regression_tests::SumOwnedCellVolumes(grid));
    unsigned long long local=grid.GetPointNo(),total=0;
    MPI_Allreduce(&local,&total,1,MPI_UNSIGNED_LONG_LONG,MPI_SUM,MPI_COMM_WORLD);
    failed|=!(std::isfinite(volume)&&std::abs(volume-1)<1e-10&&total==n);
    if(rank==0) std::cout<<"AUDIT p="<<p<<" n="<<n<<" periodic="<<periodic<<" build="<<k<<" max_seconds="<<std::setprecision(10)<<maximum<<" volume="<<std::setprecision(17)<<volume<<" total="<<total<<" failed="<<failed<<std::endl;
  }
  MPI_Finalize(); return failed;
}
