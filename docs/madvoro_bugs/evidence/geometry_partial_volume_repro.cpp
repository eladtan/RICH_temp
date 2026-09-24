#include <algorithm>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <mpi.h>
#include "regression_tests/lib/voronoi_test_common.hpp"
int main(int argc,char**argv){
 MPI_Init(&argc,&argv);int p=0;MPI_Comm_size(MPI_COMM_WORLD,&p);if(p!=1){std::cerr<<"Requires exactly one MPI rank\n";MPI_Finalize();return 2;}
 using Grid=MadVoro::regression_tests::VoronoiGrid;Vector3D lo(0,0,0),hi(1,1,1);
 auto pts=MadVoro::regression_tests::RandRectangular(32,lo,hi,1729ULL);Grid grid(lo,hi);pts=grid.BuildParallel(pts);
 std::vector<Vector3D> fullPoints;std::vector<double> fullVolumes;
 for(size_t i=0;i<grid.GetPointNo();++i){fullPoints.push_back(grid.GetMeshPoint(i));fullVolumes.push_back(grid.GetVolume(i));}
 std::vector<size_t> active(8);std::iota(active.begin(),active.end(),0);const auto retainedInput=pts;
 double referenceActiveVolume=0;for(size_t i:active)referenceActiveVolume+=fullVolumes[i];
 auto returned=grid.BuildPartiallyParallel(retainedInput,std::vector<double>(retainedInput.size(),1),active,true,true);
 size_t mismatches=0;double partialActiveVolume=0;
 std::cout<<std::setprecision(17)<<"PARTIAL_VOLUME input_all="<<retainedInput.size()<<" active_requested="<<active.size()<<" output_all="<<grid.GetAllPointsNo()<<" returned="<<returned.size()<<" active_built="<<grid.GetPointNo()<<"\n";
 for(size_t j=0;j<grid.GetPointNo();++j){size_t id=fullPoints.size();for(size_t i=0;i<fullPoints.size();++i)if(fullPoints[i]==grid.GetMeshPoint(j)){id=i;break;}
 if(id==fullPoints.size()){std::cerr<<"POINT_ID_NOT_FOUND\n";MPI_Finalize();return 3;}
 const double before=fullVolumes[id],after=grid.GetVolume(j);partialActiveVolume+=after;
 if(std::abs(before-after)>1e-11)++mismatches;
 std::cout<<"CELL id="<<id<<" full_volume="<<before<<" partial_volume="<<after<<" difference="<<after-before<<"\n";
 }
 std::cout<<"SUMMARY reference_active_volume="<<referenceActiveVolume<<" partial_active_volume="<<partialActiveVolume<<" mismatched_cells="<<mismatches<<"\n";
 MPI_Finalize();return (returned.size()!=retainedInput.size()||mismatches)?5:0;
}
