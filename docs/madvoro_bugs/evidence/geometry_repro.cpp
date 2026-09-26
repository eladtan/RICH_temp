#include <algorithm>
#include <iomanip>
#include <mpi.h>
#include "regression_tests/lib/voronoi_test_common.hpp"
using Grid=MadVoro::regression_tests::VoronoiGrid;
std::vector<MadVoro::Face3D<Vector3D>> tetra_faces() {
 using F=MadVoro::Face3D<Vector3D>;
 Vector3D o(0,0,0),x(1,0,0),y(0,1,0),z(0,0,1);
 return {F(F::point_vec_v{o,y,z}),F(F::point_vec_v{o,z,x}),F(F::point_vec_v{o,x,y}),F(F::point_vec_v{x,z,y})};
}
int main(int argc, char **argv) {
 MPI_Init(&argc, &argv);
 const std::string mode=argc>1?argv[1]:"copy";
 try {
  if(mode=="unbuilt_copy") {
   Grid src(Vector3D(0,0,0),Vector3D(1,1,1));
   std::cout<<"copying unbuilt grid"<<std::endl; Grid copy(src); std::cout<<"copied unbuilt grid"<<std::endl;
  } else if(mode=="custom_release") {
   Grid src(tetra_faces()); Vector3D q(.8,.8,.8);
   std::cout<<"point outside tetra before release="<<src.IsPointOutsideBox(q)<<std::endl;
   src.ReleaseMemory();
   std::cout<<"point outside tetra after release="<<src.IsPointOutsideBox(q)<<std::endl;
  } else if(mode=="custom_parallel") {
   auto faces=MadVoro::BuildBox(Vector3D(0,0,0),Vector3D(1,1,1));
   Grid src(faces);
   auto pts=MadVoro::regression_tests::RandRectangular(32,Vector3D(0,0,0),Vector3D(1,1,1),424242ULL);
   pts=src.BuildParallel(pts);
   std::cout<<"custom box volume="<<MadVoro::regression_tests::SumOwnedCellVolumes(src)<<std::endl;
  } else {
   auto pts=MadVoro::regression_tests::RandRectangular(32,Vector3D(0,0,0),Vector3D(1,1,1),424242ULL);
   Grid src(Vector3D(0,0,0),Vector3D(1,1,1)); pts=src.BuildParallel(pts);
   if(mode=="copy") {
    std::cout<<"source containing cell="<<src.GetContainingCell(pts[0])<<std::endl; Grid copy(src);
    std::cout<<"copy points="<<copy.GetPointNo()<<" volume="<<MadVoro::regression_tests::SumOwnedCellVolumes(copy)<<std::endl;
    std::cout<<"copy containing cell="<<copy.GetContainingCell(pts[0])<<std::endl;
   } else if(mode=="release") {
    src.ReleaseMemory();
    std::cout<<"after release reported point count="<<src.GetPointNo()<<" face count="<<src.GetTotalFacesNumber()<<std::endl;
    pts=src.BuildParallel(pts);
    std::cout<<"rebuilt volume="<<MadVoro::regression_tests::SumOwnedCellVolumes(src)<<std::endl;
   }
  }
 } catch(const MadVoro::Exception::MadVoroException &e) { MadVoro::Exception::reportError(e,std::cout); MPI_Finalize(); return 2; }
 MPI_Finalize();
}
