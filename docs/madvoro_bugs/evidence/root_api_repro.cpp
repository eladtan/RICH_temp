#include <algorithm>
#include <iostream>
#include <numeric>
#include <string>
#include "regression_tests/lib/voronoi_test_common.hpp"
int main(int argc,char**argv) {
#ifdef MADVORO_WITH_MPI
    MPI_Init(&argc,&argv);
    int rank=0; MPI_Comm_rank(MPI_COMM_WORLD,&rank);
#endif
    std::string mode=argc>1?argv[1]:"subset";
    const Vector3D ll(0,0,0),ur(1,1,1);
    auto pts=MadVoro::regression_tests::RandRectangular(32,ll,ur,1729ULL);
    MadVoro::regression_tests::VoronoiGrid grid(ll,ur);
    std::vector<size_t> mask(32);std::iota(mask.begin(),mask.end(),0);
    if(mode=="subset")mask.resize(8);
    if(mode=="permutation")std::rotate(mask.begin(),mask.begin()+1,mask.end());
    if(mode=="empty")mask.clear();
    if(mode=="invalid")mask[0]=pts.size();
    std::cerr<<"API_REPRO mode="<<mode<<" points="<<pts.size()<<" active="<<mask.size()<<"\n";
    int result=0;
    try {
#ifdef MADVORO_WITH_MPI
        if(mode=="mpi_serial") grid.Build(pts);
        else if(mode=="first_suppressed") grid.BuildPartiallyParallel(pts,std::vector<double>(pts.size(),1),mask,true,true);
        else {
            if(rank!=0)pts.clear();
            if(mode=="periodic_resolution") grid.SetPeriodicBoundaries(true,true,true);
            pts=grid.BuildParallel(pts);
            if(mode=="empty_continuity") {
                pts.clear(); grid.BuildParallel(pts,true,true);
                std::cerr<<"CHECK_EMPTY_CONTINUITY owned="<<grid.GetPointNo()<<"\n";
                std::cerr<<"CONTINUITY="<<grid.CheckContinuityOfZone()<<"\n";
            } else if(mode=="periodic_resolution") {
                for(size_t i=grid.GetPointNo()+4;i<grid.GetTotalPointNumber();++i) {
                    if(!grid.IsPeriodicImage(i))continue;
                    Vector3D physical=grid.GetMeshPoint(i);grid.WrapPeriodicPoint(physical);
                    size_t j=grid.ResolvePeriodicImageIndex(i);
                    if(j<grid.GetPointNo()&&MadVoro::abs(grid.GetMeshPoint(j)-physical)>1e-10) {
                        std::cerr<<"WRONG_PREIMAGE rank="<<rank<<" image="<<i<<" physical="<<physical<<" resolved="<<j<<" resolved_point="<<grid.GetMeshPoint(j)<<"\n";
                        result=5;break;
                    }
                }
            } else grid.BuildPartiallyParallel(pts,std::vector<double>(pts.size(),1),mask,true,true);
        }
#else
        grid.BuildPartially(pts,mask);
#endif
        std::cerr<<"BUILD_RETURNED owned="<<grid.GetPointNo()<<" all="<<grid.GetAllPointsNo()<<"\n";
#ifndef MADVORO_WITH_MPI
        if(mode=="permutation") {
            for(size_t j=0;j<mask.size();++j) {
                auto it=grid.GetIndicesInAllPoints().find(j);
                if(it==grid.GetIndicesInAllPoints().end()||it->second!=mask[j]) {
                    std::cerr<<"MAP_MISMATCH active="<<j<<" expected_all="<<mask[j]<<" got="<<(it==grid.GetIndicesInAllPoints().end()?9999:it->second)<<"\n";
                    result=4;break;
                }
            }
        }
#endif
    } catch(const MadVoro::Exception::MadVoroException& e) { std::cerr<<"MADVORO_EXCEPTION "<<e.getErrorMessage()<<"\n";result=3; }
    catch(const std::exception& e) { std::cerr<<"CAUGHT_EXCEPTION "<<e.what()<<"\n";result=3; }
#ifdef MADVORO_WITH_MPI
    MPI_Finalize();
#endif
    return result;
}
