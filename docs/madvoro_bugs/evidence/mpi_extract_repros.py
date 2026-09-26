"""Generate bounded fixtures with production algorithm bodies copied verbatim.
Only EnsureSymmetry's class qualification is replaced. The production files
are neither patched nor compiled with replacements.
"""
from pathlib import Path
import hashlib

repo = Path('/home/maorm/RICH')
src = repo / 'source/3D/tessellation/voronoi/Voronoi3D.hpp'
text = src.read_text()
out = repo / 'docs/madvoro_bugs/evidence'

def section(start, end):
    return text[text.index(start):text.index(end, text.index(start))]

sphere = section('template <typename PointT>\ninline bool SphereIntersectsAABB', '\ninline int GetBoxFaceAxis')
expand = section('template <typename QueryDataType, typename PointT>\ninline void ExpandPeriodicQueries', '\ntypedef std::array<std::size_t, 4> b_array_4;')
ensure = section('template <typename PointT>\nvoid Voronoi3D<PointT>::EnsureSymmetry', '\n// void Voronoi3D<PointT>::InitialExchange')
ensure = ensure.replace('template <typename PointT>\n', '', 1).replace('Voronoi3D<PointT>::EnsureSymmetry', 'Fixture::EnsureSymmetry', 1)

prefix = '''// Extracted verbatim production algorithm bodies; fixture types only.
#include <algorithm>
#include <array>
#include <cstddef>
#include <iostream>
#include <limits>
#include <map>
#include <tuple>
#include <type_traits>
#include <vector>
#include "utils/container_utils.hpp"
using MadVoro::ContainerOps::conditional_shrink;
namespace ContainerOps = MadVoro::ContainerOps;
struct Vec {
  using coord_type=double;
  double x=0,y=0,z=0;
  Vec()=default;
  Vec(double x,double y,double z):x(x),y(y),z(z){}
  Vec operator+(const Vec& b) const { return {x+b.x,y+b.y,z+b.z}; }
  Vec operator-(const Vec& b) const { return {x-b.x,y-b.y,z-b.z}; }
};
template<class T> struct BigRangeQueryData;
struct Query { Vec center; double radius; Vec imageTranslation; };
struct Fixture {
  std::vector<int> duplicatedprocs_;
  std::vector<std::vector<size_t>> duplicated_points_,Nghost_;
  void EnsureSymmetry(const std::vector<int>&,const std::vector<std::vector<int>>&);
};
'''
main = '''
int main() {
  const Vec lo(0,0,0), hi(1,1,1);
  for (int axes=1;axes<=3;++axes) {
    std::array<bool,3> periodic{true,axes>=2,axes>=3};
    std::vector<Query> input{{Vec(.05,.5,.5),.1,Vec()}};
    std::vector<Query> result;
    ExpandPeriodicQueries(input,periodic,lo,hi,result);
    std::map<std::tuple<double,double,double>,size_t> multiplicities;
    for(const auto& q:result) ++multiplicities[{q.imageTranslation.x,q.imageTranslation.y,q.imageTranslation.z}];
    std::cout << "PERIODIC_EXPANSION axes="<<axes<<" records="<<result.size()
              <<" unique_images="<<multiplicities.size()
              <<" negative_x_image_multiplicity="<<multiplicities[{-1,0,0}]<<"\\n";
  }
  Fixture receiver{{1},{{}},{{5}}};
  receiver.EnsureSymmetry({},{{1}});
  std::cout<<"ONE_WAY_GHOST peer_count="<<receiver.duplicatedprocs_.size()
           <<" ghost_lists="<<receiver.Nghost_.size()<<" expected_peer_count=1\\n";
}
'''
(out/'mpi_extracted_helpers.cpp').write_text(prefix+sphere+expand+ensure+main)
(out/'mpi_extracted_helpers_provenance.txt').write_text(
    f'Source: {src}\nSHA256: {hashlib.sha256(src.read_bytes()).hexdigest()}\n'
    'SphereIntersectsAABB and ExpandPeriodicQueries bodies copied verbatim.\n'
    'EnsureSymmetry body copied verbatim; only template/class qualification replaced.\n')
