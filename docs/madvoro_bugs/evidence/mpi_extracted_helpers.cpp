// Extracted verbatim production algorithm bodies; fixture types only.
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
template <typename PointT>
inline bool SphereIntersectsAABB(const PointT &center, typename PointT::coord_type radius, const PointT &tileLL, const PointT &tileUR)
{
    using coord_type = typename PointT::coord_type;
    coord_type d2 = coord_type(0);
    if(center.x < tileLL.x)
    {
        d2 += (tileLL.x - center.x) * (tileLL.x - center.x);
    }
    else if(center.x > tileUR.x)
    {
        d2 += (center.x - tileUR.x) * (center.x - tileUR.x);
    }
    if(center.y < tileLL.y)
    {
        d2 += (tileLL.y - center.y) * (tileLL.y - center.y);
    }
    else if(center.y > tileUR.y)
    {
        d2 += (center.y - tileUR.y) * (center.y - tileUR.y);
    }
    if(center.z < tileLL.z)
    {
        d2 += (tileLL.z - center.z) * (tileLL.z - center.z);
    }
    else if(center.z > tileUR.z)
    {
        d2 += (center.z - tileUR.z) * (center.z - tileUR.z);
    }
    constexpr coord_type eps = std::numeric_limits<coord_type>::epsilon();
    return d2 <= radius * radius * (coord_type(1) + coord_type(64) * eps);
}
template <typename QueryDataType, typename PointT>
inline void ExpandPeriodicQueries(const std::vector<QueryDataType> &baseQueries,
                                  const std::array<bool, 3> &periodic,
                                  const PointT &ll,
                                  const PointT &ur,
                                  std::vector<QueryDataType> &expanded)
{
    PointT L = ur - ll;
    expanded.clear();
    expanded.reserve(baseQueries.size() * 8);
    for(const QueryDataType &baseQuery : baseQueries)
    {
        expanded.push_back(baseQuery);
        std::array<int, 3> shiftRanges[3] = {{0}, {0}, {0}};
        if(periodic[0])
        {
            shiftRanges[0] = {-1, 0, 1};
        }
        if(periodic[1])
        {
            shiftRanges[1] = {-1, 0, 1};
        }
        if(periodic[2])
        {
            shiftRanges[2] = {-1, 0, 1};
        }
        for(int sx : shiftRanges[0])
        {
            for(int sy : shiftRanges[1])
            {
                for(int sz : shiftRanges[2])
                {
                    if(sx == 0 && sy == 0 && sz == 0)
                    {
                        continue;
                    }
                    PointT translation(L.x * sx, L.y * sy, L.z * sz);
                    PointT tileLL = ll + translation;
                    PointT tileUR = ur + translation;
                    if(!SphereIntersectsAABB(baseQuery.center, baseQuery.radius, tileLL, tileUR))
                    {
                        continue;
                    }
                    QueryDataType periodicQuery = baseQuery;
                    periodicQuery.imageTranslation = translation;
                    periodicQuery.center = baseQuery.center - translation;
                    if constexpr(std::is_same_v<QueryDataType, BigRangeQueryData<PointT>>)
                    {
                        periodicQuery.originalPoint = baseQuery.originalPoint - translation;
                        periodicQuery.askOnlyClose = false;
                    }
                    expanded.push_back(periodicQuery);
                }
            }
        }
    }
}
void Fixture::EnsureSymmetry(const std::vector<int> &sentProc, const std::vector<std::vector<int>> &recvProcLists)
{
    for(size_t i = 0; i < this->duplicatedprocs_.size(); i++)
    {
        int _rank =  this->duplicatedprocs_[i];
        bool notAppearingInSent = (std::find(sentProc.begin(), sentProc.end(), _rank) == sentProc.end());
        bool notAppearingInAllRecv = std::all_of(recvProcLists.cbegin(), recvProcLists.cend(), [_rank](const std::vector<int> &recvProcList){return std::find(recvProcList.cbegin(), recvProcList.cend(), _rank) == recvProcList.cend();});
        
        if(notAppearingInSent or notAppearingInAllRecv)
        {
            // not in the intersection, remove the rank
            this->duplicatedprocs_.erase(this->duplicatedprocs_.begin() + i);
            this->duplicated_points_.erase(this->duplicated_points_.begin() + i);
            this->Nghost_.erase(this->Nghost_.begin() + i);
            i--;
        }
    }
    ContainerOps::conditional_shrink(this->duplicatedprocs_);
    ContainerOps::conditional_shrink(this->duplicated_points_);
    ContainerOps::conditional_shrink(this->Nghost_);
}

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
              <<" negative_x_image_multiplicity="<<multiplicities[{-1,0,0}]<<"\n";
  }
  Fixture receiver{{1},{{}},{{5}}};
  receiver.EnsureSymmetry({},{{1}});
  std::cout<<"ONE_WAY_GHOST peer_count="<<receiver.duplicatedprocs_.size()
           <<" ghost_lists="<<receiver.Nghost_.size()<<" expected_peer_count=1\n";
}
