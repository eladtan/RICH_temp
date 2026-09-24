#include <iomanip>
#include <iostream>
#include "examples/Vector3D.hpp"
#include "elementary/Face3D.hpp"
int main(){using F=MadVoro::Face3D<Vector3D>;F f(F::point_vec_v{Vector3D(0,0,0),Vector3D(2,0,0),Vector3D(0,2,0)});auto c=MadVoro::calc_centroid(f);std::cout<<std::setprecision(17)<<"area="<<f.GetArea()<<" computed="<<c.x<<","<<c.y<<","<<c.z<<" expected="<<2.0/3<<","<<2.0/3<<",0\n";}
