#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <algorithm>
#include <stdexcept>
#include "utils/CounterRNG.hpp"
int main() {
 uint64_t maxUlp=0; double maxRelative=0;
 for(uint64_t i=0;i<1000000+8;++i) {
  uint64_t m=STORM::CounterRNG::next(828391,i)>>12;
  const uint64_t edges[]={0,1,(1ULL<<51)-1,1ULL<<51,(1ULL<<51)+1,(1ULL<<52)-3,(1ULL<<52)-2,(1ULL<<52)-1};
  if(i<8)m=edges[i];
  volatile double u=(double(m)+.5)*0x1p-52;
  double difference=u-1.;if(1.+difference!=u)throw std::runtime_error("RNG lattice subtraction lost precision");
  double a=-std::log1p(difference),b=-std::log(u);uint64_t x,y;std::memcpy(&x,&a,8);std::memcpy(&y,&b,8);
  maxUlp=std::max(maxUlp,x>y?x-y:y-x);maxRelative=std::max(maxRelative,std::abs(a-b)/a);
 }
 if(maxUlp>2)throw std::runtime_error("log entry points differ by more than 2 ULP");
 std::cout<<"PASS: 1000008 RNG lattice values including endpoints; max log/log1p difference="<<maxUlp<<" ULP, relative="<<maxRelative<<"\n";
}
