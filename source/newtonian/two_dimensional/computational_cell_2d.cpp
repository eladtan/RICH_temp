#include "computational_cell_2d.hpp"
#include "../../misc/utils.hpp"

using namespace std;

vector<string> ComputationalCell::tracerNames;
vector<string> ComputationalCell::stickerNames;

ComputationalCell::ComputationalCell(void) :
  density(0), pressure(0), velocity(Vector2D()), tracers(),
  stickers(){}

ComputationalCell::ComputationalCell(ComputationalCell const& other) :
  density(other.density), pressure(other.pressure), velocity(other.velocity), tracers(other.tracers),
  stickers(other.stickers){}

ComputationalCell& ComputationalCell::operator=(ComputationalCell const& other)
{
  density = other.density;
  pressure = other.pressure;
  velocity = other.velocity;
  tracers = other.tracers;
  stickers = other.stickers;
  return *this;
}

ComputationalCell& ComputationalCell::operator+=(ComputationalCell const& other)
{
  this->density += other.density;
  this->pressure += other.pressure;
  this->velocity += other.velocity;
  assert(this->tracers.size() == other.tracers.size());
  size_t N = this->tracers.size();
  for (size_t j = 0; j < N; ++j)
    this->tracers[j] += other.tracers[j];
  return *this;
}

ComputationalCell& ComputationalCell::operator-=(ComputationalCell const& other)
{
  this->density -= other.density;
  this->pressure -= other.pressure;
  this->velocity -= other.velocity;
  assert(this->tracers.size() == other.tracers.size());
  size_t N = this->tracers.size();
  for (size_t j = 0; j < N; ++j)
    this->tracers[j] -= other.tracers[j];
  return *this;
}

ComputationalCell& ComputationalCell::operator*=(double s)
{
  this->density *= s;
  this->pressure *= s;
  this->velocity *= s;
  size_t N = this->tracers.size();
  for (size_t j = 0; j < N; ++j)
    this->tracers[j] *= s;
  return *this;
}

void ComputationalCellAddMult(ComputationalCell &res, ComputationalCell const& other, double scalar)
{
  res.density += other.density*scalar;
  res.pressure += other.pressure*scalar;
  res.velocity += other.velocity*scalar;
  assert(res.tracers.size() == other.tracers.size());
  size_t N = res.tracers.size();
  for (size_t j = 0; j < N; ++j)
    res.tracers[j] += other.tracers[j]*scalar;
}

ComputationalCell operator+(ComputationalCell const& p1, ComputationalCell const& p2)
{
  ComputationalCell res(p1);
  res += p2;
  return res;
}

ComputationalCell operator-(ComputationalCell const& p1, ComputationalCell const& p2)
{
  ComputationalCell res(p1);
  res -= p2;
  return res;
}

ComputationalCell operator/(ComputationalCell const& p, double s)
{
  ComputationalCell res(p);
  res.density /= s;
  res.pressure /= s;
  size_t N = res.tracers.size();
  for (size_t j = 0; j < N; ++j)
    res.tracers[j] /= s;
  res.velocity = res.velocity / s;
  return res;
}

ComputationalCell operator*(ComputationalCell const& p, double s)
{
  ComputationalCell res(p);
  res.density *= s;
  res.pressure *= s;
  size_t N = res.tracers.size();
  for (size_t j = 0; j < N; ++j)
    res.tracers[j] *= s;
  res.velocity = res.velocity * s;
  return res;
}

ComputationalCell operator*(double s, ComputationalCell const& p)
{
  return p*s;
}

void ReplaceComputationalCell(ComputationalCell & cell, ComputationalCell const& other)
{
  cell.density = other.density;
  cell.pressure = other.pressure;
  cell.velocity = other.velocity;
  assert(cell.tracers.size() == other.tracers.size());
  assert(cell.stickers.size() == other.stickers.size());
  size_t N = cell.tracers.size();
  for (size_t j = 0; j < N; ++j)
    cell.tracers[j] = other.tracers[j];
  N = cell.stickers.size();
  for (size_t i = 0; i < N; ++i)
    cell.stickers[i] = other.stickers[i];
}


Slope::Slope(void) :xderivative(ComputationalCell()), yderivative(ComputationalCell()) {}

Slope::Slope(ComputationalCell const & x, ComputationalCell const & y) : xderivative(x), yderivative(y)
{}

#ifdef RICH_MPI
  size_t ComputationalCell::dump(Serializer *serializer) const
  {
    size_t bytes = 0;
    bytes += serializer->insert(this->density);
    bytes += serializer->insert(this->pressure);
    bytes += serializer->insert(this->velocity);
    bytes += serializer->insert(this->tracers.size());
    for(size_t j = 0; j < tracers.size(); ++j)
    {
      bytes += serializer->insert(this->tracers[j]);
    }
    bytes += serializer->insert(this->stickers.size());
    for(size_t i = 0; i < stickers.size(); ++i)
    {
      bytes += serializer->insert(this->stickers[i]);
    }
    return bytes;
  }

  size_t ComputationalCell::load(const Serializer *serializer, size_t byteOffset)
  {
    size_t bytes = 0;
    bytes += serializer->extract(this->density, byteOffset);
    bytes += serializer->extract(this->pressure, byteOffset + bytes);
    bytes += serializer->extract(this->velocity, byteOffset + bytes);
    size_t tracersSize;
    bytes += serializer->extract(tracersSize, byteOffset + bytes);
    for(size_t j = 0; j < tracersSize; ++j)
    {
      bytes += serializer->extract(this->tracers[j], byteOffset + bytes);
    }
    size_t stickersSize;
    bytes += serializer->extract(stickersSize, byteOffset + bytes);
    for(size_t i = 0; i < stickersSize; ++i)
    {
      bytes += serializer->extract(this->stickers[i], byteOffset + bytes);
    }
    return bytes;
  }

  size_t Slope::dump(Serializer *serializer) const
  {
    size_t bytes = 0;
    bytes += serializer->insert(this->xderivative);
    bytes += serializer->insert(this->yderivative);
    return bytes;
  }

  size_t Slope::load(const Serializer *serializer, size_t byteOffset)
  {
    size_t bytes = 0;
    bytes += serializer->extract(this->xderivative, byteOffset);
    bytes += serializer->extract(this->yderivative, byteOffset + bytes);
    return bytes;
  }
#endif // RICH_MPI
