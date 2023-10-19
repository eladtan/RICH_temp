#ifndef VOROCRUST_USING
#define VOROCRUST_USING

#include <memory>

class VoroCrustVertex;
class VoroCrustEdge;
class VoroCrustFace;

namespace VoroCrust
{
    using Vertex = std::shared_ptr<VoroCrustVertex>;
    using Edge = std::shared_ptr<VoroCrustEdge>;
    using Face = std::shared_ptr<VoroCrustFace>;
}
#endif /* VOROCRUST_USING */