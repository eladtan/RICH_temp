#include "VoroCrustVertex.hpp"
#include <sstream>

VoroCrustVertex::VoroCrustVertex(Vector3D const& vertex_, 
                                 std::size_t const index_) : vertex(vertex_),
                                                             edges(),
                                                             faces(),
                                                             divided_faces(), 
                                                             index(index_),
                                                             isSharp(false){}

void VoroCrustVertex::addFace(VoroCrust::Face const& new_face){
    //! TODO: CHECK if Face is really defined by Vertex.
    faces.push_back(new_face);
}

void VoroCrustVertex::addEdge(VoroCrust::Edge const& new_edge) {
    //! TODO: CHECK if Edge is really defined by Vertex.
    edges.push_back(new_edge);
}

std::string VoroCrustVertex::repr() const {
    std::ostringstream s;
    s << "x = " << vertex.x << ", y = " << vertex.y << ", z = " << vertex.z;
    return s.str();
}
