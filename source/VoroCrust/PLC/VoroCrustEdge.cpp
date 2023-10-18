#include "VoroCrustEdge.hpp"
#include <iostream>
#include <sstream>
#include <cmath>

VoroCrustEdge::VoroCrustEdge(VoroCrust::Vertex const& v1, 
                             VoroCrust::Vertex const& v2, 
                             std::size_t const index_) : vertex1(v1), 
                                                         vertex2(v2), 
                                                         faces(),
                                                         divided_faces(),
                                                         index(index_),
                                                         isSharp(false),
                                                         isCreased(false),
                                                         crease_index(-1) {}

bool VoroCrustEdge::checkIfEqual(VoroCrust::Vertex const& v1, VoroCrust::Vertex const& v2){
    if (v1->index == vertex1->index && v2->index == vertex2->index){
        return true;
    }

    if (v2->index == vertex1->index && v1->index == vertex2->index){
        return true;
    }

    return false;
}

void VoroCrustEdge::addFace(VoroCrust::Face const& new_face){
    faces.push_back(new_face);
}


double VoroCrustEdge::calcDihedralAngle(){
    /*  Calculates the Dihedral angle using the formula in https://en.wikipedia.org/wiki/Dihedral_angle
        for the dihedral angle between two half planes.
    */
    if(faces.size() != 2){
        std::cout << "ERROR: can't calculate dihedral angle if number of faces " << std::endl;
        exit(1);
    }

    Vector3D const& b0 = vertex2->vertex - vertex1->vertex;
    Vector3D b1, b2;
    
    // finds the edge on the first face starting at vertex1 (and is different the `this Edge`)
    for(auto const& edge : faces[0]->edges){
        if(edge->index == index) continue;

        if(edge->vertex1->index == vertex1->index){
            b1 = edge->vertex2->vertex - edge->vertex1->vertex;
            break;        
        }

        if(edge->vertex2->index == vertex1->index){
            b1 = edge->vertex1->vertex - edge->vertex2->vertex;
            break;        
        }
    }

    // finds the edge on the second face starting at vertex1 (and is different the `this Edge`)
    for(auto const& edge : faces[1]->edges){
        if(edge->index == index) continue;

        if(edge->vertex1->index == vertex1->index){
            b2 = edge->vertex2->vertex - edge->vertex1->vertex;
            break;        
        }

        if(edge->vertex2->index == vertex1->index){
            b2 = edge->vertex1->vertex - edge->vertex2->vertex;
            break;        
        }
    }

    // Formula for the dihedral angle between two half planes https://en.wikipedia.org/wiki/Dihedral_angle
    Vector3D const& b0_X_b1 = CrossProduct(b0, b1);
    Vector3D const& b0_X_b2 = CrossProduct(b0, b2);

    return CalcAngle(b0_X_b1, b0_X_b2);
}

void VoroCrustEdge::orientWithRespectTo(VoroCrust::Edge const& edge){
    //! WARNING: this assumes that one of the vertices are shared by the edge
    if(edge->vertex2->index == vertex2->index || edge->vertex1->index == vertex1->index){
        // flip orientation of edge
        std::swap(vertex1, vertex2);
    }
}

std::string VoroCrustEdge::repr() {
    std::ostringstream s;

    s << "\n\t\t\tvertex " << vertex1->index << ": " << vertex1->repr() << ", vertex " << vertex2->index << " : " << vertex2->repr();

    return s.str();
}