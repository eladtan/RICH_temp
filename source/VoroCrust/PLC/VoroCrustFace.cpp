#include "VoroCrustFace.hpp"
#include "source/3D/elementary/Vector3D.hpp"
#include "source/3D/elementary/Mat33.hpp"
#include <iostream>
#include <sstream>
#include <algorithm>

bool sameSide(Vector3D const& p1, Vector3D const& p2, Vector3D const& a, Vector3D const& b);

VoroCrustFace::VoroCrustFace(std::vector<VoroCrust::Vertex> const &vertices_,
                             std::size_t const index_) : vertices(vertices_),
                                                         edges(),
                                                         index(index_),
                                                         current_normal(),
                                                         centeroid(),
                                                         isPatched(false), 
                                                         patch_index(0) {}

void VoroCrustFace::addEdge(VoroCrust::Edge const& edge)
{
    //! TODO: CHECK if edge is really on the boundary.
    edges.push_back(edge);
}

Vector3D VoroCrustFace::calcNormal()
{
    Vector3D const &v1 = vertices[1]->vertex - vertices[0]->vertex;
    Vector3D const &v2 = vertices[2]->vertex - vertices[1]->vertex;

    Vector3D const &cross_product = CrossProduct(v2, v1);
    current_normal = cross_product / abs(cross_product);

    return current_normal;
}

double VoroCrustFace::calcSignedArea(){
    /*calculates the signed area using formula in https://math.stackexchange.com/questions/3207981/how-do-you-calculate-the-area-of-a-2d-polygon-in-3d*/
    //! WARNING: This is not really a signed area !!!!
    Vector3D const& v = vertices[0]->vertex;

    Vector3D signed_area_vec({0, 0, 0});
    for (std::size_t i=1; i<vertices.size()-1; ++i){
        Vector3D const& v1 = vertices[i]->vertex - v;
        Vector3D const& v2 = vertices[i+1]->vertex - v;

        signed_area_vec += CrossProduct(v1, v2);
    }

    return 0.5*abs(signed_area_vec);
}

double VoroCrustFace::calcArea(){
    return std::abs(calcSignedArea());
}

void VoroCrustFace::flipOrientation(){
    std::reverse(vertices.begin(), vertices.end());
}

void VoroCrustFace::orientWithRespectTo(VoroCrust::Face const& face){
    Vector3D const& n1 = this->calcNormal();
    Vector3D const& n2 = face->calcNormal();
    
    // if normals are opposite flip orientation
    if(ScalarProd(n1, n2) < 0){
        flipOrientation();
    }
}

Vector3D VoroCrustFace::calcCenteroid() {
    // assumes face is a triangle!
    centeroid  = vertices[0]->vertex;
    centeroid += vertices[1]->vertex;
    centeroid += vertices[2]->vertex;

    centeroid = centeroid / 3.0;

    return centeroid;
}

std::pair<bool, Vector3D> VoroCrustFace::pointZparallelRayIntersectsAt(Vector3D const& point) const {
    //https://math.stackexchange.com/questions/2686606/equation-of-a-plane-passing-through-3-points
    Vector3D const& p1 = vertices[0]->vertex;
    Vector3D const& p2 = vertices[1]->vertex;
    Vector3D const& p3 = vertices[2]->vertex;

    double const coeff_z = Mat33<double>(p1.x, p1.y, 1.0, p2.x, p2.y, 1.0, p3.x, p3.y, 1.0).determinant();
    
    // face is parallel to ray return success := false
    if(std::abs(coeff_z) < 1e-14) {
        return std::pair(false, Vector3D(0.0, 0.0, 0.0));
    }

    double const coeff_x = Mat33(p1.y, p1.z, 1.0, p2.y, p2.z, 1.0, p3.y, p3.z, 1.0).determinant();
    double const coeff_y = -Mat33(p1.x, p1.z, 1.0, p2.x, p2.z, 1.0, p3.x, p3.z, 1.0).determinant();
    
    double const value = -Mat33(p1.x, p1.y, p1.z, p2.x, p2.y, p2.z, p3.x, p3.y, p3.z).determinant();

    double const z_intersect_plane = -(coeff_x*point.x + coeff_y*point.y + value)/coeff_z;

    return std::pair(true, Vector3D(point.x, point.y, z_intersect_plane));
}

std::pair<bool, Vector3D> VoroCrustFace::calculateLinePlaneIntesection(Vector3D const& point, Vector3D const& line_point) const {
    //https://en.wikipedia.org/wiki/Line%E2%80%93plane_intersection

    Vector3D const& l = line_point - point;
    Vector3D const& normal = current_normal;
    
    double const l_dot_n = ScalarProd(l, normal);
    //! EPSILONTICA:
    // line is parallel to face
    if(std::abs(l_dot_n) < 1e-14){
        return std::pair(false, Vector3D(0.0, 0.0, 0.0));
    }

    Vector3D const& p0 = vertices[0]->vertex;
    double const d = ScalarProd(p0 - point, normal) / l_dot_n;

    // intersection point opposite to the direction of the line
    if(d < 0) {
        return std::pair(false, Vector3D(0.0, 0.0, 0.0));
    }

    return std::pair(true, point + d*l);
}

bool sameSide(Vector3D const& p1, Vector3D const& p2, Vector3D const& a, Vector3D const& b){
    // https://blackpawn.com/texts/pointinpoly/
    return ScalarProd(CrossProduct(b-a, p1-a), CrossProduct(b-a, p2-a)) >= 0;
}

bool VoroCrustFace::pointIsInsideFace(Vector3D const& point) const {
    Vector3D const& a = vertices[0]->vertex;
    Vector3D const& b = vertices[1]->vertex;
    Vector3D const& c = vertices[2]->vertex;

    return sameSide(point, a, b, c) && sameSide(point, b, a, c) && sameSide(point, c, a, b);
}

std::pair<bool, Vector3D> VoroCrustFace::isIntersectionBetweenLineAndPlaneIsInsideFace(Vector3D const& p1, Vector3D const& p2) const {
    auto const& [success, p_inter] = calculateLinePlaneIntesection(p1, p2);

    if(not success) return std::pair(false, Vector3D(0.0, 0.0, 0.0));

    return std::pair(pointIsInsideFace(p_inter), p_inter);
}

bool VoroCrustFace::isPointCompletelyOffFace(Vector3D const& p) const {
    auto const& v1 = vertices[0]->vertex;
    auto const& v2 = vertices[1]->vertex;
    auto const& v3 = vertices[2]->vertex;

    bool above_z = (p.z > v1.z && p.z > v2.z && p.z > v3.z);
    
    if(above_z) return above_z;

    bool below_x = (p.x < v1.x && p.x < v2.x && p.x < v3.x);

    if(below_x) return below_x;

    bool below_y = (p.y < v1.y && p.y < v2.y && p.y < v3.y);

    if(below_y) return below_y;

    bool above_x = (p.x > v1.x && p.x > v2.x && p.x > v3.x);

    if(above_x) return above_x;

    bool above_y = (p.y > v1.y && p.y > v2.y && p.y > v3.y);

    if(above_y) return above_y;

    return false;
}

std::string VoroCrustFace::repr() const
{
    std::ostringstream s;
    s << "\tEdges: ";
    for (auto const& edge : edges)
    {
        s << "\n\t\tedge " << edge->index << ": " << edge->repr() << ", ";
    }

    return s.str();
}