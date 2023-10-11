#ifndef VOROCRUST_FACE_HPP
#define VOROCRUST_FACE_HPP 1

#include <vector>
#include "source/3D/elementary/Vector3D.hpp"

#include "VoroCrustUsing.hpp"
#include "VoroCrustVertex.hpp"

class VoroCrustFace
{
    public:
        //! \brief Vertices defining the face
        std::vector<VoroCrust::Vertex> vertices;
        //! \brief Edges on the boundary of the Face
        std::vector<VoroCrust::Edge> edges;
        
        //! \brief index in the PLC Faces vector
        std::size_t index;

        //! \brief the current calculated normal to the face.
        Vector3D current_normal;

        //! \brief the centeroid
        Vector3D centeroid;

        //! \brief true if Face was already assigned to a Patch
        bool isPatched;
        
        //! \brief patch index in the PL_Complex `patches` vector
        std::size_t patch_index;

        VoroCrustFace(std::vector<VoroCrust::Vertex> const& vertices_, std::size_t const index_);
        
        //! \brief adds an Edge to the Face's Edges vector.
        void addEdge(VoroCrust::Edge const& edge);

        /*! \brief calculate the normal to the surface defined by the vectors `vertices[2] - vertices[1]` X `vertices[1]-vertices[0]`*/
        Vector3D calcNormal();

        /*! \brief returns current normal as a const& */
        Vector3D const& getNormal() { return current_normal; }

        //! \brief calculates the Centeroid of the face
        Vector3D calcCenteroid();

        /*! \brief returns the centeroid as a const& */
        Vector3D const& getCenteroid() { return centeroid; }

        /*! \brief calculates the signed ared using the formula in the answer at https://math.stackexchange.com/questions/3207981/how-do-you-calculate-the-area-of-a-2d-polygon-in-3d */
        double calcSignedArea();

        /*! \brief return the Area of the face (absolute value of the signed area) */
        double calcArea();

        //! \brief flips the orientation of the face i.e. reverse the vector `vertices`
        void flipOrientation();

        //! \brief orient `this` with respect to `face` (normals scalar product is positive) 
        void orientWithRespectTo(VoroCrust::Face const& face);

        /*! \brief finds the intersection of the ray coming out of `point` in the positive z direction with the plane defined by the face.
            \return a pair <success, point> where `success` is a flag indicating if there exists such an intersection point, and `point` is the intersection point with the plane (assuming success) 
        */
        std::pair<bool, Vector3D> pointZparallelRayIntersectsAt(Vector3D const& point) const;

        /*! \brief calculates the intersection on the line `point + d*(line_point - point)` with the plane defined by the face.
            \return a pair <success, point>. `success` is false if line is parallel to the plane or if the intersection is in the opposite direction of the line
        */
        std::pair<bool, Vector3D> calculateLinePlaneIntesection(Vector3D const& point, Vector3D const& line_point) const;

        //! \brief returns `true` if point is inside face
        bool pointIsInsideFace(Vector3D const& point) const;

        //! \brief `first` is true if the intersection between the line `p1 + d*(p2 - p1)` with the plane defined by face is inside the face. `second` is the point of intersection.
        std::pair<bool, Vector3D> 
        isIntersectionBetweenLineAndPlaneIsInsideFace(Vector3D const& p1, Vector3D const& p2) const;

        //! \brief returns `true` if point is completely off face (all vertices x or y are above or below the point x or y)
        bool isPointCompletelyOffFace(Vector3D const& p) const;

        std::string repr() const;

};

#endif /* VOROCRUST_FACE_HPP */