#ifndef VOROCRUST_VERTEX_HPP
#define VOROCRUST_VERTEX_HPP 1


#include <vector>
#include <string>
#include "source/3D/elementary/Vector3D.hpp"

#include "VoroCrustUsing.hpp"
#include "VoroCrustFace.hpp"
#include "VoroCrustEdge.hpp"


class VoroCrustVertex
{
    public:
        //! \brief Vector3D representing the Vertex in 3D space
        Vector3D vertex; 
        
        //! \brief Edges incident to the Vertex.
        std::vector<VoroCrust::Edge> edges;

        //! \brief Faces incident to the Vertex.
        std::vector<VoroCrust::Face> faces;

        //! \brief group faces to surface patches
        std::vector<std::vector<VoroCrust::Face>> divided_faces;

        //! \brief index of Vertex in PLC vertices vector.
        std::size_t index;

        //! \brief true if Vertex is a sharp corner.
        bool isSharp;

        VoroCrustVertex(Vector3D const& vertex_, std::size_t const index_);

        //! \brief adds a Face to Vertex's `faces` vector.
        void addFace(VoroCrust::Face const& new_face);
        
        //! \brief adds a Face to Vertex's `edges` vector.
        void addEdge(VoroCrust::Edge const& new_edge);

        std::string repr() const;
};
#endif /* VOROCRUST_VERTEX_HPP */