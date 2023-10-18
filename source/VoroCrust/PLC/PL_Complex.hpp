#ifndef PL_Complex_HPP
#define PL_Complex_HPP 1

#include <sstream>
#include <string>
#include <vector>
#include <memory>

#include "VoroCrustVertex.hpp"
#include "VoroCrustEdge.hpp"
#include "VoroCrustFace.hpp"
#include "../VoroCrust_kd_tree/VoroCrust_kd_tree.hpp"

//! \brief a Crease is a chain of sharp edges, ends in a sharp corner or forms a cycle. 
using Crease = std::vector<VoroCrust::Edge>;

//! \brief a Surface patch is the connected componnent containing no sharp features. A Surface patch is enveloped by Creases.
struct SurfacePatch {
    std::vector<VoroCrust::Face> patch;
    std::vector<std::size_t> patch_creases;
    std::vector<std::size_t> patch_corners;

    SurfacePatch() : patch(), patch_creases(), patch_corners() {}

    void findCreasesAndCorners();
    void push_back(VoroCrust::Face const& new_face) { patch.push_back(new_face); }

    VoroCrust::Face const& operator[](std::size_t const index) const { return patch[index]; }
    VoroCrust::Face& operator[](std::size_t const index)  { return patch[index]; }

    // so it can be used in the range for loops
    auto begin() { return patch.begin();}
    auto end() { return patch.end();}
    auto begin() const { return patch.cbegin();}
    auto end() const { return patch.cend();}
};

/*! \brief Piecewise Linear Complex 
    \details Holds the mesh and does the preprocessing of the VoroCrust algorithm.
*/
class PL_Complex 
{
    public:
        enum class Location {
            IN,
            OUT
        };

        //! \brief the PLC mesh vertices.
        std::vector<VoroCrust::Vertex> vertices;
        //! \brief the PLC mesh edges.
        std::vector<VoroCrust::Edge> edges;
        //! \brief holds the edges vertices in a kd-tree to speed up the search for already defined edges
        VoroCrust_KD_Tree_Boundary edges_tree_vertex1, edges_tree_vertex2;
        //! \brief the PLC mesh faces.
        std::vector<VoroCrust::Face> faces;
        //! \brief holds the x, y va;ue of the centeroid of the face
        VoroCrust_KD_Tree_Ball centeroids_z;

        //! \brief holds the sharp edges of the PLC.
        std::vector<VoroCrust::Edge> sharp_edges;
        //! \brief holds the sharp corners (sharp vertices) of the PLC.
        std::vector<VoroCrust::Vertex> sharp_corners;

        //! \brief holds the Creases of the PLC.
        std::vector<Crease> creases;
        //! \brief holds the Surface Patches of the PLC.
        std::vector<SurfacePatch> patches;

        /*! \brief Constructor, consturcts a PLC from a given set of Vectors representing the vertices of the mesh.
            \param vertices a vector<Vector3D> of the initial vertices of the mesh.
        */
        PL_Complex(std::vector<Vector3D> const& vertices);
        
        /*! \brief adds an edge to the PLC starting at v1 and ending at v2.
            \param v1 first Vertex of Edge.
            \param v2 second Vertex of Edge.
        */
        VoroCrust::Edge addEdge(VoroCrust::Vertex const& v1, VoroCrust::Vertex const& v2);
        
        /*! \brief adds a face to the PLC
            \param indices vector of indices of the `vertices` of the Face.
        */
        void addFace(std::vector<std::size_t> const& indices);

        /*! \brief Checks if all vertices are unique. */
        bool checkAllVerticesAreUnique();

        /*! \brief Checks if all the vertices are assigned at least on face.*/
        bool checkAllVerticesAreOnFace();

        /*! \brief Determines the sharp features of the PLC (sharp_edges, sharp_corners) and using them build the Creases and Surface Patches.
            \param sharpTheta determins the sharp features.
            \param flatTheta constraint on the flatness of the non-sharp features.
        */
        void detectFeatures(double const sharpTheta, double const flatTheta);

        /*! \brief builds the Creases */
        void buildCreases();

        /*! \brief builds a Crease s.t. `edge` is in it.
            \param edge 
        */
        Crease createCrease(VoroCrust::Edge const& edge);
        
        /*! \brief builds the Surface Patches */
        void buildSurfacePatches();
        
        /*! \brief create a Surface Patch s.t. face is in it 
            \param face
        */
        SurfacePatch createSurfacePatch(VoroCrust::Face const& face);

        /*! \brief returns the bounding box of the PL_Complex
            \return array of 6 {ll_x, ll_y , ll_z, ur_x, ur_y, ur_z} where ll := 'lower left', ur := 'upper right'
        */
        std::array<double, 6> getBoundingBox() const;

        /*! \brief determine if a point p is inside or outside the PL_Complex 
            \return if point is inside return PL_Complex::Location::IN else return PL_Complex::Location::OUT 
        */
        Location determineLocation(Vector3D const& p) const; 
        
        std::string repr() const;

        private:
            //! \brief Orders the crease s.t. crease[0]->vertex1 is the start and crease[end]->vertex2 is the end and crease[i]->vertex2 == crease[i+1]->vertex1
            void orderCrease(Crease &crease);

            /*! \brief divides the Faces vector of the vertices and edges to surface patches */
            void divideFacesOfVerticesAndEdgesToPatches();

            /*! \brief calculate the normals of all the faces in the PLC*/
            void calcNormalsAndCenteroidsOfAllFaces();
};

/*! \brief a utility function which takes a vector of faces divides it to surface patches
    \param faces vector of faces
    \return vector<vector<Face>> each face in vector<Face> is on the same surface patch
*/
std::vector<std::vector<VoroCrust::Face>> divideFacesToPatches(std::vector<VoroCrust::Face> const& faces);

#endif /* PL_Complex_HPP */