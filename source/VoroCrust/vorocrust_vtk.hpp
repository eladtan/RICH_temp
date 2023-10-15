#ifndef WRITE_VTU_PL_COMPLEX
#define WRITE_VTU_PL_COMPLEX

#include "PLC/PL_Complex.hpp"
#include "VoroCrust_kd_tree/VoroCrust_kd_tree.hpp"
#include "trees.hpp"
#include <filesystem>
#include "source/3D/elementary/Vector3D.hpp"
#include <memory>
#include <string>

namespace vorocrust_vtk{

/*! \brief writes a PLC as vtu file.
    \param filename path of output file.
    \param plc  PL_Complex to be exported as vtu. 
*/
void write_vtu_PL_Complex(std::filesystem::path const& filename, PL_Complex const& plc);

/*! \brief writes a vector of faces as a vtu file
*/
void write_vtu_faces(std::filesystem::path const& filename, std::vector<std::vector<Vector3D>> const& faces);

/*! \brief writes an arbitrary set of orienter `vectors` starting at `startPoints` */
void write_arbitrary_oriented_vectors(std::filesystem::path const& filename, 
                                      std::vector<Vector3D> const& startPoints, 
                                      std::vector<Vector3D> const& vectors, 
                                      std::string const& name, 
                                      double const factor);

/*! \brief writes the points in the boundary kd trees in `trees`*/
void write_vtu_trees(std::filesystem::path const& filename, 
                     Trees const& trees);

/*! \brief writes `tree.points` and `query` and paints query and the nearest neighbors to it in different colors */
void write_nearestNeighbor(std::filesystem::path const& filename, 
                           VoroCrust_KD_Tree const& tree, 
                           Vector3D const& query);

/*! \brief writes `tree.points` and `query` and paints query and the `k` nearest neighbors to it in different colors */
void write_kNearestNeighbors(std::filesystem::path const& filename, 
                             VoroCrust_KD_Tree const& tree, 
                             Vector3D const& query, 
                             int k);

/*! \brief writes `tree.points` and `query` and paints query and all the points up to a distance of radius from query in different colors*/
void write_radiusSearch(std::filesystem::path const& filename,
                        VoroCrust_KD_Tree const& tree,
                        Vector3D const& query,
                        double const radius);

/*! \brief writes `tree.points` and `segment` (where segment is made from equidistributed `num_points_on_segment` points) and finds the closest point to segment*/
void write_nearestNeighborToSegment(std::filesystem::path const& filename,
                                    VoroCrust_KD_Tree const& tree,
                                    std::array<Vector3D, 2> const& segment,
                                    std::size_t const num_points_on_segment);

/*! \brief writes a ball tree as spheres each has a center `b_tree.points[i]` and radius `b_tree.ball_radii[i]` */
void write_ballTree(std::filesystem::path const& filename, 
                    VoroCrust_KD_Tree_Ball const& b_tree);

/*! \brief write arbitrary points to file*/
void write_points(std::filesystem::path const& filename, std::vector<Vector3D> const& point_vectors);

} // namespace vorocrust_vtk

#endif /* WRITE_VTU_PL_COMPLEX */