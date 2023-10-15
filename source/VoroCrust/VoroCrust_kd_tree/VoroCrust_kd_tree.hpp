#ifndef VOROCRUST_KD_TREE
#define VOROCRUST_KD_TREE

#include <memory>
#include <vector>
#include <algorithm>
#include <iostream>
#include "source/3D/elementary/Vector3D.hpp"

#define DIM 3

/*! \brief a Node in a VoroCrust_KD_Tree
*/
struct Node;
using NodePtr = std::shared_ptr<Node>;

struct Node
{
    //! \brief index of point in the VoroCrust_KD_Tree::points vector
    std::size_t index;
    //! \brief depth % DIM. depth of node is depth in the tree
    std::size_t axis;
    //! \brief pointers to left and right nodes in tree
    NodePtr left, right;

    Node() : index(0), axis(0), left(nullptr), right(nullptr) {}
};

NodePtr newNode(std::size_t index, std::size_t axis);

class VoroCrust_KD_Tree {
    public:
        //! \brief pointer to root node of the tree  
        NodePtr root;
        
        //! \brief holds the actual points of the tree
        std::vector<Vector3D> points;

        //! \brief construct an empty VoroCrust_KD_Tree
        VoroCrust_KD_Tree();

        //! \brief construct a VoroCrust_KD_Tree from a given vector of points
        VoroCrust_KD_Tree(std::vector<Vector3D> const& points);
        
        virtual ~VoroCrust_KD_Tree() = default;

        //! \brief make the VoroCrust_KD_Tree from a given vector of points
        void makeTree(std::vector<Vector3D> const& points_);

        //! \brief clear and erase the tree
        void clear();

        //! \brief insert a new point to the tree
        void insert(Vector3D const& point);

        //! \brief remakes the tree (used after a lot of new insertions)
        void remakeTree();

        //! \brief finds nearest neighbor in tree to `query`
        std::size_t nearestNeighbor(Vector3D const& query) const;

        double distanceToNearestNeighbor(Vector3D const& query, double const initial_min_dist) const;

        /*! \brief finds the `k` nearest neighbors to `query` in the tree 
            \param query
            \param k number of nearest neighbors */
        std::vector<std::size_t> kNearestNeighbors(Vector3D const& query, std::size_t const k) const;
        
        //! \brief finds all points with a distance to query that is less than radius
        std::vector<std::size_t> radiusSearch(Vector3D const& query, double const radius) const;

        //! \brief finds the nearest point in the tree to some given `segment`
        std::size_t nearestNeighborToSegment(std::array<Vector3D, 2> const& segment) const;

        //! \brief checks if two trees are equal
        bool operator==(VoroCrust_KD_Tree const& tree) const;

        bool empty() const { return points.empty(); }

        std::size_t size() const { return points.size(); }

    private:
        //! \brief builds the tree recursively
        NodePtr buildRecursive(std::size_t *indices, std::size_t npoints, std::size_t depth);

        //! \brief recursively checks where to insert the new point
        void insertRecursive(Vector3D const& point, NodePtr const& node);

        //! \brief finds nearest neighbor recursively in tree to `query`
        void nearestNeighborRecursive(Vector3D const& query, NodePtr const& node, std::size_t &guess, double &minDist) const;
        
        //! \brief finds the `k` nearest neighbors in the tree recursively
        void kNearestNeighborsRecursive(Vector3D const& query, std::size_t const k, NodePtr const& node, std::vector<std::size_t>& indices, std::vector<double> &minDist) const;        
        
        //! \brief finds all points with a distance to query that is less than radius recursively
        void radiusSearchRecursive(Vector3D const& query, double const radius, NodePtr const& node, std::vector<std::size_t> &indices) const;

        //! \brief finds the nearest point in the tree for some given `segment` recursively
        void nearestNeighborToSegmentRecursive(std::array<Vector3D, 2> const& segment, NodePtr const& node, std::size_t &guess, double &minDist) const;
        
        //! \brief calculates the distance of the `point` from a `segment`
        double distancePointToSegment(std::array<Vector3D, 2> const& segment, Vector3D const& point) const;
        
        //! \brief checks if two trees are equal recursively
        bool equalRecursive(NodePtr const& node1, NodePtr const& node2, VoroCrust_KD_Tree const& t) const;
};

//! \brief a boundary tree for F_C, F_E and T_S
class VoroCrust_KD_Tree_Boundary : public VoroCrust_KD_Tree {
    public:
        std::vector<Vector3D> vectors;
        std::vector<std::size_t> feature_index;
        std::vector<std::size_t> plc_index;

        VoroCrust_KD_Tree_Boundary();

        VoroCrust_KD_Tree_Boundary(std::vector<Vector3D> const& points) : VoroCrust_KD_Tree(points) {}

        VoroCrust_KD_Tree_Boundary(std::vector<Vector3D> const& points, 
                                   std::vector<Vector3D> const& vecs, 
                                   std::vector<std::size_t> const& feature_index_, 
                                   std::vector<std::size_t> const& plc_index_);

        virtual ~VoroCrust_KD_Tree_Boundary() = default;

        void insert(Vector3D const& point, Vector3D const& vec, std::size_t f_index, std::size_t plc_index_);

        //! \brief finds the nearest point in the tree wthat is not cosmooth with `query`
        long nearestNonCosmoothPoint(Vector3D const& query, 
                                     std::vector<Vector3D> const& vecs, 
                                     std::size_t const f_index, 
                                     double const angle, 
                                     double const initial_min_dist) const;

        double distanceToNearestNonCosmoothPoint(Vector3D const& query, 
                                                 std::vector<Vector3D> const& vecs, 
                                                 std::size_t const f_index, 
                                                 double const angle, 
                                                 double const initial_min_dist) const;

        //! \brief find the nearest neighbor excluding some given features
        long nearestNeighborExcludingFeatures(Vector3D const& query, 
                                              std::vector<std::size_t> const& to_exclude, 
                                              double const initial_min_dist) const;

        double distanceToNearestNeighborExcludingFeatures(Vector3D const& query,                                        
                                                          std::vector<std::size_t> const& to_exclude, 
                                                          double const initial_min_dist) const;

    private:

        //! \brief finds the nearest non cosmooth point to `query` recursively 
        void nearestNonCosmoothPointRecursive(Vector3D const& query, 
                                              std::vector<Vector3D> const& vecs, 
                                              std::size_t const f_index, 
                                              double const angle, 
                                              NodePtr const& node, 
                                              long &guess, 
                                              double &minDist) const;

        void nearestNeighborExcludingFeaturesRecursive(Vector3D const& query, 
                                                       std::vector<std::size_t> const& to_exclude, 
                                                       NodePtr const& node, 
                                                       long &guess, 
                                                       double &minDist) const;
};

class VoroCrust_KD_Tree_Ball : public VoroCrust_KD_Tree_Boundary {
    public:
        std::vector<double> ball_radii;

        VoroCrust_KD_Tree_Ball();

        VoroCrust_KD_Tree_Ball(std::vector<Vector3D> const& points, 
                               std::vector<Vector3D> const& vecs, 
                               std::vector<std::size_t> const& feature_index_, 
                               std::vector<std::size_t> const& plc_index_,
                               std::vector<double> const& radii);

        virtual ~VoroCrust_KD_Tree_Ball() = default;

        void insert(Vector3D const& point, Vector3D const& vec, double const radius, std::size_t const f_index, std::size_t const plc_index_);

        //! \brief returns the overlapping balls of ball defined by `center` `radius`, only overlapping balls with centers up to `r_max`
        std::vector<std::size_t> getOverlappingBalls(Vector3D const& center, 
                                                     double const radius, 
                                                     double const r_max) const;

        //! \brief returns the ball at `index`
        std::pair<Vector3D, double> getBall(std::size_t const index) const { return std::pair(points[index], ball_radii[index]); }

        //! \brief returns the nearest Ball to p as a pair <center, radius>
        std::pair<Vector3D, double> getBallNearestNeighbor(Vector3D const& p) const {return getBall(nearestNeighbor(p));}

        //! \brief return true if p is contained in the ball with the nearest center 
        bool isContainedInNearestBall(Vector3D const& p) const {
            auto const& [center, radius] = getBallNearestNeighbor(p);
            return distance(p, center) < radius;
        }        

        bool isContainedInBall(Vector3D const& p) const {
            auto const& suspects = radiusSearch(p, max_radius*1.1);
            for(auto const index : suspects){
                auto const& [center, radius] = getBall(index);

                if(distance(p, center) < radius*1.1){
                    return true;
                }
            }
            
            return false;
        }

        double getMaxRadius() const { return max_radius; }

    private:
        double max_radius;
};



#endif /* VOROCRUST_KD_TREE */