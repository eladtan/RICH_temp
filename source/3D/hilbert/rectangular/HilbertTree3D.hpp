#ifndef HILBERT_TREE_3D_HPP
#define HILBERT_TREE_3D_HPP

#include <iostream> // todo: remove
#include <vector>
#include <boost/container/small_vector.hpp>
#include "3D/elementary/Vector3D.hpp" // for Vector3D
#include "../hilbertTypes.h"

class HilbertTree3D
{
private:
    static constexpr int CHILDREN_MAX = 8; // can actually be 5
    using coord_t = Vector3D::coord_type; // coordinate type

    struct DirectionVector3D
    {
        int x, y, z;
    };

    class Node
    {
    public:
        Node(const Vector3D &ll = Vector3D(std::numeric_limits<coord_t>::max(), std::numeric_limits<coord_t>::max(), std::numeric_limits<coord_t>::max()),
             const Vector3D &ur = Vector3D(std::numeric_limits<coord_t>::min(), std::numeric_limits<coord_t>::min(), std::numeric_limits<coord_t>::min())): ll(ll), ur(ur){};

        inline bool contains(coord_t x, coord_t y, coord_t z) const{return (this->ll[0] <= x) and (x <= this->ur[0]) and (this->ll[1] <= y) and (y <= this->ur[1]) and (this->ll[2] <= z) and (z <= this->ur[2]);};
        inline bool contains(const Vector3D &point) const{return this->contains(point.x, point.y, point.z);};
        void calculateLLUR();

        Vector3D ll, ur; // box ll and ur
        hilbert_index_t d_start;
        hilbert_index_t d_end;
        bool isLeaf;
        boost::container::small_vector<Node*, CHILDREN_MAX> children;
    };

    Node *root;
    Vector3D ll, step;

public:
    explicit HilbertTree3D(const Vector3D &ll, const Vector3D &ur, size_t order);
    inline ~HilbertTree3D(){this->deleteTree();};
    
    inline void deleteTree(){this->deleteTreeHelper(this->root); this->root = nullptr;};
    inline hilbert_index_t getHilbertSize() const{return this->root->d_end;};
    hilbert_index_t xyz2d(coord_t x, coord_t y, coord_t z) const;
    inline hilbert_index_t xyz2d(const Vector3D &point){return this->xyz2d(point.x, point.y, point.z);};
    Vector3D d2xyz(hilbert_index_t d) const;

private:
    void buildBaseStep(Node *node, const DirectionVector3D &startPoint, int length, const DirectionVector3D &direction, hilbert_index_t &d);
    void buildHelper(Node *node, const DirectionVector3D &ll, const DirectionVector3D &a, const DirectionVector3D &b, const DirectionVector3D &c, hilbert_index_t &d);
    Vector3D WidthHeightDepthToXYZ(int width, int height, int depth) const;
    void deleteTreeHelper(Node *root);
};

#endif // HILBERT_TREE_3D_HPP