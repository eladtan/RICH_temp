#ifndef HILBERT_TREE_2D_HPP
#define HILBERT_TREE_2D_HPP

#include <iostream> // todo: remove
#include <vector>
#include <boost/container/small_vector.hpp>
#include "tessellation/geometry.hpp" // for Vector2D
#include "../hilbertTypes.h"

#define SIGN(x) ((x > 0) - (x < 0))

class HilbertTree2D
{
private:
    static constexpr int CHILDREN_MAX = 4; // can actually be 3
    using coord_t = double; // coordinate type

    class Node
    {
    public:
        Node(const Vector2D &ll = Vector2D(std::numeric_limits<coord_t>::max(), std::numeric_limits<coord_t>::max()),
             const Vector2D &ur = Vector2D(std::numeric_limits<coord_t>::min(), std::numeric_limits<coord_t>::min())): ll(ll), ur(ur){};

        inline bool contains(double x, double y) const{return (this->ll[0] <= x) and (x <= this->ur[0]) and (this->ll[1] <= y) and (y <= this->ur[1]);};
        inline bool contains(const Vector2D &point) const{return this->contains(point.x, point.y);};
        void calculateLLUR();

        Vector2D ll, ur; // box ll and ur
        hilbert_index_t d_start;
        hilbert_index_t d_end;
        bool isLeaf;
        boost::container::small_vector<Node*, CHILDREN_MAX> children;
    };

    Node *root;
    Vector2D ll, step, div;

public:
    explicit HilbertTree2D(const Vector2D &ll, const Vector2D &ur, size_t order);
    inline ~HilbertTree2D(){this->deleteTree();};
    
    inline void deleteTree(){this->deleteTreeHelper(this->root); this->root = nullptr;};
    inline hilbert_index_t getHilbertSize() const{return this->root->d_end;};
    hilbert_index_t xy2d(coord_t x, coord_t y) const;
    inline hilbert_index_t xy2d(const Vector2D &point){return this->xy2d(point.x, point.y);};
    Vector2D d2xy(hilbert_index_t d) const;

private:
    void buildBaseStep(Node *node, const std::pair<int, int> &startPoint, int length, const std::pair<int, int> &direction, hilbert_index_t d_start);
    void buildHelper(Node *node, const std::pair<int, int> &ll, const std::pair<int, int> &a, const std::pair<int, int> &b, hilbert_index_t d_start);
    Vector2D WidthHeightToXY(int width, int height) const;
    void deleteTreeHelper(Node *root);
};

#endif // HILBERT_TREE_2D_HPP