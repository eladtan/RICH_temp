#ifndef _OCTTREE_HPP
#define _OCTTREE_HPP

#include <vector>
#include <assert.h>
#include <utility>
#include "../geometry_utils.hpp"
#include <mpi.h> // todo: remove

#define DIM 3
#define CHILDREN 8 // 2^DIM
#define PATH_END_DIRECTION (-1)
#define MAX_DEPTH 50
#define DEBUG_MODE

#ifdef DEBUG_MODE
#include <iostream>
#endif // DEBUG_MODE

typedef char direction_t;

template<typename T>
class OctTree
{
    // todo: necessary?
    template<typename U>
    class DistributedOctTree;
    template<typename U>
    friend class DistributedOctTree;

public:
    class OctTreeNode
    {
        friend class OctTree;

    public:
        inline OctTreeNode(const T &ll, const T &ur): isValue(false), value((ll + ur)/2), boundingBox(_BoundingBox(ll, ur)), parent(nullptr), height(0), depth(0)
        {
            for(int i = 0; i < CHILDREN; i++)
            {
                this->children[i] = nullptr;
            }
        }

        inline OctTreeNode(const T &point): isValue(true), value(point), boundingBox(_BoundingBox(point, point)), parent(nullptr), height(0), depth(0)
        {
            for(int i = 0; i < CHILDREN; i++)
            {
                this->children[i] = nullptr;
            }
        }

        inline OctTreeNode(OctTreeNode &&other): isValue(other.isValue), value(other.value), boundingBox(other.boundingBox)
        {
            for(int i = 0; i < CHILDREN; i++)
            {
                this->children[i] = other.children[i];
                other.children[i] = nullptr;
            }
            this->parent = other.parent;
            other.parent = nullptr;
        }

        OctTreeNode(OctTreeNode *parent, int childNumber);
        virtual ~OctTreeNode() = default;

        int getChildNumberContaining(const T &point) const;
        const OctTreeNode *getChildContaining(const T &point) const{return this->children[this->getChildNumberContaining(point)];};
        virtual OctTreeNode *addLeafChild(int childIndex, const T &point);
        virtual OctTreeNode *createChild(int childNumber);

        virtual inline void print() const
        {
            std::cout << this->value << ", BB: " << this->boundingBox.ll << ", " << this->boundingBox.ur << " (depth: " << this->depth << ", height: " << this->height << ")" << std::endl;
        }

        bool isValue;
        T value; // if a leaf, that's a point value, otherwise, thats the value for partition
        _BoundingBox<T> boundingBox; // the bounding box this node induces
        OctTreeNode *children[CHILDREN];
        OctTreeNode *parent;
        int height; // height of a leaf is 0
        int depth; // depth of the root is 0
    
    protected:
        void fixHeightsRecursively();
        void splitNode();
    };

protected:
    void deleteSubtree(OctTreeNode *node);

    const OctTreeNode *tryFind(const T &point) const;
    inline OctTreeNode *tryFind(const T &point){return const_cast<OctTreeNode*>(std::as_const(*this).tryFind(point));};
    const OctTreeNode *tryFindParent(const T &point) const;
    inline OctTreeNode *tryFindParent(const T &point){return const_cast<OctTreeNode*>(std::as_const(*this).tryFindParent(point));};
    virtual OctTreeNode *tryInsert(const T &point);

    #ifdef DEBUG_MODE
    void printHelper(const OctTreeNode *node, int indent) const;
    #endif // DEBUG_MODE

    void getAllDecendantsHelper(const OctTreeNode *node, std::vector<T> &result) const;
    inline std::vector<T> getAllDecendants(const OctTreeNode *node) const
    {
        std::vector<T> result;
        this->getAllDecendantsHelper(node, result);
        return result;
    };

    void rangeHelper(const OctTreeNode *node, const _Sphere<T> &sphere, std::vector<T> &result) const;
    void getClosestPointHelper(const T &point, const OctTreeNode *node, T &closestPoint, typename T::coord_type &cloesestDistance) const;
    
    OctTreeNode *root;
    size_t treeSize;

public:
    explicit OctTree(const T &ll, const T &ur): root(nullptr), treeSize(0){this->setBounds(ll, ur);};
    template<typename InputIterator>
    explicit OctTree(const T &ll, const T &ur, const InputIterator &first, const InputIterator &last): OctTree(ll, ur)
    {
        for(InputIterator it = first; it != last; it++)
        {
            this->insert(*it);
        }
    };
    template<typename Container>
    inline OctTree(const T &ll, const T &ur, Container container): OctTree(ll, ur, container.begin(), container.end()){};
    inline explicit OctTree(): root(nullptr), treeSize(0){};
    virtual inline ~OctTree(){this->deleteSubtree(this->getRoot());};

    inline bool insert(const T &point)
    {
        if(this->tryInsert(point) != nullptr)
        {
            this->treeSize++;
            return true;
        }
        return false;
    };
    inline bool find(const T &point) const{return this->tryFind(point) != nullptr;};

    virtual inline OctTreeNode *getRoot(){return this->root;};
    virtual inline const OctTreeNode *getRoot() const{return this->root;};
    virtual inline void setRoot(OctTreeNode *other){this->root = other;};
    virtual void setBounds(const T &ll, const T &ur)
    {
        assert(this->getRoot() == nullptr);
        this->setRoot(new OctTreeNode(ll, ur));
        this->getRoot()->parent = nullptr;
    }

    #ifdef DEBUG_MODE
    void print() const{this->printHelper(this->getRoot(), 0);};
    #endif // DEBUG_MODE

    inline int getDepth() const{assert(this->getRoot() != nullptr); return this->getRoot()->height;};
    inline size_t getSize() const{return this->treeSize;};
    inline std::vector<T> range(const _Sphere<T> &sphere) const
    {
        std::vector<T> result;
        this->rangeHelper(this->getRoot(), sphere, result);
        return result;
    };

    inline const OctTreeNode *getNodeByDirections(const direction_t *directions) const
    {
        if(directions == nullptr)
        {
            return nullptr;
        }
        int rank; MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        const OctTreeNode *current = this->getRoot();
        size_t i = 0;
        while(directions[i] != PATH_END_DIRECTION)
        {
            if(current == nullptr)
            {
                break;
            };
            current = current->children[directions[i]];
            i++;
        }

        assert(current != nullptr);
        if(current == nullptr)
        {
            std::cerr << "Illegal path in rank " << rank << std::endl;
            size_t j = 0;
            std::cout << "path is ";
            while(directions[j] != PATH_END_DIRECTION){std::cout << directions[j++] << " ";};  std::cout << std::endl;
            exit(8200);
        }
        return current;
    }

    inline T closestPoint(const T &point) const
    {
        T closestPoint;
        typename T::coord_type closestDistance = std::numeric_limits<typename T::coord_type>::max();
        this->getClosestPointHelper(point, this->getRoot(), closestPoint, closestDistance);
        return closestPoint;
    }

    inline double closestPointDistance(const T &point) const
    {
        T closestPoint;
        typename T::coord_type closestDistance = std::numeric_limits<typename T::coord_type>::max();
        this->getClosestPointHelper(point, this->getRoot(), closestPoint, closestDistance);
        return closestDistance;
    }
};

template<typename T>
void OctTree<T>::deleteSubtree(OctTreeNode *node)
{
    if(node == nullptr)
    {
        return;
    }
    for(int i = 0; i < CHILDREN; i++)
    {
        this->deleteSubtree(node->children[i]);
    }
    delete node;
}

template<typename T>
OctTree<T>::OctTreeNode::OctTreeNode(OctTreeNode *parent, int childNumber): isValue(false), parent(parent), depth(0)
{
    assert(parent != nullptr);

    // determine box:
    for(int i = 0; i < DIM; i++)
    {
        if((childNumber >> ((DIM - 1) - i)) & 1)
        {
            this->boundingBox.ll[i] = (parent->boundingBox.ll[i] + parent->boundingBox.ur[i]) / 2;
            this->boundingBox.ur[i] = parent->boundingBox.ur[i];
        }
        else
        {
            this->boundingBox.ll[i] = parent->boundingBox.ll[i];
            this->boundingBox.ur[i] = (parent->boundingBox.ll[i] + parent->boundingBox.ur[i]) / 2;
        }
        this->value[i] = (this->boundingBox.ll[i] + this->boundingBox.ur[i]) / 2;
    }

    for(int i = 0; i < CHILDREN; i++)
    {
        this->children[i] = nullptr;
    }
    this->fixHeightsRecursively();
}

template<typename T>
typename OctTree<T>::OctTreeNode *OctTree<T>::OctTreeNode::addLeafChild(int childIndex, const T &point)
{
    this->children[childIndex] = new OctTreeNode(point);
    this->children[childIndex]->parent = this;
    this->children[childIndex]->fixHeightsRecursively();
    return this->children[childIndex];
}

template<typename T>
typename OctTree<T>::OctTreeNode *OctTree<T>::OctTreeNode::createChild(int childNumber)
{
    assert(this->children[childNumber] == nullptr);
    this->children[childNumber] = new OctTreeNode(this, childNumber);
    return this->children[childNumber];
}

template<typename T>
int OctTree<T>::OctTreeNode::getChildNumberContaining(const T &point) const
{
    assert(this->boundingBox.contains(point));
    int direction = 0;
    for(int i = 0; i < DIM; i++)
    {
        direction = (direction << 1) | ((this->value[i] < point[i])? 1 : 0);
    }
    return direction;
}

template<typename T>
const typename OctTree<T>::OctTreeNode *OctTree<T>::tryFindParent(const T &point) const
{
    const OctTreeNode *current = this->getRoot();
    while(current != nullptr)
    {
        if(current->isValue)
        {
            if(current->value == point)
            {
                return current;
            }
            return nullptr;
        }
        // otherwise, determine the direction to go
        const OctTreeNode *nextChild = current->getChildContaining(point);
        if(nextChild == nullptr)
        {
            return current;
        }
        current = nextChild;
    }
    return nullptr;
}

template<typename T>
const typename OctTree<T>::OctTreeNode *OctTree<T>::tryFind(const T &point) const
{
    const OctTreeNode *current = this->getRoot();
    while(current != nullptr)
    {
        if(current->isValue)
        {
            if(current->value == point)
            {
                return current;
            }
            return nullptr;
        }
        // otherwise, determine the direction to go
        current = current->getChildContaining(point);
    }
    return nullptr;
}

template<typename T>
void OctTree<T>::OctTreeNode::fixHeightsRecursively()
{
    if(this->parent == nullptr)
    {
        this->depth = 0;
        return;
    }
    bool leaf = true;
    for(int i = 0; i < CHILDREN; i++)
    {
        if(this->children[i] != nullptr)
        {
            leaf = false;
            break;
        }
    }
    if(leaf)
    {
        this->height = 0;
    }
    this->parent->fixHeightsRecursively();

    this->parent->height = std::max<int>(this->parent->height, this->height + 1);
    this->depth = this->parent->depth + 1;
}

template<typename T>
void OctTree<T>::OctTreeNode::splitNode()
{
    assert(this->parent != nullptr);
    assert(this->isValue);

    int i;
    for(i = 0; i < CHILDREN; i++)
    {
        if(this == this->parent->children[i])
        {
            break;
        }
    }

    // this node is the `i`th child of its parent
    // replace it with a new (non-value) node, which will be our parent
    this->parent->children[i] = nullptr;
    this->parent->createChild(i);
    
    this->parent = this->parent->children[i];
    this->parent->isValue = false;
    
    int myIndex = this->parent->getChildNumberContaining(this->value);
    this->parent->children[myIndex] = this; 
    this->fixHeightsRecursively();
}

#ifdef DEBUG_MODE
template<typename T>
void OctTree<T>::printHelper(const OctTreeNode *node, int indent) const
{
    if(node == nullptr)
    {
        std::cout << "nullptr" << std::endl;
        return;
    }
    if(node->isValue)
    {
        std::cout << node->value << std::endl;
    }
    else
    {
        node->print();
    }
    int minNull = -1;
    for(int i = 0; i < CHILDREN - 1; i++)
    {
        if(node->children[i] != nullptr)
        {
            if(minNull != -1)
            {
                for(int j = 0; j < indent; j++) std::cout << "\t";
                if(minNull == i - 1)
                {
                    std::cout << "[" << i-1 << "] nullptr" << std::endl;
                }
                else
                {
                    std::cout << "[" << minNull << " - " << i-1 << "] nullptr" << std::endl;
                }
                minNull = -1;
            }
            for(int j = 0; j < indent; j++) std::cout << "\t";
            std::cout << "[" << i << "] ";
            this->printHelper(node->children[i], indent + 1);
        }
        else
        {
            if(minNull == -1) minNull = i;
        }
    }
    if(minNull == -1)
    {
        for(int j = 0; j < indent; j++) std::cout << "\t";
        std::cout << "[" << (CHILDREN - 1) << "] ";
        this->printHelper(node->children[CHILDREN - 1], indent + 1);
    }
    else
    {
        if(node->children[CHILDREN - 1] != nullptr)
        {
            for(int j = 0; j < indent; j++) std::cout << "\t";
            std::cout << "[" << minNull << " - " << (CHILDREN - 2) << "] nullptr" << std::endl;
            for(int j = 0; j < indent; j++) std::cout << "\t";
            std::cout << "[" << (CHILDREN - 1) << "] ";
            this->printHelper(node->children[CHILDREN - 1], indent + 1);
        }
        else
        {
            for(int j = 0; j < indent; j++) std::cout << "\t";
            std::cout << "[" << minNull << " - " << (CHILDREN - 1) << "] nullptr" << std::endl;
        }
    }
}
#endif // DEBUG_MODE

template<typename T>
typename OctTree<T>::OctTreeNode *OctTree<T>::tryInsert(const T &point)
{
    assert(this->getRoot() != nullptr);
    if(!this->getRoot()->boundingBox.contains(point))
    {
        return nullptr;
    }

    OctTreeNode *current = this->getRoot();
    while(current != nullptr)
    {
        // if we reached a leaf with the value `v`, start splitting until `v` and `point` are not in the same rectangle
        while(current->isValue)
        {
            if(current->value == point)
            {
                return current;
            }
            current->splitNode();
            current = current->parent;
            int childIndex = current->getChildNumberContaining(point);
            if(current->children[childIndex] == nullptr)
            {
                break;
            }
            current = current->children[childIndex];
        }
        // otherwise, determine the direction to go

        int childIndex = current->getChildNumberContaining(point);
        if(current->children[childIndex] == nullptr)
        {
            return current->addLeafChild(childIndex, point);
        }
        current = current->children[childIndex];
    }
    return nullptr;
}

template<typename T>
void OctTree<T>::getAllDecendantsHelper(const OctTreeNode *node, std::vector<T> &result) const
{
    if(node == nullptr)
    {
        return;
    }
    if(node->isValue)
    {
        result.push_back(node->value);
    }
    for(int i = 0; i < CHILDREN; i++)
    {
        this->getAllDecendantsHelper(node->children[i], result);
    }
}

template<typename T>
void OctTree<T>::rangeHelper(const OctTreeNode *node, const _Sphere<T> &sphere, std::vector<T> &result) const
{
    if(node == nullptr)
    {
        return;
    }
    if(!SphereBoxIntersection(node->boundingBox, sphere))
    {
        return;
    }
    if(node->isValue)
    {
        // DO NOT CHANGE THIS LINE TO "if `node->value` is in `sphere`"
        // that is because a leaf does not necessarily has to be a point (it can be a box, as in `DistributedOctTree`)
        if(SphereBoxIntersection(node->boundingBox, sphere))
        {
            result.push_back(node->value);
        }
    }
    for(int i = 0; i < CHILDREN; i++)
    {
        this->rangeHelper(node->children[i], sphere, result);
    }
}

#define EPSILON 1e-12

template<typename T>
void OctTree<T>::getClosestPointHelper(const T &point, const OctTreeNode *node, T &closestPoint, typename T::coord_type &closestDistance) const
{
    if(node == nullptr)
    {
        return;
    }
    T closestPointInBox = node->boundingBox.closestPoint(point);
    // calculate distance squared
    typename T::coord_type dist = 0;
    for(int i = 0; i < DIM; i++)
    {
        dist += (closestPointInBox[i] - point[i]) * (closestPointInBox[i] - point[i]);
    }
    if(dist >= closestDistance)
    {
        return;
    }
    // there might be a closer point in the subtrees
    if(node->isValue)
    {
        if(node->value == point)
        {
            // don't check that point (otherwise the distance is 0...)
            return;
        }
        closestPoint = node->value;
        closestDistance = dist;
    }
    else
    {
        for(int i = 0; i < CHILDREN; i++)
        {
            this->getClosestPointHelper(point, node->children[i], closestPoint, closestDistance);
        }
    }
}

#endif // _OCTTREE_HPP

