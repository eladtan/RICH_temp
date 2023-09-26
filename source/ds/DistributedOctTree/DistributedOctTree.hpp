#ifndef _DISTRIBUTED_OCTTREE_HPP
#define _DISTRIBUTED_OCTTREE_HPP

#ifdef RICH_MPI

#include <vector>
#include <assert.h>
#include <utility>
#include <array>
#include <bitset>
#include <boost/container/flat_set.hpp>
#include <mpi.h>
#include "ds/OctTree/OctTree.hpp"

#ifdef DEBUG_MODE
#include <iostream>
#endif // DEBUG_MODE

#define UNDEFINED_OWNER -1

template<typename T>
class DistributedOctTree
{

public:
    DistributedOctTree(const MPI_Comm &comm, const OctTree<T> *tree);
    inline DistributedOctTree(const OctTree<T> *tree): DistributedOctTree(MPI_COMM_WORLD, tree){};
    ~DistributedOctTree(){delete this->octTree;};

    void print() const{this->octTree->print();};
    boost::container::flat_set<int> getIntersectingRanks(const _Sphere<T> &sphere) const;
    inline boost::container::flat_set<int> getIntersectingRanks(const T &center, const typename T::coord_type radius) const{return this->getIntersectingRanks(_Sphere(center, radius));};
    int getOwnerRank(const T &point) const;
    int getDepth() const{return this->octTree->getDepth();};
    inline int getClosestRank(const T &point) const
    {
        int minRank = this->rank;
        typename T::coord_type minDistance = std::numeric_limits<typename T::coord_type>::max();
        this->getClosestRankHelper(this->octTree->getRoot(), point, minRank, minDistance);
        return minRank;
    }

    inline std::vector<std::pair<typename T::coord_type, typename T::coord_type>> closestFurthestPoints(const T &point) const
    {
        const typename T::coord_type &maxVal = std::numeric_limits<typename T::coord_type>::max();
        const typename T::coord_type &minVal = std::numeric_limits<typename T::coord_type>::min();
        std::pair<T, T> initialPair = std::make_pair<T, T>(T(maxVal, maxVal, maxVal), T(minVal, minVal, minVal));
        std::vector<std::pair<typename T::coord_type, typename T::coord_type>> distances(size, {maxVal, minVal});
        this->closestFurthestPointsHelper(this->octTree->getRoot(), point, distances);
        return distances;
    }

    #ifdef DEBUG_MODE
    inline bool validate() const{if(this->octTree != nullptr) return this->validateHelper(this->octTree->getRoot()); return true;};
    #endif // DEBUG_MODE

private:
    class _Wrapper
    {
        friend class DistributedOctTree;
    public:
        using coord_type = typename T::coord_type;

        T value;
        int owner; // rank of the owner

        _Wrapper(const T &value, int owner): value(value), owner(owner){};
        _Wrapper(): _Wrapper(T(), UNDEFINED_OWNER){};
        _Wrapper(const _Wrapper &other): _Wrapper(other.value, other.owner){};
        _Wrapper &operator=(const _Wrapper &other){this->value = other.value; this->owner = other.owner; return (*this);};
        inline typename T::coord_type operator[](size_t idx) const{return this->value[idx];};
        inline typename T::coord_type &operator[](size_t idx){return this->value[idx];};
        inline _Wrapper operator+(const _Wrapper &other) const{int owner = (this->owner == other.owner)? this->owner : UNDEFINED_OWNER; return _Wrapper(this->value + other.value, owner);};
        inline _Wrapper operator-(const _Wrapper &other) const{int owner = (this->owner == other.owner)? this->owner : UNDEFINED_OWNER; return _Wrapper(this->value - other.value, owner);};
        inline _Wrapper operator*(typename T::coord_type constant) const{return _Wrapper(this->value * constant, owner);};
        inline _Wrapper operator/(typename T::coord_type constant) const{return this->operator*(1/constant);};
        inline bool operator==(const _Wrapper &other) const{return this->value == other.value;};
        inline bool operator!=(const _Wrapper &other) const{return this->value != other.value;};
        friend std::ostream &operator<<(std::ostream &stream, const _Wrapper &wrapper)
        {
            return stream << wrapper.value << " [owner: " << wrapper.owner << "]";
        }

    };

    OctTree<_Wrapper> *octTree = nullptr;
    MPI_Comm comm;
    int rank, size;

    void buildTreeHelper(typename OctTree<_Wrapper>::OctTreeNode *newNode, const typename OctTree<T>::OctTreeNode *node);
    void buildTree(const OctTree<T> *tree);
    void getClosestRankHelper(const typename OctTree<_Wrapper>::OctTreeNode *node, const T &point, int &closestRank, typename T::coord_type &closestRankDistance) const;
    void closestFurthestPointsHelper(const typename OctTree<_Wrapper>::OctTreeNode *node, const T &point, std::vector<std::pair<typename T::coord_type, typename T::coord_type>> &distances) const;

    #ifdef DEBUG_MODE
    bool validateHelper(const typename OctTree<_Wrapper>::OctTreeNode *node) const;
    #endif // DEBUG_MODE
};

template<typename T>
DistributedOctTree<T>::DistributedOctTree(const MPI_Comm &comm, const OctTree<T> *tree): comm(comm)
{
    MPI_Comm_rank(this->comm, &this->rank);
    MPI_Comm_size(this->comm, &this->size);
    this->buildTree(tree);
}

template<typename T>
void DistributedOctTree<T>::buildTreeHelper(typename OctTree<_Wrapper>::OctTreeNode *newNode, const typename OctTree<T>::OctTreeNode *node)
{
    assert(newNode != nullptr);
        
    unsigned char valueToSend = 0; // assumes `CHILDREN` is 8. this variable contains 1 in the `i`th bit iff child `i` exists
    if(node != nullptr)
    {
        for(int i = 0; i < CHILDREN; i++)
        {
            bool bit = (node->children[i] != nullptr || (node->isValue and newNode->getChildNumberContaining(_Wrapper(node->value, UNDEFINED_OWNER)) == i));
            valueToSend |= (bit << i);
        }
    }
    std::vector<unsigned char> childBuff(this->size);
    MPI_Allgather(&valueToSend, 1, MPI_UNSIGNED_CHAR, &childBuff[0], 1, MPI_BYTE, this->comm);

    for(int i = 0; i < CHILDREN; i++)
    {
        bool recursiveBuild = false;
        int containingValue = UNDEFINED_OWNER; // who has the `i`th child
        for(int _rank = 0; _rank < this->size; _rank++)
        {
            if(((childBuff[_rank] >> i) & 0x1) == 1)
            {
                if(containingValue == UNDEFINED_OWNER)
                {
                    containingValue = _rank;
                }
                else
                {
                    // more than one child has a point in this route of the tree, so we continue to split recursively
                    recursiveBuild = true;
                    break;
                }
            }
        }
        // if somebody holds the value
        if(containingValue != UNDEFINED_OWNER)
        {
            // someone holds the `i`th child
            newNode->createChild(i); // creates the child in my own tree
            if(recursiveBuild)
            {
                // there are several holders, call recursive build (until we reach one holder)
                // determine what's the next node in my own tree to continue the recursive build with
                // this node might be null, if I don't have any nodes this depth in the tree
                const typename OctTree<T>::OctTreeNode *nextNode = nullptr;
                if(node != nullptr)
                {
                    if(node->isValue)
                    {
                        nextNode = newNode->children[i]->boundingBox.contains(_Wrapper(node->value, UNDEFINED_OWNER))? node : nullptr;
                    }
                    else
                    {
                        nextNode = node->children[i];
                    }
                }
                else
                {
                    nextNode = nullptr;
                }
                // continue recursively
                this->buildTreeHelper(newNode->children[i], nextNode);
                newNode->children[i]->value.owner = UNDEFINED_OWNER; // several owners
                newNode->children[i]->boundingBox.ll.owner = newNode->children[i]->boundingBox.ur.owner = UNDEFINED_OWNER;
            }
            else
            {
                // there is only one holder, set its owner field
                newNode->children[i]->value.owner = containingValue;
                newNode->children[i]->boundingBox.ll.owner = newNode->children[i]->boundingBox.ur.owner = containingValue;
                newNode->children[i]->isValue = true;
            }
        }
        else
        {
            newNode->children[i] = nullptr;
        }
    }
}

template<typename T>
int DistributedOctTree<T>::getOwnerRank(const T &point) const
{
    _Wrapper pointWrapping(point, UNDEFINED_OWNER);
    const typename OctTree<_Wrapper>::OctTreeNode *current = this->octTree->getRoot();

    while(current != nullptr and (!current->isValue))
    {
        current = current->getChildContaining(pointWrapping);
    }
    if(current == nullptr)
    {
        std::cerr << "Error! (point is " << point << ")" << std::endl;
    }
    return current->value.owner;
}

#ifdef DEBUG_MODE
template<typename T>
bool DistributedOctTree<T>::validateHelper(const typename OctTree<_Wrapper>::OctTreeNode *node) const
{
    if(node == nullptr)
    {
        return true;
    }
    bool hasChildren = false;
    for(int i = 0; i < CHILDREN; i++)
    {
        if(node->children[i] != nullptr)
        {
            hasChildren = true;
            break;
        }
    }
    if(hasChildren)
    {
        assert(node->isValue);
        assert(node->boundingBox.ll.owner != UNDEFINED_OWNER);
        assert(node->boundingBox.ur.owner != UNDEFINED_OWNER);
        assert(node->boundingBox.ll.owner == node->boundingBox.ur.owner);
    }
    for(int i = 0; i < CHILDREN; i++)
    {
        assert(this->validateHelper(node->children[i]));
    }
    return true;
}
#endif // DEBUG_MODE

template<typename T>
void DistributedOctTree<T>::buildTree(const OctTree<T> *tree)
{
    assert(this->octTree == nullptr);
    if(tree == nullptr or tree->getRoot() == nullptr)
    {
        return;
    }
    this->octTree = new OctTree<_Wrapper>(_Wrapper(tree->getRoot()->boundingBox.ll, UNDEFINED_OWNER), _Wrapper(tree->getRoot()->boundingBox.ur, UNDEFINED_OWNER));
    this->buildTreeHelper(this->octTree->getRoot(), tree->getRoot());
}

template<typename T>
boost::container::flat_set<int> DistributedOctTree<T>::getIntersectingRanks(const _Sphere<T> &sphere) const
{
    boost::container::flat_set<int> ranks;
    for(const _Wrapper &point : this->octTree->range(_Sphere<_Wrapper>(_Wrapper(sphere.center, UNDEFINED_OWNER), sphere.radius)))
    {
        ranks.insert(point.owner);
    }
    return ranks;
}

template<typename T>
void DistributedOctTree<T>::getClosestRankHelper(const typename OctTree<_Wrapper>::OctTreeNode *node, const T &point, int &closestRank, typename T::coord_type &closestRankDistance) const
{
    if(node == nullptr)
    {
        return;
    }
    _Wrapper closestPoint = node->boundingBox.closestPoint(_Wrapper(point, UNDEFINED_OWNER));
    typename T::coord_type distance = 0;
    for(int i = 0; i < DIM; i++)
    {
        distance += (closestPoint[i] - point[i]) * (closestPoint[i] - point[i]);
    }
    if(distance >= closestRankDistance)
    {
        return;
    }
    if(node->isValue)
    {
        if(node->value.owner == this->rank)
        {
            return;
        }
        closestRank = node->value.owner;
        closestRankDistance = distance;
    }
    else
    {
        for(int i = 0; i < CHILDREN; i++)
        {
            this->getClosestRankHelper(node->children[i], point, closestRank, closestRankDistance);
        }
    }
}

template<typename T>
void DistributedOctTree<T>::closestFurthestPointsHelper(const typename OctTree<_Wrapper>::OctTreeNode *node, const T &point, std::vector<std::pair<typename T::coord_type, typename T::coord_type>> &distances) const
{
    if(node == nullptr)
    {
        return;
    }
    if(!node->isValue)
    {
        for(int i = 0; i < CHILDREN; i++)
        {
            this->closestFurthestPointsHelper(node->children[i], point, distances);
        }
        return;
    }
    // node is a value node
    int owner = node->value.owner;
    T closestPoint = node->boundingBox.closestPoint(_Wrapper(point, UNDEFINED_OWNER)).value;
    T furthestPoint = node->boundingBox.furthestPoint(_Wrapper(point, UNDEFINED_OWNER)).value;
    typename T::coord_type closestDist = 0, furthestDist = 0;
    for(int i = 0; i < DIM; i++)
    {
        closestDist += (closestPoint[i] - point[i]) * (closestPoint[i] - point[i]);
        furthestDist += (furthestPoint[i] - point[i]) * (furthestPoint[i] - point[i]);
    }
    if(distances[owner].first > closestDist)
    {
        distances[owner].first = closestDist;
    }
    if(distances[owner].second < furthestDist)
    {
        distances[owner].second = furthestDist;
    }
}
#endif // RICH_MPI

#endif // _DISTRIBUTED_OCTTREE_HPP

