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
#define MAX_DIRECTIONS_SIZE (MAX_DEPTH + 1)

template<typename T>
class DistributedOctTree
{
public:
    struct RankedValue
    {
        // friend class DistributedOctTree;
        using coord_type = typename T::coord_type;

        T value;
        int owner; // rank of the owner
        direction_t directions[MAX_DIRECTIONS_SIZE]; // where it is in the owner's tree (directions)

        RankedValue(const T &value, int owner): value(value), owner(owner){this->directions[0] = PATH_END_DIRECTION;};
        RankedValue(): RankedValue(T(), UNDEFINED_OWNER){};
        RankedValue(const RankedValue &other): RankedValue(other.value, other.owner){};
        RankedValue &operator=(const RankedValue &other){this->value = other.value; this->owner = other.owner; return (*this);};
        inline typename T::coord_type operator[](size_t idx) const{return this->value[idx];};
        inline typename T::coord_type &operator[](size_t idx){return this->value[idx];};
        inline RankedValue operator+(const RankedValue &other) const{int owner = (this->owner == other.owner)? this->owner : UNDEFINED_OWNER; return RankedValue(this->value + other.value, owner);};
        inline RankedValue operator-(const RankedValue &other) const{int owner = (this->owner == other.owner)? this->owner : UNDEFINED_OWNER; return RankedValue(this->value - other.value, owner);};
        inline RankedValue operator*(double constant) const{return RankedValue(this->value * constant, owner);};
        inline RankedValue operator/(double constant) const{return this->operator*(1/constant);};
        inline bool operator==(const RankedValue &other) const{return this->value == other.value;};
        inline bool operator!=(const RankedValue &other) const{return this->value != other.value;};
        friend std::ostream &operator<<(std::ostream &stream, const RankedValue &wrapper)
        {
            return stream << wrapper.value << " [owner: " << wrapper.owner << "]";
        }

        int getRank() const{return this->owner;};
    };

    using DistributedOctTreeNode = typename OctTree<RankedValue>::OctTreeNode;

    DistributedOctTree(const OctTree<T> *tree, bool detailedNodeInfo = false, const MPI_Comm &comm = MPI_COMM_WORLD);
    ~DistributedOctTree(){delete this->octTree;};

    void print() const{this->octTree->print();};
    boost::container::flat_set<int> getIntersectingRanks(const _Sphere<T> &sphere) const;
    inline boost::container::flat_set<int> getIntersectingRanks(const T &center, const typename T::coord_type radius) const{return this->getIntersectingRanks(_Sphere(center, radius));};
    int getDepth() const{return this->octTree->getDepth();};
    OctTree<RankedValue> *getOctTree(){return this->octTree;};

    int getClosestRank(const T &point) const;

    std::vector<T> getRankValues(int _rank) const;

    std::vector<_BoundingBox<T>> getRankBoundingBoxes(int _rank) const;

    inline std::vector<T> getMyValues() const{return this->getRankValues(this->rank);};

    inline std::vector<_BoundingBox<T>> getMyBoundingBoxes() const{return this->getRankBoundingBoxes(this->rank);};

    std::vector<std::pair<typename T::coord_type, typename T::coord_type>> getClosestFurthestPointsByRanks(const T &point) const;

    template<typename U>
    inline int getOwner(const U &value) const
    {
        return this->octTree->findParent(value).owner;
    }

    #ifdef DEBUG_MODE
    inline bool validate() const{if(this->octTree != nullptr) return this->validateHelper(this->octTree->getRoot()); return true;};
    #endif // DEBUG_MODE

private:
    OctTree<RankedValue> *octTree = nullptr;
    MPI_Comm comm;
    int rank, size;
    bool detailedNodeInfo;

    void buildTreeHelper(DistributedOctTreeNode *newNode, const typename OctTree<T>::OctTreeNode *node, std::vector<direction_t> &directionsInMyTree);
    void buildTree(const OctTree<T> *tree);

    #ifdef DEBUG_MODE
    bool validateHelper(const DistributedOctTreeNode *node) const;
    #endif // DEBUG_MODE
};

template<typename T>
DistributedOctTree<T>::DistributedOctTree(const OctTree<T> *tree, bool detailedNodeInfo, const MPI_Comm &comm): comm(comm), detailedNodeInfo(detailedNodeInfo)
{
    MPI_Comm_rank(this->comm, &this->rank);
    MPI_Comm_size(this->comm, &this->size);
    this->buildTree(tree);
}

template<typename T>
void DistributedOctTree<T>::buildTreeHelper(DistributedOctTreeNode *newNode, const typename OctTree<T>::OctTreeNode *node, std::vector<direction_t> &directionsInMyTree)
{
    assert(newNode != nullptr);
    unsigned char valueToSend = 0; // assumes `CHILDREN` is 8. this variable contains 1 in the `i`th bit iff child `i` exists
    if(node != nullptr)
    {
        for(int i = 0; i < CHILDREN; i++)
        {
            bool bit = (node->children[i] != nullptr || (node->isValue and newNode->getChildNumberContaining(RankedValue(node->value, UNDEFINED_OWNER)) == i));
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

                bool advancedNode = false; // if went down to a child of current node
                if(node != nullptr)
                {
                    if(node->isValue)
                    {
                        nextNode = newNode->children[i]->boundingBox.contains(RankedValue(node->value, UNDEFINED_OWNER))? node : nullptr;
                    }
                    else
                    {
                        advancedNode = true;
                        directionsInMyTree.push_back(i);
                        nextNode = node->children[i];
                    }
                }
                else
                {
                    nextNode = nullptr;
                }
                // continue recursively
                this->buildTreeHelper(newNode->children[i], nextNode, directionsInMyTree);
                if(advancedNode)
                {
                    directionsInMyTree.pop_back();
                }
                newNode->children[i]->value.owner = UNDEFINED_OWNER; // several owners
                newNode->children[i]->isValue = false;
            }
            else
            {
                // there is only one holder, set its owner field
                if(this->detailedNodeInfo)
                {
                    if(rank == containingValue)
                    {
                        const typename OctTree<T>::OctTreeNode *childNode;
                        if(node->isValue)
                        {
                            childNode = node;
                        }
                        else
                        {
                            childNode = node->children[i];
                            directionsInMyTree.push_back(i);
                        }
        
                        std::memcpy(&newNode->children[i]->value.value, &childNode->value, sizeof(T));
                        std::memcpy(newNode->children[i]->value.directions, directionsInMyTree.data(), sizeof(direction_t) * directionsInMyTree.size());
                        // newNode->children[i]->value.directions[directionsInMyTree.size()] = PATH_END_DIRECTION;
                        newNode->children[i]->value.directions[directionsInMyTree.size()] = PATH_END_DIRECTION;

                        if(!node->isValue)
                        {
                            directionsInMyTree.pop_back();
                        }
                    }
                    MPI_Bcast(&newNode->children[i]->value.value, sizeof(T), MPI_BYTE, containingValue, this->comm);
                    MPI_Bcast(newNode->children[i]->value.directions, sizeof(direction_t) * MAX_DIRECTIONS_SIZE, MPI_BYTE, containingValue, this->comm);
                }
                newNode->children[i]->value.owner = containingValue;
                newNode->children[i]->isValue = true;
            }
        }
        else
        {
            newNode->children[i] = nullptr;
        }
    }
}

#ifdef DEBUG_MODE
template<typename T>
bool DistributedOctTree<T>::validateHelper(const DistributedOctTreeNode *node) const
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
    std::vector<direction_t> directions;
    directions.reserve(MAX_DIRECTIONS_SIZE);
    // tree->print();
    this->octTree = new OctTree<RankedValue>(RankedValue(tree->getRoot()->boundingBox.getLL(), UNDEFINED_OWNER), RankedValue(tree->getRoot()->boundingBox.getUR(), UNDEFINED_OWNER));
    this->buildTreeHelper(this->octTree->getRoot(), tree->getRoot(), directions);
    // this->octTree->print();
}

template<typename T>
boost::container::flat_set<int> DistributedOctTree<T>::getIntersectingRanks(const _Sphere<T> &sphere) const
{
    boost::container::flat_set<int> ranks;
    for(const RankedValue &point : this->octTree->range(_Sphere<RankedValue>(RankedValue(sphere.center, UNDEFINED_OWNER), sphere.radius)))
    {
        ranks.insert(point.owner);
    }
    return ranks;
}

template<typename T>
int DistributedOctTree<T>::getClosestRank(const T &point) const
{
    int closestRank = this->rank;
    typename T::coord_type closestRankDistance = std::numeric_limits<typename T::coord_type>::max();

    std::vector<const DistributedOctTreeNode*> nodes;
    nodes.reserve(this->getDepth() * CHILDREN);

    nodes.push_back(this->octTree->getRoot());

    while(!nodes.empty())
    {
        const DistributedOctTreeNode *node = nodes[nodes.size() - 1];
        nodes.pop_back();

        if(node == nullptr)
        {
            continue;
        }
        RankedValue closestPoint = node->boundingBox.closestPoint(RankedValue(point, UNDEFINED_OWNER));
        typename T::coord_type distance = 0;
        for(int i = 0; i < DIM; i++)
        {
            distance += (closestPoint[i] - point[i]) * (closestPoint[i] - point[i]);
        }
        if(distance >= closestRankDistance)
        {
            continue;
        }
        if(node->isValue)
        {
            if(node->value.owner == this->rank)
            {
                continue; // do not count
            }
            closestRank = node->value.owner;
            closestRankDistance = distance;
        }
        else
        {
            for(int i = 0; i < CHILDREN; i++)
            {
                nodes.push_back(node->children[i]);
            }
        }
    }
    return closestRank;
}

template<typename T>
std::vector<T> DistributedOctTree<T>::getRankValues(int _rank) const
{
    std::vector<T> values;
    std::vector<const DistributedOctTreeNode*> nodes;
    nodes.reserve(this->getDepth() * CHILDREN);

    nodes.push_back(this->octTree->getRoot());

    while(!nodes.empty())
    {
        const DistributedOctTreeNode *node = nodes[nodes.size() - 1];
        nodes.pop_back();

        if(node == nullptr)
        {
            continue;
        }

        if(node->value.owner == UNDEFINED_OWNER)
        {
            for(int i = 0; i < CHILDREN; i++)
            {
                nodes.push_back(node->children[i]);
            }
        }
        else
        {
            if(node->value.owner == _rank)
            {
                values.emplace_back(node->value.value);
            }
        }
    }
    return values;
}

template<typename T>
std::vector<_BoundingBox<T>> DistributedOctTree<T>::getRankBoundingBoxes(int _rank) const
{
    std::vector<_BoundingBox<T>> boxes;
    std::vector<const DistributedOctTreeNode*> nodes;
    nodes.reserve(this->getDepth() * CHILDREN);

    nodes.push_back(this->octTree->getRoot());

    while(!nodes.empty())
    {
        const DistributedOctTreeNode *node = nodes[nodes.size() - 1];
        nodes.pop_back();

        if(node == nullptr)
        {
            continue;
        }

        if(node->value.owner == UNDEFINED_OWNER)
        {
            for(int i = 0; i < CHILDREN; i++)
            {
                nodes.push_back(node->children[i]);
            }
        }
        else
        {
            if(node->value.owner == _rank)
            {
                boxes.emplace_back(_BoundingBox<T>(node->boundingBox.getLL().value, node->boundingBox.getUR().value));
            }
        }
    }
    return boxes;
}

template<typename T>
std::vector<std::pair<typename T::coord_type, typename T::coord_type>> DistributedOctTree<T>::getClosestFurthestPointsByRanks(const T &point) const
{
    const typename T::coord_type &maxVal = std::numeric_limits<typename T::coord_type>::max();
    const typename T::coord_type &minVal = std::numeric_limits<typename T::coord_type>::min();
    std::pair<T, T> initialPair = std::make_pair<T, T>(T(maxVal, maxVal, maxVal), T(minVal, minVal, minVal));
    std::vector<std::pair<typename T::coord_type, typename T::coord_type>> distances(size, {maxVal, minVal});

    std::vector<const DistributedOctTreeNode*> nodes;
    nodes.reserve(this->getDepth() * CHILDREN);
    nodes.push_back(this->octTree->getRoot());

    T closestPoint, furthestPoint;
    while(!nodes.empty())
    {
        const DistributedOctTreeNode *node = nodes[nodes.size() - 1];
        nodes.pop_back();

        if(node == nullptr)
        {
            continue;
        }
        if(!node->isValue)
        {
            for(int i = 0; i < CHILDREN; i++)
            {
                nodes.push_back(node->children[i]);
            }
            continue;
        }
        // node is a value node
        int owner = node->value.owner;
        closestPoint = node->boundingBox.closestPoint(RankedValue(point, UNDEFINED_OWNER)).value;
        furthestPoint = node->boundingBox.furthestPoint(RankedValue(point, UNDEFINED_OWNER)).value;
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
    return distances;
}

#endif // RICH_MPI

#endif // _DISTRIBUTED_OCTTREE_HPP