#ifndef _DISTRIBUTED_GRAVITY_TREE_HPP
#define _DISTRIBUTED_GRAVITY_TREE_HPP

#include <vector>
#include "ds/DistributedOctTree/DistributedOctTree.hpp"
#include "GravityTree.hpp"

struct GravityTreeLocation
{
    direction_t directions[MAX_DIRECTIONS_SIZE];
    int rank;
};

template<typename T>
class DistributedGravityTree
{
private:
    template<typename U>
    using Ranking = typename DistributedOctTree<U>::RankedValue; // adding ranks to struct `U`
    template<typename U>
    using Mass = typename GravityTree<U>::_MassedNodeInfo; // adding mass to struct `U`
    /* define `Node` as a node of octtree, when its values are ranked (that is, they have ranks), and massed (that is, each one has `mass` and `CM`), where the points theirself are of type `U` */
    
    using OctNode = typename OctTree<Ranking<Mass<T>>>::OctTreeNode;

public:
    DistributedGravityTree(const GravityTree<T> *gravityTree, const MPI_Comm &comm = MPI_COMM_WORLD): comm(comm)
    {
        MPI_Comm_size(this->comm, &this->size);
        MPI_Comm_rank(this->comm, &this->rank);
        this->distributedTree = new DistributedOctTree<Mass<T>>(gravityTree->getOctTree(), true /* copy data */, this->comm);
        this->fixGravityValues();
        this->theta = gravityTree->getTheta();
        this->thetaSquared = this->theta * this->theta;
    }

    ~DistributedGravityTree()
    {
        delete this->distributedTree;
    }

    /**
     * Returns a pair - the gravity force of the un-opened cells, and a list of locations the should be opened
    */
    inline std::pair<T, std::vector<GravityTreeLocation>> getLocationList(const T &point) const
    {
        T initialGravity;
        std::vector<GravityTreeLocation> locations;
        this->getLocationListHelper(this->distributedTree->getOctTree()->getRoot(), point, locations, initialGravity);
        return std::make_pair<T, std::vector<GravityTreeLocation>>(std::move(initialGravity), std::move(locations));
    };

private:    
    void getLocationListHelper(const OctNode *node, const T &point, std::vector<GravityTreeLocation> &locations, T &unopenedGravity) const;
    
    inline bool shouldOpenBox(const T &point, const OctNode *node) const
    {
        if(node == nullptr)
        {
            return false;
        }
        const Mass<T> massedPoint = node->value.value;
        const _BoundingBox<T> boundingBox(node->boundingBox.getLL().value.value, node->boundingBox.getUR().value.value);
        return /*(!node->isValue) and */(boundingBox.contains(point) or (std::abs(GetAngleBoxPoint(point, boundingBox, massedPoint.CM)) >= this->thetaSquared));
    }

    void fixGravityValuesHelper(OctNode *node);

    inline void fixGravityValues(){this->fixGravityValuesHelper(this->distributedTree->getOctTree()->getRoot());};

    MPI_Comm comm;
    int rank, size;
    DistributedOctTree<Mass<T>> *distributedTree;
    double theta, thetaSquared;
};

template<typename T>
void DistributedGravityTree<T>::fixGravityValuesHelper(OctNode *node)
{
    if(node == nullptr)
    {
        return;
    }

    Mass<T> &value = node->value.value;

    if(!node->isValue)
    {
        // the mass should be the accumulative mass
        value.mass = 0;
        value.CM = T();
        for(int i = 0; i < CHILDREN; i++)
        {
            OctNode *child = node->children[i];
            if(child != nullptr)
            {
                Mass<T> &childValue = child->value.value;
                this->fixGravityValuesHelper(child);
                value.CM += (childValue.CM) * (childValue.mass);
                value.mass += childValue.mass;
            }
        }
        value.CM = value.CM  / value.mass;
    }
}

template<typename T>
void DistributedGravityTree<T>::getLocationListHelper(const OctNode *node, const T &point, std::vector<GravityTreeLocation> &locations, T &unopenedGravity) const
{
    if(node == nullptr)
    {
        return;
    }

    if(node->value.owner == this->rank)
    {
        return; // self gravity will be dealt with later
    }
    if(this->shouldOpenBox(point, node))
    {
        // open the box            
        if(node->value.owner != UNDEFINED_OWNER)
        {
            // one owner, to just add it to locations
            locations.push_back({});
            GravityTreeLocation &currLoc = locations[locations.size() - 1];
            currLoc.rank = node->value.owner;
            std::memcpy(currLoc.directions, node->value.directions, sizeof(direction_t) * MAX_DIRECTIONS_SIZE);
        }
        else
        {
            // several ranks holding values in this node's subtree, so we should go even deeper
            for(int i = 0; i < CHILDREN; i++)
            {
                this->getLocationListHelper(node->children[i], point, locations, unopenedGravity);
            }
        }
    }
    else
    {
        // if(rank == 0) std::cout << "rank " << rank << " DOES NOT open the box " << node->value << std::endl;
        // do not open the box
        gravity_result_t mass = node->value.value.mass;
        const T temp = node->value.value.CM - point;
        gravity_result_t length = fastabs(temp);
        gravity_result_t sizeOfForce = 1 / (length * length * length);
        unopenedGravity += (temp * sizeOfForce) * mass; // will create a vector in the direction of `temp`, which is in length 1/|temp|^2
    }
}

#endif // _DISTRIBUTED_GRAVITY_TREE_HPP