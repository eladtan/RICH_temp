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
    using Mass = typename GravityTree<U>::MassedValue; // adding mass to struct `U`
    /* define `Node` as a node of octtree, when its values are ranked (that is, they have ranks), and massed (that is, each one has `mass` and `CM`), where the points theirself are of type `U` */
    using OctNode = typename OctTree<Ranking<Mass<T>>>::OctTreeNode;

public:
    DistributedGravityTree(const GravityTree<T> *gravityTree, bool quadrupole = false, const MPI_Comm &comm = MPI_COMM_WORLD): 
        quadrupole(quadrupole)
    {
        MPI_Comm_size(comm, &this->size);
        MPI_Comm_rank(comm, &this->rank);
        this->distributedTree = new DistributedOctTree<Mass<T>>(gravityTree->getOctTree(), true /* copy data */, comm);
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

    #ifdef DEBUG_MODE
    inline void print() const{this->distributedTree->print();};
    #endif // DEBUG_MODE

private:    
    void getLocationListHelper(const OctNode *node, const T &point, std::vector<GravityTreeLocation> &locations, T &unopenedGravity) const;

    void fixGravityValuesHelper(OctNode *node);

    inline void fixGravityValues(){this->fixGravityValuesHelper(this->distributedTree->getOctTree()->getRoot());};

    int rank, size;
    DistributedOctTree<Mass<T>> *distributedTree;
    double theta, thetaSquared;
    bool quadrupole;
};

template<typename T>
void DistributedGravityTree<T>::fixGravityValuesHelper(OctNode *node)
{
    if(node == nullptr)
    {
        return;
    }

    Mass<T> &value = node->value.value;
    typename T::coord_type *Q = value.Q;

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

        // reset Q
        for(int i = 0; i < 6; i++)
        {
            Q[i] = 0;
        }

        // calculate Q
        for(int i = 0; i < CHILDREN; i++)
        {
            const OctNode *child = node->children[i];
            if(child != nullptr)
            {
                const Mass<T> &childValue = child->value.value;
                const gravity_result_t &childMass = childValue.mass;
                double qx = childValue.CM[0] - value.CM[0];
                double qy = childValue.CM[1] - value.CM[1];
                double qz = childValue.CM[2] - value.CM[2];
                double qr2 = (qx * qx) + (qy * qy) + (qz * qz);
                Q[0] += childValue.Q[0] + childMass * (3 * (qx * qx) - qr2);
                Q[1] += childValue.Q[1] + (3 * childMass) * (qx * qy);
                Q[2] += childValue.Q[2] + (3 * childMass) * (qx * qz);
                Q[3] += childValue.Q[3] + childMass * (3 * (qy * qy) - qr2);
                Q[4] += childValue.Q[4] + (3 * childMass) * (qz * qy);
            }
        }
		Q[5] = -Q[0] - Q[3];
    }
}

template<typename T>
void DistributedGravityTree<T>::getLocationListHelper(const OctNode *node, const T &point, std::vector<GravityTreeLocation> &locations, T &unopenedGravity) const
{
    if(node == nullptr)
    {
        return;
    }

    if(ShouldOpenBox(point, node->boundingBox, node->value.value.CM, this->thetaSquared))
    {
        // open the box            
        if(node->value.owner != UNDEFINED_OWNER)
        {
            // one owner, to just add it to locations (a location contains rank and direction within this rank)
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
        unopenedGravity += CalculateLeafGravityContribution(node->value.value, point, this->quadrupole);
    }
}

#endif // _DISTRIBUTED_GRAVITY_TREE_HPP