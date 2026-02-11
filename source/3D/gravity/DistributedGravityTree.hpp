#ifndef DISTRIBUTED_GRAVITY_TREE
#define DISTRIBUTED_GRAVITY_TREE

#ifdef RICH_MPI

#include <mpi.h>
#include "3D/gravity/GravityTree.hpp"
#include "3D/elementary/Vector3D.hpp"
#include "3D/tessellation/Tessellation3D.hpp"
#include "mpi/serialize/Serializer.hpp"
#include "mpi/serialize/mpi_commands.hpp"

#define DEFAULT_OWNER_SPLIT 2

struct GravityNodeData
                    #ifdef RICH_MPI
                        : public Serializable
                    #endif // RICH_MPI
{
    BoundingBox<Vector3D> boundingBox;
    gravity_result_t mass;
    Vector3D CM;
    std::array<double, 6> Q;

    #ifdef RICH_MPI
        force_inline size_t dump(Serializer *serializer) const override
        {
            size_t bytes = 0;
            bytes += serializer->insert(this->boundingBox);
            bytes += serializer->insert(this->mass);
            bytes += serializer->insert(this->CM);
            bytes += serializer->insert_array(this->Q);
            return bytes;
        }

        force_inline size_t load(const Serializer *serializer, size_t byteOffset) override
        {
            size_t bytes = 0;
            bytes += serializer->extract(this->boundingBox, byteOffset);
            bytes += serializer->extract(this->mass, bytes + byteOffset);
            bytes += serializer->extract(this->CM, bytes + byteOffset);
            bytes += serializer->extract_array(this->Q, bytes + byteOffset);
            return bytes;
        }
    #endif // RICH_MPI
};

class DistributedGravityTree
{
public:
    explicit DistributedGravityTree(const GravityTree<Vector3D> *gravityTree, const Tessellation3D &tess, double theta, bool quadrupole, int ownerSplit = DEFAULT_OWNER_SPLIT, const MPI_Comm &comm = MPI_COMM_WORLD):
        comm(comm), root(nullptr), thetaSquared(theta * theta), quadrupole(quadrupole), ownerSplit(ownerSplit)
    {
        MPI_Comm_size(comm, &this->size);
        MPI_Comm_rank(comm, &this->rank);
        Vector3D myLL = Vector3D(std::numeric_limits<double>::max(), std::numeric_limits<double>::max(), std::numeric_limits<double>::max());
        Vector3D myUR = Vector3D(std::numeric_limits<double>::lowest(), std::numeric_limits<double>::lowest(), std::numeric_limits<double>::lowest());

        size_t N = tess.GetPointNo();
        const std::vector<Vector3D> &points = tess.getMeshPoints();
        for(size_t i = 0; i < N; i++)
        {
            const Vector3D &point = points[i];
            // set ll and ur if needed, to be the min and max given `point`
            myLL[0] = std::min<double>(myLL[0], point[0]);
            myLL[1] = std::min<double>(myLL[1], point[1]);
            myLL[2] = std::min<double>(myLL[2], point[2]);
            myUR[0] = std::max<double>(myUR[0], point[0]);
            myUR[1] = std::max<double>(myUR[1], point[1]);
            myUR[2] = std::max<double>(myUR[2], point[2]);
        }
        this->myData.boundingBox = BoundingBox<Vector3D>(myLL, myUR);
        const GravityTree<Vector3D>::Node *gravityTreeRoot = gravityTree->getOctTree()->getRoot();
        this->myData.CM = gravityTreeRoot->value.CM;
        this->myData.mass = gravityTreeRoot->value.mass;
        this->myData.Q = gravityTreeRoot->value.Q;
        this->build(gravityTree);
    };
    
    /**
     * returns a tuple: calculated gravity (in the distributed tree), and a set of ranks we have to request from
    */
    std::pair<Vector3D, boost::container::flat_set<int>> gravity(const Vector3D &point) const;

    inline void print() const{this->printHelper(this->root, 0);};

private:
    class Node
    {
    public:
        Node *parent;
        std::vector<Node*> children;
        BoundingBox<Vector3D> boundingBox;
        GravityNodeData value;
        int owner;
        bool isLeaf;
    };

    void printHelper(const Node *node, int indentation) const;

    void updateData(Node *node);

    void buildHelper(Node *node, int minRank, int maxRank);

    void build(const GravityTree<Vector3D> *gravityTree)
    {
        this->deleteTree();
        this->root = new Node();
        this->root->parent = nullptr;
        this->root->boundingBox = gravityTree->getOctTree()->getRoot()->boundingBox;
        this->buildHelper(this->root, 0, this->size);
    }

    void deleteTreeHelper(Node *node)
    {
        if(node == nullptr)
        {
            return;
        }
        for(Node *child : node->children)
        {
            this->deleteTreeHelper(child);
        }
        delete node;
    }

    inline void deleteTree(){this->deleteTreeHelper(this->root); this->root = nullptr;};

private:
    const MPI_Comm &comm;
    Node *root;
    int rank, size;
    GravityNodeData myData;
    bool quadrupole;
    int ownerSplit;
    double thetaSquared;
    mutable std::vector<std::pair<Node*, bool>> stack;
};

std::pair<Vector3D, boost::container::flat_set<int>> DistributedGravityTree::gravity(const Vector3D &point) const
{
    Vector3D gravity;
    boost::container::flat_set<int> ranksToRequest;

    this->stack.push_back({this->root, true});

    while(not this->stack.empty())
    {
        const Node *node = this->stack.back().first;
        bool containsPoint = this->stack.back().second;
        this->stack.pop_back();

        if(node == nullptr)
        {
            continue;
        }

        if(containsPoint or ShouldOpenBox(point, node->boundingBox, node->value.CM, this->thetaSquared))
        {
            if(node->isLeaf)
            {
                // node is leaf. That is, a single rank. Add it to the ranks set
                if(node->owner != this->rank)
                {
                    ranksToRequest.insert(node->owner);
                }
                ranksToRequest.insert(node->owner);
            }
            else
            {
                // node is not a leaf, push all its children
                for(Node *child : node->children)
                {
                    if(child != nullptr)
                    {
                        this->stack.push_back({child, containsPoint and child->boundingBox.contains(point)});
                    }
                }
            }
        }
        else
        {
            // do not open. Add contribution only
            if(not node->isLeaf or node->owner != this->rank)
            {
                // otherwise counted twice
                gravity += CalculateLeafGravityContribution(node->value, point, this->quadrupole);
            }
        }
    }
    return std::make_pair(gravity, ranksToRequest);
}

void DistributedGravityTree::updateData(DistributedGravityTree::Node *node)
{
    if(node == nullptr)
    {
        return;
    }

    GravityNodeData &value = node->value;
    std::array<double, 6> &Q = value.Q;

    value.mass = 0;
    value.CM = Vector3D();
    Vector3D newLL = Vector3D(std::numeric_limits<double>::max(), std::numeric_limits<double>::max(), std::numeric_limits<double>::max());
    Vector3D newUR = Vector3D(std::numeric_limits<double>::lowest(), std::numeric_limits<double>::lowest(), std::numeric_limits<double>::lowest());
    for(const Node *child : node->children)
    {
        if(child != nullptr)
        {
            const GravityNodeData &childValue = child->value;
            value.CM += (childValue.CM) * (childValue.mass);
            value.mass += childValue.mass;
            newLL[0] = std::min<double>(newLL[0], child->boundingBox.getLL()[0]);
            newLL[1] = std::min<double>(newLL[1], child->boundingBox.getLL()[1]);
            newLL[2] = std::min<double>(newLL[2], child->boundingBox.getLL()[2]);
            newUR[0] = std::max<double>(newUR[0], child->boundingBox.getUR()[0]);
            newUR[1] = std::max<double>(newUR[1], child->boundingBox.getUR()[1]);
            newUR[2] = std::max<double>(newUR[2], child->boundingBox.getUR()[2]);
        }
    }
    node->boundingBox = BoundingBox<Vector3D>(newLL, newUR);
    value.CM = value.CM  / value.mass;

    // reset Q
    for(int i = 0; i < 6; i++)
    {
        Q[i] = 0;
    }

    // calculate Q
    if(this->quadrupole)
    {
        for(const Node *child : node->children)
        {
            if(child != nullptr)
            {
                const GravityNodeData &childValue = child->value;
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

void DistributedGravityTree::buildHelper(DistributedGravityTree::Node *node, int minRank, int maxRank)
{
    if(node == nullptr)
    {
        return;
    }    
    
    if(maxRank == minRank + 1)
    {
        // a single owner
        int owner = minRank;
        node->owner = owner;
        GravityNodeData nodeData = MPI_Bcast_serializable(this->myData, owner, this->comm);
        node->boundingBox = nodeData.boundingBox;
        node->value = nodeData;
        node->isLeaf = true;
    }
    else
    {
        // split owners
        int ownersPerChild = (maxRank - minRank) / this->ownerSplit;
        for(int i = 0; i < this->ownerSplit; i++)
        {
            Node *child = new Node();
            int firstOwner = minRank + i * ownersPerChild;
            int lastOwner = (i == this->ownerSplit - 1)? maxRank : firstOwner + ownersPerChild;
            this->buildHelper(child, firstOwner, lastOwner);
            child->parent = node;
            node->children.push_back(child);
        }
        node->owner = UNDEFINED_OWNER;
        node->isLeaf = false;
        this->updateData(node);
    }
}

void DistributedGravityTree::printHelper(const Node *node, int indentation) const
{
    if(node == nullptr)
    {
        return;
    }
    for(int i = 0; i < indentation; i++)
    {
        std::cout << "\t";
    }
    std::cout << "BB: " << node->boundingBox << ", Mass: " << node->value.mass << ", CM: " << node->value.CM;
    if(node->isLeaf)
    {
        std::cout << ", owner: " << node->owner << std::endl;
    }
    else
    {
        std::cout << std::endl;
        for(const Node *child : node->children)
        {
            this->printHelper(child, indentation + 1);
        }
    }
}

#endif // RICH_MPI

#endif // DISTRIBUTED_GRAVITY_TREE
