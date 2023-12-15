#ifndef HILBERT_TREE_3D
#define HILBERT_TREE_3D

#ifdef DEBUG_MODE
    #include <iostream>
#endif // DEBUG_MODE

#include <vector>
#include <boost/container/flat_set.hpp>
#include <mpi.h>
#include "ds/utils/geometry.hpp"
#include "HilbertConvertor3D.hpp"

#define DEFAULT_RANKS_IN_LEAVES 4
#define UNDEFINED_OWNER -1

template<int max_leaf_ranks = DEFAULT_RANKS_IN_LEAVES>
class HilbertTree3D
{    
private:
    using RanksSet = boost::container::flat_set<int>;

    class Node
    {
    public:
        explicit Node(Node *parent): parent(parent){};
        
        _BoundingBox<Vector3D> boundingBox;
        hilbert_index_t d_start, d_end;
        size_t num_points; // number of points in the subtree
        std::vector<Node*> children;
        bool is_leaf;
        int owners[max_leaf_ranks];
        size_t num_owners;
        Node *parent;
    };

    Node *root;
    mutable std::vector<const Node*> nodes_stack;
    MPI_Comm comm;
    int rank, size;

    void buildTreeHelper(Node *currentNode, const typename HilbertConvertor3D::RecursionArguments &current_args, hilbert_index_t &current_d, const HilbertConvertor3D *convertor, const std::vector<hilbert_index_t> &responsibilityRange);
    void buildTree(const HilbertConvertor3D *convertor, const std::vector<hilbert_index_t> &responsibilityRange);

    #ifdef DEBUG_MODE
        void printHelper(const Node *node, int tabs = 0) const;
    #endif // DEBUG_MODE

public:
    HilbertTree3D(const HilbertConvertor3D *convertor, const std::vector<hilbert_index_t> &responsibilityRange, const MPI_Comm &comm = MPI_COMM_WORLD): comm(comm)
    {
        MPI_Comm_rank(this->comm, &this->rank);
        MPI_Comm_size(this->comm, &this->size);
        this->buildTree(convertor, responsibilityRange);
    }

    ~HilbertTree3D();

    template<typename U>
    RanksSet getIntersectingRanks(const _Sphere<U> &sphere) const;

    template<typename U>
    inline RanksSet getIntersectingRanks(const U &point, typename U::coord_type radius) const
    {
        return this->getIntersectingRanks(_Sphere<U>(point, radius));
    }

    template<typename U>
    std::vector<std::pair<typename Vector3D::coord_type, typename Vector3D::coord_type>> getClosestFurthestPointsByRanks(const U &point) const;

    #ifdef DEBUG_MODE
        void print() const{this->printHelper(this->root);};
    #endif // DEBUG_MODE
};

template<int max_ranks_per_leaf>
HilbertTree3D<max_ranks_per_leaf>::~HilbertTree3D()
{
    this->nodes_stack.push_back(this->root);
    while(not this->nodes_stack.empty())
    {
        const Node *node = this->nodes_stack.back();
        this->nodes_stack.pop_back();

        if(node == nullptr)
        {
            continue;
        }

        if(!node->is_leaf)
        {
            for(const Node *child : node->children)
            {
                this->nodes_stack.push_back(child);
            }
        }

        delete node;
    }

    this->root = nullptr;
}

#ifdef DEBUG_MODE
template<int max_ranks_per_leaf>
void HilbertTree3D<max_ranks_per_leaf>::printHelper(const HilbertTree3D<max_ranks_per_leaf>::Node *node, int tabs) const
{
    if(node == nullptr)
    {
        return;
    }
    for(int i = 0; i < tabs; i++) std::cout << "\t";
    std::cout << "LL = " << node->boundingBox.getLL() << ", UR = " << node->boundingBox.getUR() << ", d: " << node->d_start << " - " << node->d_end << " (points: " << node->num_points << "). ";

    if(node->num_owners == 0)
    {
        std::cout << "No explicit owner";
    }
    else
    {
        if(node->num_owners == 1)
        {
            std::cout << "Owner: " << node->owners[0];
        }
        else
        {
            std::cout << "Owners: ";
            for(size_t i = 0; i < node->num_owners - 1; i++)
            {
                std::cout << node->owners[i] << ", ";
            }
            std::cout << node->owners[node->num_owners - 1];
        }
    }

    std::cout << ((node->is_leaf)? " LEAF" : " NOT LEAF");

    std::cout << std::endl;

    for(const Node *child : node->children)
    {
        this->printHelper(child, tabs + 1);
    }
}
#endif // DEUBG_MODE

template<int max_ranks_per_leaf>
void HilbertTree3D<max_ranks_per_leaf>::buildTreeHelper(Node *currentNode, const typename HilbertConvertor3D::RecursionArguments &current_args, hilbert_index_t &current_d, const HilbertConvertor3D *convertor, const std::vector<hilbert_index_t> &responsibilityRange)
{
    using DirectionVector3D = HilbertConvertor3D::DirectionVector3D;
    using RecursionArguments = HilbertConvertor3D::RecursionArguments;
    using direction_t = HilbertConvertor3D::direction_t;

    if(currentNode == nullptr)
    {
        return;
    }

    const DirectionVector3D &startPoint = current_args.startPoint;
    const DirectionVector3D &a = current_args.a;
    const DirectionVector3D &b = current_args.b;
    const DirectionVector3D &c = current_args.c;
    direction_t width = std::abs(a.x + a.y + a.z);
    direction_t height = std::abs(b.x + b.y + b.z);
    direction_t depth = std::abs(c.x + c.y + c.z);

    size_t num_points = width * height * depth;

    currentNode->num_points = num_points;
    
    currentNode->d_start = current_d;
    currentNode->d_end = current_d + num_points;
    
    // set bounding box
    const std::pair<DirectionVector3D, DirectionVector3D> &boundingBox = convertor->getBoundingBox(current_args);
    const DirectionVector3D &ll = boundingBox.first;       
    const DirectionVector3D &ur = boundingBox.second;       
    currentNode->boundingBox = _BoundingBox<Vector3D>(convertor->WidthHeightDepthToXYZ(ll.x, ll.y, ll.z), convertor->WidthHeightDepthToXYZ(ur.x, ur.y, ur.z));

    std::pair<int, int> ranksMatching = {0, this->size - 1};
    for(int index = 0; index < this->size; index++)
    {
        if(currentNode->d_start <= responsibilityRange[index])
        {
            ranksMatching.first = index;
            break;
        }
    }

    for(int index = ranksMatching.first; index < this->size; index++)
    {
        if((currentNode->d_end - 1) <= responsibilityRange[index])
        {
            ranksMatching.second = index;
            break;
        }

    }

    if((ranksMatching.second - ranksMatching.first) < max_ranks_per_leaf)
    {
        // don't have to call recursively
        currentNode->is_leaf = true;
        currentNode->num_owners = ranksMatching.second - ranksMatching.first + 1;
        for(size_t i = 0; i < currentNode->num_owners; i++)
        {
            currentNode->owners[i] = ranksMatching.first + i;
        }
        current_d += num_points; // one big step for d
    }
    else
    {
        currentNode->is_leaf = false;
        currentNode->owners[0] = UNDEFINED_OWNER;
        currentNode->num_owners = 0;

        // should call recursively
        direction_t dax = SIGN(a.x), day = SIGN(a.y), daz = SIGN(a.z);
        direction_t dbx = SIGN(b.x), dby = SIGN(b.y), dbz = SIGN(b.z);
        direction_t dcx = SIGN(c.x), dcy = SIGN(c.y), dcz = SIGN(c.z);

        // for base case:
        bool baseCase = false;
        DirectionVector3D baseCaseUnitDirection;
        direction_t baseCaseLength;
        // check for base cases
        if(height == 1 and depth == 1)
        {
            baseCase = true;
            baseCaseUnitDirection = {dax, day, daz};
            baseCaseLength = width;
        }

        if(width == 1 and depth == 1)
        {
            baseCase = true;
            baseCaseUnitDirection = {dbx, dby, dbz};
            baseCaseLength = height;
        }

        if(width == 1 and height == 1)
        {
            baseCase = true;
            baseCaseUnitDirection = {dcx, dcy, dcz};
            baseCaseLength = depth;
        }

        if(baseCase)
        {
            direction_t x = startPoint.x, y = startPoint.y, z = startPoint.z;
            for(int i = 0; i < baseCaseLength; i++)
            {
                currentNode->children.push_back(new Node(currentNode));
                this->buildTreeHelper(currentNode->children.back(), {{x, y, z}, {dax, day, daz}, {dbx, dby, dbz}, {dcx, dcy, dcz}}, current_d, convertor, responsibilityRange);
                x += baseCaseUnitDirection.x;
                y += baseCaseUnitDirection.y;
                z += baseCaseUnitDirection.z;
            }
        }
        else
        {
            for(const RecursionArguments &nextArgs : convertor->getRecursionArguments(current_args))
            {
                currentNode->children.push_back(new Node(currentNode));
                this->buildTreeHelper(currentNode->children.back(), nextArgs, current_d, convertor, responsibilityRange);
            }                
        }
    }
}

template<int max_ranks_per_leaf>
void HilbertTree3D<max_ranks_per_leaf>::buildTree(const HilbertConvertor3D *convertor, const std::vector<hilbert_index_t> &responsibilityRange)
{
    using DirectionVector3D = HilbertConvertor3D::DirectionVector3D;
    using RecursionArguments = HilbertConvertor3D::RecursionArguments;
    using direction_t = HilbertConvertor3D::direction_t;
    hilbert_index_t d = 0;

    this->root = new Node(nullptr);
    this->buildTreeHelper(this->root, {{0, 0, 0}, {convertor->div.x, 0, 0}, {0, convertor->div.y, 0}, {0, 0, convertor->div.z}}, d, convertor, responsibilityRange);

    if(d != convertor->total_points_num)
    {
        throw UniversalError("HilbertTree3D::buildTree: d (" + std::to_string(d) + " != convertor->total_points_num " + std::to_string(convertor->total_points_num) + ")");
    }
}

template<int max_ranks_per_leaf>
template<typename U>
typename HilbertTree3D<max_ranks_per_leaf>::RanksSet HilbertTree3D<max_ranks_per_leaf>::getIntersectingRanks(const _Sphere<U> &sphere) const
{
    RanksSet result;
    this->nodes_stack.push_back(this->root);

    while(not this->nodes_stack.empty())
    {
        const Node *node = this->nodes_stack.back();
        this->nodes_stack.pop_back();

        if(node == nullptr)
        {
            continue;
        }

        if(not SphereBoxIntersection(node->boundingBox, sphere))
        {
            continue;
        }

        if(node->is_leaf)
        {
            for(size_t i = 0; i < node->num_owners; i++)
            {
                result.insert(node->owners[i]);
            }
        }
        else
        {
            for(const Node *child : node->children)
            {
                this->nodes_stack.push_back(child); // recursively iterate
            }
        }
    }

    if(result.empty())
    {
        UniversalError eo("In HilbertTree3D::getIntersectingRanks: result is empty, (should at least contain the rank itself)");
        eo.addEntry("Sphere", sphere);
        eo.addEntry("Root Bounding Box", this->root->boundingBox);
        throw eo;
    }

    return result;
}

template<int max_ranks_per_leaf>
template<typename U>
std::vector<std::pair<typename Vector3D::coord_type, typename Vector3D::coord_type>> HilbertTree3D<max_ranks_per_leaf>::getClosestFurthestPointsByRanks(const U &point) const
{
    using coord_type = typename Vector3D::coord_type;

    const coord_type &maxVal = std::numeric_limits<coord_type>::max();
    const coord_type &minVal = std::numeric_limits<coord_type>::min();
    
    std::pair<Vector3D, Vector3D> initialPair = std::make_pair<Vector3D, Vector3D>(Vector3D(maxVal, maxVal, maxVal), Vector3D(minVal, minVal, minVal));
    std::vector<std::pair<coord_type, coord_type>> distances(this->size, {maxVal, minVal});

    this->nodes_stack.push_back(this->root);

    Vector3D closestPoint, furthestPoint;
    while(not this->nodes_stack.empty())
    {
        const Node *node = this->nodes_stack.back();
        this->nodes_stack.pop_back();

        if(node == nullptr)
        {
            continue;
        }
        if(!node->is_leaf)
        {
            for(const Node *child : node->children)
            {
                this->nodes_stack.push_back(child);
            }
            continue;
        }
        // node is a value node
        closestPoint = node->boundingBox.closestPoint(point);
        furthestPoint = node->boundingBox.furthestPoint(point);
        coord_type closestDist = 0, furthestDist = 0;
        for(int i = 0; i < DIM; i++)
        {
            closestDist += (closestPoint[i] - point[i]) * (closestPoint[i] - point[i]);
            furthestDist += (furthestPoint[i] - point[i]) * (furthestPoint[i] - point[i]);
        }

        for(size_t i = 0; i < node->num_owners; i++)
        {
            int owner = node->owners[i];
            if(distances[owner].first > closestDist)
            {
                distances[owner].first = closestDist;
            }
            if(distances[owner].second < furthestDist)
            {
                distances[owner].second = furthestDist;
            }
        }
    }
    return distances;
}

#endif // HILBERT_TREE_3D