#ifndef TIMING_TREE_HPP
#define TIMING_TREE_HPP

#ifdef USE_VCL_VECTORIZATION
    #include <vectorclass.h>
#endif // USE_VCL_VECTORIZATION
#include <vector>
#include <boost/container/flat_map.hpp>
#include "../timesteps.h"
#include "3D/tesselation/Tessellation3D.hpp"
#include "newtonian/three_dimensional/computational_cell.hpp"
#include "ds/OctTree/OctTree.hpp"
#include "ds/utils/raw_type.h"

#define MAX_ID_OF_CELL 1e15

template<typename T>
class TimingTree;

namespace
{
    template<typename T>
    inline double Calculate_v_Signal(const typename TimingTree<T>::NodeData &point1, const typename TimingTree<T>::NodeData &point2, const T &r, const typename T::coord_type &rAbs)
    {
        return (point1.cs + point2.cs) - (ScalarProd(point1.v - point2.v, r) / rAbs);
    }
}

template<typename T>
dt_t CalculateTau(const typename TimingTree<T>::NodeData &point1, const typename TimingTree<T>::NodeData &point2)
{
    T r = point1.value - point2.value;
    typename T::coord_type rAbs = abs(r);
    return rAbs / (Calculate_v_Signal(point1, point2, r, rAbs));
}

template<typename T, typename BB_T = T>
inline bool ShouldOpenNode(const typename TimingTree<T>::NodeData &currentCell, const BoundingBox<BB_T> &currentCellBoundingBox, const dt_t &t_current, double other_c_max_plus_v_max, const BoundingBox<BB_T> &otherCellBoundingBox)
{
    // if(currentCell.c_plus_v < EPSILON)
    // {
    //     UniversalError eo("c_plus_v is small");
    //     eo.addEntry("c_plus_v", currentCell.c_plus_v);
    //     eo.addEntry("c", currentCell.cs);
    //     eo.addEntry("v", currentCell.v_abs);
    //     eo.addEntry("currentCell", currentCell);
    //     throw eo;
    // }
    T closestPoint = otherCellBoundingBox.closestPointToOther(currentCellBoundingBox);
    typename T::coord_type distanceToCellSquared = otherCellBoundingBox.distanceSquared(closestPoint);
    if(t_current == MAX_TIME)
    {
        return true;
    }
    double possibleDistanceOfSpeed = t_current * (currentCell.c_plus_v + other_c_max_plus_v_max);
    return (distanceToCellSquared < ((1 + EPSILON) * (possibleDistanceOfSpeed * possibleDistanceOfSpeed)));
}

template<typename T>
class TimingTree
{
public:
    class NodeData
        #ifdef RICH_MPI
            : public Serializable
        #endif // RICH_MPI
    {
    public:
        using coord_type = typename T::coord_type;
        using Raw_type = T;

        T value;
        T v; // velocity
        size_t id_of_cell; // if a leaf (a value), the id of the cell
        double v_abs; // |v|
        double cs; // speed of sound
        double c_plus_v; // c + |v|
        double cell_width; // R
        double v_tag_abs; // |v - w| where `w` is the face velocity  
        double c_max; // max sound speed in subtree
        double v_max; // max velocity in subtree
        double c_max_plus_v_max; // c_max + |v_max|
        dt_t min_time_in_subtree;

        typename T::coord_type operator[](size_t idx) const{return this->value[idx];};
        typename T::coord_type &operator[](size_t idx){return this->value[idx];};
        inline NodeData operator+(const NodeData &other) const{return NodeData(this->value + other.value);};
        inline NodeData operator-(const NodeData &other) const{return NodeData(this->value - other.value);};
        inline NodeData operator*(typename T::coord_type scalar) const{return NodeData(this->value * scalar);};
        inline NodeData operator/(typename T::coord_type scalar) const{return this->operator*(1 / scalar);};
        inline bool operator==(const T &other) const{return this->value == other;};
        inline bool operator==(const NodeData &other) const{return this->operator==(other.value);};
        inline bool operator!=(const NodeData &other) const{return !this->operator==(other);};
        inline friend std::ostream &operator<<(std::ostream &stream, const NodeData &value)
        {
            stream << "[Point: " << value.value << ", Max c: " << value.c_max << ", Max |v|: " << value.v_max << "]";
            return stream;
        };

        explicit inline NodeData(const T &value = T(), size_t cellIndex = std::numeric_limits<size_t>::max(), const ComputationalCell3D *cell = nullptr, double cellWidth = 0, const T &faceVelocity = T()):
            value(value), min_time_in_subtree(MAX_TIME)
        {
            this->v = (cell == nullptr ? T() : cell->velocity);
            this->id_of_cell = cellIndex;
            this->v_abs = abs(this->v);
            this->cs = (cell == nullptr ? 0 : cell->cs);
            // if(cell != nullptr and this->cs < EPSILON)
            // {
            //     UniversalError eo("NodeData: cs is 0");
            //     eo.addEntry("value", value);
            //     eo.addEntry("cellIndex", cellIndex);
            //     throw eo;
            // }
            this->cell_width = cellWidth;
            this->v_tag_abs = abs(this->v - faceVelocity);
            this->c_max = this->cs;
            this->v_max = this->v_abs;
            this->c_plus_v = this->cs + this->v_abs;
            this->c_max_plus_v_max = this->c_max + this->v_max;
        };

    size_t getChunkSize() const
    {
        return 16; // 3 for two T's and 10 for 10 double variables
    }

    std::vector<double> serialize() const
    {
        std::vector<double> serialized;
        std::vector<double> serialized_point = this->value.serialize();
        serialized.insert(serialized.end(), serialized_point.cbegin(), serialized_point.cend());
        serialized_point = this->v.serialize();
        serialized.insert(serialized.end(), serialized_point.cbegin(), serialized_point.cend());
        serialized.push_back((this->id_of_cell >= MAX_ID_OF_CELL)? MAX_ID_OF_CELL : this->id_of_cell);
        serialized.push_back(this->v_abs);
        serialized.push_back(this->cs);
        serialized.push_back(this->c_plus_v);
        serialized.push_back(this->cell_width);
        serialized.push_back(this->v_tag_abs);
        serialized.push_back(this->c_max);
        serialized.push_back(this->v_max);
        serialized.push_back(this->c_max_plus_v_max);
        serialized.push_back(this->min_time_in_subtree);
        return serialized;
    }

    void unserialize(const std::vector<double> &data)
    {
        size_t TChunkSize = this->value.getChunkSize();
        std::vector<double> serData(data.cbegin(), data.cbegin() + TChunkSize);
        this->value.unserialize(serData);
        serData = std::vector<double>(data.cbegin() + TChunkSize, data.cbegin() + 2 * TChunkSize);
        this->v.unserialize(serData);
        this->id_of_cell = data[2 * TChunkSize];
        this->v_abs = data[2 * TChunkSize + 1];
        this->cs = data[2 * TChunkSize + 2];
        this->c_plus_v = data[2 * TChunkSize + 3];
        this->cell_width = data[2 * TChunkSize + 4];
        this->v_tag_abs = data[2 * TChunkSize + 5];
        this->c_max = data[2 * TChunkSize + 6];
        this->v_max = data[2 * TChunkSize + 7];
        this->c_max_plus_v_max = data[2 * TChunkSize + 8];
        this->min_time_in_subtree = data[2 * TChunkSize + 9];
    }

    };

    using Node = typename OctTree<NodeData>::OctTreeNode;

private:
    void calculateDataHelper(Node *node);
    void updateMinTimesInSubtreesRecursively(Node *node);

    NodeData ll, ur;
    OctTree<NodeData> *octTree;
    boost::container::flat_map<size_t, T> cellsToPoints;
    mutable std::vector<std::pair<const Node*, bool>> stack;

public:
    TimingTree(const T &ll_, const T &ur_): ll(NodeData(ll_)), ur(NodeData(ur_)), octTree(nullptr)
    {};

    ~TimingTree(){delete this->octTree;};

    inline void calculateData(){this->calculateDataHelper(this->octTree->getRoot());};

    template<typename U>
    bool build(const Tessellation3D &tess, const std::vector<ComputationalCell3D> &cells, const std::vector<U> &faceVelocities);

    inline bool find(const T &point){return this->octTree->find(point);};

    template<typename BB_T = T>
    dt_t time(const NodeData &boxValue, const BoundingBox<BB_T> &boxBB) const;

    dt_t time(size_t cellID);

    void addExternalValues(const std::vector<NodeData> &data);

    void calculateSelfTimes(const std::vector<size_t> &cellsIdx)
    {
        for(const size_t &idx : cellsIdx)
        {
            this->time(idx); // updates `min_time_in_subtree`
        }
        // update `min_time_in_subtree` recursively
        this->updateMinTimesInSubtreesRecursively(this->octTree->getRoot());
    }

    inline const OctTree<NodeData> *getOctTree() const{return this->octTree;};
};

template<typename T>
template<typename U>
bool TimingTree<T>::build(const Tessellation3D &tess, const std::vector<ComputationalCell3D> &cells, const std::vector<U> &faceVelocities)
{
    size_t N = tess.GetPointNo();
    if(N != cells.size())
    {
        UniversalError eo("The number of points and cells must be the same");
        eo.addEntry("Number of points", N);
        eo.addEntry("Number of cells", cells.size());
        throw eo;
    }
    if(faceVelocities.size() != cells.size())
    {
        UniversalError eo("The number of faceVelocities and cells must be the same");
        eo.addEntry("Number of faceVelocities", faceVelocities.size());
        eo.addEntry("Number of cells", cells.size());
        throw eo;
    }

    delete this->octTree;
    this->octTree = new OctTree<NodeData>(this->ll, this->ur);
    this->cellsToPoints.clear();

    for(size_t i = 0; i < N; i++)
    {
        T point = tess.GetMeshPoint(i);
        NodeData value(point, i, &cells[i], tess.GetWidth(i), faceVelocities[i]);
        if(!this->octTree->insert(value))
        {
            UniversalError eo("Could not add a point to the timing tree");
            eo.addEntry("Cell index", i);
            eo.addEntry("Point", point);
            eo.addEntry("Width", tess.GetWidth(i));
            throw eo;
        }
        size_t cellID = i;
        this->cellsToPoints[cellID] = point;
    }

    // traverse the tree and initialize the time as MAX_TIME

    std::vector<Node*> nodes({this->octTree->getRoot()});
    while(not nodes.empty())
    {
        Node *node = nodes.back();
        nodes.pop_back();
        if(node == nullptr)
        {
            continue;
        }
        node->value.min_time_in_subtree = MAX_TIME;
        if(node->isLeaf)
        {
            continue;
        }
        for(int i = 0; i < CHILDREN; i++)
        {
            nodes.push_back(node->children[i]);
        }
    }
    
    this->calculateData();
    return true;
}

template<typename T>
void TimingTree<T>::addExternalValues(const std::vector<NodeData> &data)
{
    for(const NodeData &pointData : data)
    {
        if(!this->octTree->insert(pointData))
        {
            UniversalError eo("TimeTree::addExternalValues: Could not add a point to the timetree");
            eo.addEntry("point", pointData.value);
            eo.addEntry("cell width", pointData.cell_width);
            throw eo;
        }
    }
    // this->calculateData();
}

template<typename T>
void TimingTree<T>::calculateDataHelper(Node *node)
{
    if(node == nullptr)
    {
        return;
    }

    NodeData &value = node->value;

    if(node->isLeaf)
    {
        value.c_max = value.cs;
        value.v_max = value.v_abs;

        if(value.c_max < 0)
        {
            throw UniversalError("TimingTree::calculateDataHelper: c_max is 0 for a leaf");
        }
    }
    else
    {
        for(int i = 0; i < CHILDREN; i++)
        {
            Node *child = node->children[i];
            if(child != nullptr)
            {
                this->calculateDataHelper(child);
                value.c_max = std::max(value.c_max, child->value.c_max);
                value.v_max = std::max(value.v_max, child->value.v_max);
            }
        }
    }

    value.c_max_plus_v_max = value.c_max + value.v_max;
}

template<typename T>
dt_t TimingTree<T>::time(size_t cellID)
{
    if(this->cellsToPoints.find(cellID) == this->cellsToPoints.cend())
    {
        UniversalError eo("TimingTree time calculation: cell index was not in build");
        eo.addEntry("Cell index", cellID);
        throw eo;
    }

    const T &point = this->cellsToPoints.at(cellID);
    Node *pointNode = this->octTree->findNode(point);
    NodeData &pointValue = pointNode->value;
    #ifdef DEBUG_MODE
        if(pointValue.value != point)
        {
            UniversalError eo("TimingTree::time, PointValue != point");
            eo.addEntry("point value", pointValue.value);
            eo.addEntry("point", point);
            throw eo;
        }
    #endif // DEBUG_MODE
    pointValue.min_time_in_subtree = this->time(pointValue, pointNode->boundingBox);
    return pointValue.min_time_in_subtree;
}

template<typename T>
template<typename BB_T>
dt_t TimingTree<T>::time(const NodeData &boxValue, const BoundingBox<BB_T> &boxBB) const
{
    stack.push_back({this->octTree->getRoot(), true});
    dt_t time = boxValue.cell_width / (boxValue.cs + boxValue.v_tag_abs);
    const T &point = boxValue.value;

    while(!stack.empty())
    {
        const Node *node = stack.back().first;
        bool containsPoint = stack.back().second;
        stack.pop_back();

        if(node == nullptr)
        {
            continue;
        }

        // always push the child that contains the node
        if(!node->isLeaf and (containsPoint or ShouldOpenNode<T>(boxValue, boxBB, time, node->value.c_max_plus_v_max, node->boundingBox)))
        {
            int childContains = -1;
            // open the box
            if(containsPoint)
            {
                childContains = node->getChildNumberContaining(point); // child index that contains that node
                stack.push_back({node->children[childContains], true});
            }
            for(int i = 0; i < CHILDREN; i++)
            {
                if(i == childContains)
                {
                    continue;
                }
                stack.push_back({node->children[i], false});
            }
        }
        else
        {
            if(boxValue.value == node->value.value)
            {
                continue;
            }
            // do not open the box
            dt_t tau = CalculateTau<T>(boxValue, node->value);
            time = std::min<dt_t>(time, tau);
        }
    }
    return time;
}

template<typename T>
void TimingTree<T>::updateMinTimesInSubtreesRecursively(Node *node)
{
    if(node == nullptr)
    {
        return;
    }

    NodeData &value = node->value;
    if(node->isLeaf)
    {
        return;
    }

    for(int i = 0; i < CHILDREN; i++)
    {
        this->updateMinTimesInSubtreesRecursively(node->children[i]);
    }

    value.min_time_in_subtree = MAX_TIME;
    for(int i = 0; i < CHILDREN; i++)
    {
        if(node->children[i] != nullptr)
        {
            value.min_time_in_subtree = std::min(value.min_time_in_subtree, node->children[i]->value.min_time_in_subtree);
        }
    }
}

#endif // TIMING_TREE_HPP