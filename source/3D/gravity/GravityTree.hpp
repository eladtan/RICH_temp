#ifndef _GRAVITY_TREE_HPP
#define _GRAVITY_TREE_HPP

#ifdef USE_VCL_VECTORIZATION
    #include <vectorclass.h>
#endif // USE_VCL_VECTORIZATION
#include <vector>
#include "GravityTypes.h"
#include "ds/OctTree/OctTree.hpp"
#include "misc/serializable.hpp"
#include "misc/universal_error.hpp"

template<typename T>
class GravityTree;

template<typename T, typename BB_T>
bool ShouldOpenBox(const T &point, const _BoundingBox<BB_T> &boundingBox, const T &centerOfMass, double thetaSquared)
{
    typename T::coord_type width2 = boundingBox.getWidthSquared();
    #ifdef USE_VCL_VECTORIZATION
        Vec4d diff(point[0] - centerOfMass[0], point[1] - centerOfMass[1], point[2] - centerOfMass[2], 0);
        Vec4d squared = diff * diff;
        typename T::coord_type distanceToCM2 = squared[0] + squared[1] + squared[2];
    #else // USE_VCL_VECTORIZATION
        typename T::coord_type distanceToCM2 = ((point[0] - centerOfMass[0]) * (point[0] - centerOfMass[0])) + 
                                            ((point[1] - centerOfMass[1]) * (point[1] - centerOfMass[1])) +
                                            ((point[2] - centerOfMass[2]) * (point[2] - centerOfMass[2]));
    #endif // USE_VCL_VECTORIZATION
    return width2 >= (distanceToCM2 * thetaSquared);
}

template<typename T>
T CalculateLeafGravityContribution(const typename GravityTree<T>::MassedValue &nodeValue, const T &point, bool quadrupole = false)
{
    T gravity;
    const T &CM = nodeValue.CM;
    const T &temp = point - CM;
    gravity_result_t length = abs(temp); // abs(temp);
    gravity_result_t length2 = length * length;
    if(length < EPSILON)
    {
        // this leaf is the point itself. Gravity should not be calculated
        return gravity;
    }
    gravity_result_t sizeOfForce = 1 / (length2 * length);
    gravity -= temp * (sizeOfForce * nodeValue.mass); // will create a vector in the direction of `temp`, which is in length 1/|temp|^2
    
    T quadrupoleAddition;
    if(quadrupole)
    {
        typename T::coord_type Qfactor = sizeOfForce / length2;
        const typename T::coord_type *Q = nodeValue.Q;
        typename T::coord_type dx = temp[0];
        typename T::coord_type dy = temp[1];
        typename T::coord_type dz = temp[2];
        quadrupoleAddition[0] += Qfactor * (dx * Q[0] + dy * Q[1] + dz * Q[2]);
        quadrupoleAddition[1] += Qfactor * (dx * Q[1] + dy * Q[3] + dz * Q[4]);
        quadrupoleAddition[2] += Qfactor * (dx * Q[2] + dy * Q[4] + dz * Q[5]);
        typename T::coord_type mrr = dx * dx * Q[0] + dy * dy * Q[3] + dz * dz * Q[5] + 2 * dx * dy * Q[1] + 2 * dx * dz * Q[2] + 2 * dy * dz * Q[4];
        Qfactor *= -5 * mrr / (2 * length2);
        quadrupoleAddition[0] += Qfactor * dx;
        quadrupoleAddition[1] += Qfactor * dy;
        quadrupoleAddition[2] += Qfactor * dz;
    }
    gravity += quadrupoleAddition;
    return gravity;
}

template<typename T>
class GravityTree
{
public:
    struct MassedValue : public Serializable
    {
        using coord_type = typename T::coord_type;
        using Raw_type = T;

        T value;
        T CM; // center of mass
        gravity_result_t mass;
        coord_type Q[6]; // for quadropole calculations

        typename T::coord_type operator[](size_t idx) const{return this->value[idx];};
        typename T::coord_type &operator[](size_t idx){return this->value[idx];};
        inline MassedValue operator+(const MassedValue &other) const{return MassedValue(this->value + other.value);};
        inline MassedValue operator-(const MassedValue &other) const{return MassedValue(this->value - other.value);};
        inline MassedValue operator*(typename T::coord_type scalar) const{return MassedValue(this->value * scalar);};
        inline MassedValue operator/(typename T::coord_type scalar) const{return this->operator*(1 / scalar);};
        inline bool operator==(const T &other) const{return this->value == other;};
        inline bool operator==(const MassedValue &other) const{return this->operator==(other.value);};
        inline bool operator!=(const MassedValue &other) const{return !this->operator==(other);};
        inline friend std::ostream &operator<<(std::ostream &stream, const MassedValue &value)
        {
            stream << "[Point: " << value.value << ", Mass: " << value.mass << ", CM: " << value.CM << ", Q: (" << value.Q[0] << ", " << value.Q[1] << ", " << value.Q[2] << ", " << value.Q[3] << ", " << value.Q[4] << ", " << value.Q[5] << ")]";
            return stream;
        };
        explicit inline MassedValue(const T &value, gravity_result_t mass): value(value), CM(value), mass(mass)
        {
            // reset Q
            for(int i = 0; i < 6; i++)
            {
                this->Q[i] = 0;
            }
        };
        explicit inline MassedValue(const T &value): MassedValue(value, 0){};
        explicit inline MassedValue(): MassedValue(T(), 0){};

        inline size_t getChunkSize() const override
        {
            return 3 + 3 + 1 + 6; // 3, 3 for value and CM, 1 for mass, 6 for Q
        }

        inline std::vector<double> serialize() const override
        {
            std::vector<double> valueSerialized = this->value.serialize();
            std::vector<double> CMSerialized = this->CM.serialize();
            valueSerialized.insert(valueSerialized.end(), CMSerialized.begin(), CMSerialized.end());
            valueSerialized.push_back(this->mass);
            valueSerialized.insert(valueSerialized.end(), this->Q, this->Q + 6);
            return valueSerialized;
        }

        inline void unserialize(const std::vector<double> &data) override
        {
            this->value.unserialize(std::vector<double>(data.cbegin(), data.cbegin() + 3));
            this->CM.unserialize(std::vector<double>(data.cbegin() + 3, data.cbegin() + 6));
            this->mass = data[6];
            for(int i = 0; i < 6; i++)
            {
                this->Q[i] = data[7 + i];
            }
        }
    };

    using Node = typename OctTree<MassedValue>::OctTreeNode;


private:
    void calculateMassHelper(Node *node);

    OctTree<MassedValue> *octTree;
    double theta, thetaSquared;
    bool quadrupole;
    mutable std::vector<std::pair<const Node*, bool>> stack;

public:
    GravityTree(const T &ll, const T &ur, double theta, bool quadrupole = false): 
            octTree(new OctTree<MassedValue>(MassedValue(ll), MassedValue(ur))),
            theta(theta), thetaSquared(theta * theta),
            quadrupole(quadrupole){};

    ~GravityTree(){delete this->octTree;};

    inline void calculateMasses(){this->calculateMassHelper(this->octTree->getRoot());};

    bool build(const std::vector<MassedPoint<T>> &points);

    inline bool find(const T &point){return this->octTree->find(point);};

    T gravity(const T &point, const direction_t *directions = nullptr) const;

    void addExternalValues(const std::vector<MassedValue> &values);

    template<typename U>
    MassedValue findMatchingMassedValue(const _BoundingBox<U> &boundingBox) const;

    std::vector<std::pair<octnode_id_t, MassedValue>> getOpenNodesData(const T &point) const;

    inline bool getQuadrupole() const{return this->quadrupole;};

    inline const OctTree<MassedValue> *getOctTree() const{return this->octTree;};

    inline double getTheta() const{return this->theta;};

    #ifdef DEBUG_MODE
        void print() const{this->octTree->print();};
    #endif // DEBUG_MODE
};

template<typename T>
bool GravityTree<T>::build(const std::vector<MassedPoint<T>> &points)
{
    for(const MassedPoint<T> &_point : points)
    {
        MassedValue value(_point.point, _point.mass);
        if(!this->octTree->insert(value))
        {
            UniversalError eo("Could not add a point to the gravity tree");
            eo.addEntry("Point", value);
            throw eo;
        }
    }
    this->calculateMasses();
    return true;

}

template<typename T>
void GravityTree<T>::addExternalValues(const std::vector<MassedValue> &values)
{
    for(const MassedValue &value : values)
    {
        MassedValue correctedValue;
        correctedValue.value = value.CM;
        correctedValue.CM = value.CM;
        correctedValue.mass = value.mass;
        for(int i = 0; i < 6; i++)
        {
            correctedValue.Q[i] = value.Q[i];
        }
        if(!this->octTree->insert(correctedValue))
        {
            UniversalError eo("Could not add a point to the gravity tree");
            eo.addEntry("Point", correctedValue);
            throw eo;
        }
    }
    this->calculateMasses();
}

template<typename T>
template<typename U>
typename GravityTree<T>::MassedValue GravityTree<T>::findMatchingMassedValue(const _BoundingBox<U> &boundingBox) const
{
    const Node *node = this->octTree->findNodeContainingBoundingBox(boundingBox);
    if(node == nullptr)
    {
        UniversalError eo("GravityTree::findMatchingMassedValue: node containing bounding box could not be found");
        eo.addEntry("Bounding Box", boundingBox);
        throw eo;
    }
    return node->value;
}

template<typename T>
std::vector<std::pair<octnode_id_t, typename GravityTree<T>::MassedValue>> GravityTree<T>::getOpenNodesData(const T &point) const
{
    std::vector<std::pair<octnode_id_t, MassedValue>> values;
    const Node *startingNode = this->octTree->getRoot();
    stack.push_back({startingNode, startingNode->boundingBox.contains(point)});

    while(!stack.empty())
    {
        const Node *node = stack[stack.size() - 1].first;
        bool containsPoint = stack[stack.size() - 1].second;
        stack.pop_back();

        if(node == nullptr)
        {
            continue;
        }
        
        // always push the child that contains the node
        if(!node->isLeaf and (containsPoint or ShouldOpenBox(point, node->boundingBox, node->value.CM, this->thetaSquared)))
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
            values.push_back({node->id, node->value});
        }
    }
    return values;
}

template<typename T>
void GravityTree<T>::calculateMassHelper(Node *node)
{
    if(node == nullptr)
    {
        return;
    }

    MassedValue &value = node->value;
    typename T::coord_type *Q = value.Q;

    if(!node->isLeaf)
    {
        // internal node
        // the mass should be the accumulative mass. Calculate also the center of mass
        value.mass = 0;
        value.CM = T();
        for(int i = 0; i < CHILDREN; i++)
        {
            Node *child = node->children[i];
            if(child != nullptr)
            {
                const MassedValue &childValue = child->value;
                this->calculateMassHelper(child);
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
        if(this->quadrupole)
        {
            for(int i = 0; i < CHILDREN; i++)
            {
                const Node *child = node->children[i];
                if(child != nullptr)
                {
                    const MassedValue &childValue = child->value;
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
}

template<typename T>
T GravityTree<T>::gravity(const T &point, const direction_t *directions) const
{
    T gravity;
    const Node *startingNode = this->octTree->getNodeByDirections(directions);
    stack.push_back({startingNode, startingNode->boundingBox.contains(point)});

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
        if(!node->isLeaf and (containsPoint or ShouldOpenBox(point, node->boundingBox, node->value.CM, this->thetaSquared)))
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
            // do not open the box
            // activate quadrupole only for non-value nodes
            gravity += CalculateLeafGravityContribution(node->value, point, this->quadrupole);
        }
    }
    return gravity;
}

#endif // _GRAVITY_TREE_HPP