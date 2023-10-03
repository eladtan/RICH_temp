#ifndef _GRAVITY_TREE_HPP
#define _GRAVITY_TREE_HPP

#include <vectorclass.h>
#include <vector>
#include "GravityTypes.h"
#include "ds/OctTree/OctTree.hpp"

template<typename T>
class GravityTree;

template<typename T, typename BB>
double GetAngleBoxPoint(const T &point, const BB &boundingBox, const T &centerOfMass, typename T::coord_type &distanceToCM)
{
    typename T::coord_type width = boundingBox.getWidthSquared(); // std::max(boundingBox.ur[0] - boundingBox.ll[0], std::max(boundingBox.ur[1] - boundingBox.ll[1], boundingBox.ur[2] - boundingBox.ll[2]));
    Vec4d diff(point[0] - centerOfMass[0], point[1] - centerOfMass[1], point[2] - centerOfMass[2], 0);
    Vec4d squared = diff * diff;
    distanceToCM = squared[0] + squared[1] + squared[2];
    return width / distanceToCM; // todo: avoid division!
}

template<typename T>
T CalculateLeafGravityContribution(const typename GravityTree<T>::MassedValue &nodeValue, const T &point, double distanceToCM, bool quadrupole = false)
{
    T gravity;
    const T &CM = nodeValue.CM;
    const T &temp = point - CM;
    gravity_result_t length = fastabs(temp); // abs(temp);
    if(length < EPSILON)
    {
        // this leaf is the point itself. Gravity should not be calculated
        return gravity;
    }
    gravity_result_t sizeOfForce = 1 / (length * length * length);
    gravity -= (temp * sizeOfForce) * nodeValue.mass; // will create a vector in the direction of `temp`, which is in length 1/|temp|^2
    
    T quadrupoleAddition;
    if((distanceToCM > 0) and (quadrupole))
    {
        typename T::coord_type Qfactor = sizeOfForce / distanceToCM;
        const typename T::coord_type *Q = nodeValue.Q;
        typename T::coord_type dx = point[0] - CM[0];
        typename T::coord_type dy = point[1] - CM[1];
        typename T::coord_type dz = point[2] - CM[2];
        quadrupoleAddition[0] += Qfactor * (dx * Q[0] + dy * Q[1] + dz * Q[2]);
        quadrupoleAddition[1] += Qfactor * (dx * Q[1] + dy * Q[3] + dz * Q[4]);
        quadrupoleAddition[2] += Qfactor * (dx * Q[2] + dy * Q[4] + dz * Q[5]);
        typename T::coord_type mrr = dx * dx * Q[0] + dy * dy * Q[3] + dz * dz * Q[5] + 2 * dx * dy * Q[1] + 2 * dx * dz * Q[2] + 2 * dy * dz * Q[4];
        Qfactor *= -5 * mrr / (2 * distanceToCM);
        quadrupoleAddition[0] += Qfactor * dx;
        quadrupoleAddition[1] += Qfactor * dy;
        quadrupoleAddition[2] += Qfactor * dz;
        // std::cout << "gravity is " << gravity << ", quadrupole adds " << quadrupoleAddition << std::endl;
    }
    gravity += quadrupoleAddition;
    return gravity;
}

template<typename T>
class GravityTree
{
public:
    struct MassedValue
    {
        using coord_type = typename T::coord_type;

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
        inline bool operator==(const MassedValue &other) const{return this->value == other.value;};
        inline bool operator!=(const MassedValue &other) const{return !this->operator==(other);};
        inline friend std::ostream &operator<<(std::ostream &stream, const MassedValue &value)
        {
            stream << "[Point: " << value.value << ", Mass: " << value.mass << ", CM: " << value.CM << ", Q: (" << value.Q[0] << ", " << value.Q[1] << ", " << value.Q[2] << ", " << value.Q[3] << ", " << value.Q[4] << ", " << value.Q[5] << ")]";
            return stream;
        };
        explicit inline MassedValue(const T &value, gravity_result_t mass): value(value), CM(value), mass(mass){};
        explicit inline MassedValue(const T &value): MassedValue(value, 0){};
        explicit inline MassedValue(): MassedValue(T(), 0){};
    };

    using Node = typename OctTree<MassedValue>::OctTreeNode;


private:
    void calculateMassHelper(Node *node);
    inline bool shouldOpenBox(const T &point, const Node *node, double &distanceToCM) const
    {
        if(node == nullptr)
        {
            return false;
        }
        return ((std::abs(GetAngleBoxPoint(point, node->boundingBox, node->value.CM, distanceToCM)) >= this->thetaSquared) or (node->boundingBox.contains(point)));
    }

    OctTree<MassedValue> *octTree;
    mutable std::vector<std::pair<const Node*, bool>> nodes_stack_;
    double theta, thetaSquared;
    bool quadrupole;

public:
    GravityTree(const T &ll, const T &ur, double theta, bool quadrupole = false): 
            octTree(new OctTree<MassedValue>(MassedValue(ll), MassedValue(ur))),
            theta(theta), thetaSquared(theta * theta),
            quadrupole(quadrupole){};

    ~GravityTree(){delete this->octTree;};

    inline void calculateMasses(){this->calculateMassHelper(this->octTree->getRoot());};

    inline bool build(const std::vector<MassedPoint<T>> &points)
    {
        for(const MassedPoint<T> &_point : points)
        {
            // std::cout << "rank " << rank << " inserts point " << _point.point << " with mass " << _point.mass << std::endl;
            if(!this->octTree->insert(MassedValue(_point.point, _point.mass)))
            {
                continue; // todo: something else?
            }
        }
        this->calculateMasses();
        return true;
    }

    inline bool find(const T &point){return this->octTree->find(point);};

    T gravityHelper(const T &point, const Node *node) const;

    inline T gravity(const T &point, const direction_t *directions = nullptr) const
    {
        T gravity;
        const Node *startingNode = this->octTree->getNodeByDirections(directions);
        this->nodes_stack_.push_back({startingNode, startingNode->boundingBox.contains(point)});

        while(!this->nodes_stack_.empty())
        {
            const Node *node = this->nodes_stack_[this->nodes_stack_.size() - 1].first;
            bool containsPoint = this->nodes_stack_[this->nodes_stack_.size() - 1].second;
            this->nodes_stack_.pop_back();

            if(node == nullptr)
            {
                continue;
            }

            double distanceToCM;
            // always push the child that contains the node
            if(!node->isValue and (containsPoint or this->shouldOpenBox(point, node, distanceToCM)))
            {
                int childContains = -1;
                // open the box
                if(containsPoint)
                {
                    childContains = node->getChildNumberContaining(point); // child index that contains that node
                    this->nodes_stack_.push_back({node->children[childContains], true});
                }
                for(int i = 0; i < CHILDREN; i++)
                {
                    if(i == childContains)
                    {
                        continue;
                    }
                    this->nodes_stack_.push_back({node->children[i], false});
                }
            }
            else
            {
                if(node->isValue)
                {
                    distanceToCM = -1;
                }
                // do not open the box
                gravity += CalculateLeafGravityContribution(node->value, point, distanceToCM, this->quadrupole);
            }
        }
        return gravity;
    }

    inline bool getQuadrupole() const{return this->quadrupole;};

    inline const OctTree<MassedValue> *getOctTree() const{return this->octTree;}; // todo: can remove?

    inline double getTheta() const{return this->theta;};
};

template<typename T>
void GravityTree<T>::calculateMassHelper(Node *node)
{
    if(node == nullptr)
    {
        return;
    }
    MassedValue &value = node->value;
    typename T::coord_type *Q = value.Q;
    // reset Q
    for(int i = 0; i < 6; i++)
    {
        Q[i] = 0;
    }

    if(!node->isValue)
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
    else
    {
        value.CM = value.value;
    }
}

#endif // _GRAVITY_TREE_HPP