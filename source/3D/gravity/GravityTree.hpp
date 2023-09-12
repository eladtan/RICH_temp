#ifndef _GRAVITY_TREE_HPP
#define _GRAVITY_TREE_HPP

#include "GravityTypes.h"
#include "ds/OctTree/OctTree.hpp"

namespace
{
    template<typename T>
    double GetAngleBoxPoint(const T &point, const _BoundingBox<T> boundingBox, const T &centerOfMass)
    {
        T urMll = boundingBox.ur - boundingBox.ll;
        typename T::coord_type L = (urMll.x * urMll.x) + (urMll.y * urMll.y) + (urMll.z * urMll.z);
        T dist = point - centerOfMass;
        typename T::coord_type D = (dist.x * dist.x) + (dist.y * dist.y) + (dist.z * dist.z);
        // std::cout << "point = " << point << ", boundingBox = (" << boundingBox.ll << " x " << boundingBox.ur << "), centerOfMass = " << centerOfMass << ", L = " << L << ", D = " << D << ", L/D = " << (L / D) << std::endl;
        /*
        const Vector3D diff = point - centerOfMass;
        typename Vector3D::coord_type _max = std::abs(diff[0]);
        for(int i = 1; i < DIM; i++)
        {
            typename Vector3D::coord_type _val = std::abs(diff[i]);
            if(_val > _max)
            {
                _max = _val;
            }
        }
        return ((_max * _max) / ScalarProd(diff, diff));
        */
       return L / D;
    }
}

template<typename T>
class GravityTree
{

public:
    struct _MassedNodeInfo
    {
        T value;
        T CM; // center of mass
        gravity_result_t mass;

        typename T::coord_type operator[](size_t idx) const{return this->value[idx];};
        typename T::coord_type &operator[](size_t idx){return this->value[idx];};
        inline _MassedNodeInfo operator+(const _MassedNodeInfo &other) const{return _MassedNodeInfo(this->value + other.value);};
        inline _MassedNodeInfo operator-(const _MassedNodeInfo &other) const{return _MassedNodeInfo(this->value - other.value);};
        inline _MassedNodeInfo operator*(typename T::coord_type scalar) const{return _MassedNodeInfo(this->value * scalar);};
        inline _MassedNodeInfo operator/(typename T::coord_type scalar) const{return this->operator*(1 / scalar);};
        inline bool operator==(const _MassedNodeInfo &other) const{return this->value == other.value;};
        inline bool operator!=(const _MassedNodeInfo &other) const{return !this->operator==(other);};
        inline friend std::ostream &operator<<(std::ostream &stream, const _MassedNodeInfo &value){return operator<<(stream, value.value);};
        inline _MassedNodeInfo(const T &value, gravity_result_t mass): value(value), CM(value), mass(mass){};
        inline _MassedNodeInfo(const T &value): _MassedNodeInfo(value, 0){};
        inline _MassedNodeInfo(): _MassedNodeInfo(T(), 0){};
    };

    using Node = typename OctTree<_MassedNodeInfo>::OctTreeNode;

    static inline bool ShouldOpenBox(const T &point, const Node *node, double theta)
    {
        if(node == nullptr)
        {
            return false;
        }
        const _BoundingBox<T> boundingBox(node->boundingBox.ll.value, node->boundingBox.ur.value);
        return (!node->isValue) and (boundingBox.contains(point) or (std::abs(GetAngleBoxPoint(point, boundingBox, node->value.CM) - theta) >= EPSILON));
    }

private:
    void calculateMassHelper(Node *node);

    OctTree<_MassedNodeInfo> *octTree;
    double theta;

public:
    GravityTree(const T &ll, const T &ur, double theta): octTree(new OctTree<_MassedNodeInfo>(_MassedNodeInfo(ll), _MassedNodeInfo(ur))){};

    ~GravityTree(){delete this->octTree;};

    inline void calculateMasses(){this->calculateMassHelper(this->octTree->getRoot());};

    inline bool build(const std::vector<MassedPoint<T>> &points)
    {
        for(const MassedPoint<T> &_point : points)
        {
            if(!this->octTree->insert(_MassedNodeInfo(_point.point, _point.mass)))
            {
                return false; // todo: something else?
            }
        }
        this->calculateMasses();
        return true;
    }

    inline bool find(const T &point){return this->octTree->find(point);};

    T gravityHelper(const T &point, const Node *node) const;

    inline T gravity(gravity_result_t mass, const T &point, const direction_t *directions) const
    {
        return this->gravityHelper(point, this->octTree->getNodeByDirections(directions)) * mass;
    }

    const OctTree<_MassedNodeInfo> *getOctTree() const{return this->octTree;};
};

template<typename T>
void GravityTree<T>::calculateMassHelper(Node *node)
{
    if(node == nullptr)
    {
        return;
    }
    if(!node->isValue)
    {
        // mass is the accumulative mass
        node->value.mass = 0;
        node->value.CM = T();
        for(int i = 0; i < CHILDREN; i++)
        {
            Node *child = node->children[i];
            if(child != nullptr)
            {
                this->calculateMassHelper(child);
                node->value.CM += (child->value.CM) * (child->value.mass);
                node->value.mass += child->value.mass;
            }
        }
        node->value.CM = node->value.CM  / node->value.mass;
    }
    else
    {
        node->value.CM = node->value.value;
    }
}

template<typename T>
T GravityTree<T>::gravityHelper(const T &point, const Node *node) const
{
    T gravity;
    if(node == nullptr)
    {
        return gravity;
    }

    if(GravityTree<T>::ShouldOpenBox(point, node, this->theta))
    {
        // open the box
        for(int i = 0; i < CHILDREN; i++)
        {
            const Node *child = node->children[i];
            gravity += this->gravityHelper(point, child);
        }
    }
    else
    {
        // do not open the box
        const T temp(node->value.CM - point);
        gravity_result_t sizeOfForce = 1 / (std::pow(abs(temp), 3));
        gravity = (temp * sizeOfForce) * node->value.mass; // will create a vector in the direction of `temp`, which is in length 1/|temp|^2
    }
    return gravity;
}


#endif // _GRAVITY_TREE_HPP