#ifndef _GRAVITY_TREE_HPP
#define _GRAVITY_TREE_HPP

#include "GravityTypes.h"
#include "ds/OctTree/OctTree.hpp"

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

template<typename T>
class GravityTree
{

public:
    struct _MassedNodeInfo
    {
        using coord_type = typename T::coord_type;

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
        inline friend std::ostream &operator<<(std::ostream &stream, const _MassedNodeInfo &value)
        {
            stream << "[Point: " << value.value << ", Mass: " << value.mass << ", CM: " << value.CM << "]";
            return stream;
        };
        inline _MassedNodeInfo(const T &value, gravity_result_t mass): value(value), CM(value), mass(mass){};
        inline _MassedNodeInfo(const T &value): _MassedNodeInfo(value, 0){};
        inline _MassedNodeInfo(): _MassedNodeInfo(T(), 0){};
    };

    using Node = typename OctTree<_MassedNodeInfo>::OctTreeNode;


private:
    void calculateMassHelper(Node *node);
    inline bool shouldOpenBox(const T &point, const Node *node) const
    {
        if(node == nullptr)
        {
            return false;
        }
        const _BoundingBox<T> boundingBox(node->boundingBox.ll.value, node->boundingBox.ur.value);
        return (!node->isValue) and (boundingBox.contains(point) or (std::abs(GetAngleBoxPoint(point, boundingBox, node->value.CM)) >= this->theta));
    }

    OctTree<_MassedNodeInfo> *octTree;
    double theta;

public:
    GravityTree(const T &ll, const T &ur, double theta): octTree(new OctTree<_MassedNodeInfo>(_MassedNodeInfo(ll), _MassedNodeInfo(ur))), theta(theta){};

    ~GravityTree(){delete this->octTree;};

    inline void calculateMasses(){this->calculateMassHelper(this->octTree->getRoot());};

    inline bool build(const std::vector<MassedPoint<T>> &points)
    {
        int rank; MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        for(const MassedPoint<T> &_point : points)
        {
            // std::cout << "rank " << rank << " inserts point " << _point.point << " with mass " << _point.mass << std::endl;
            if(!this->octTree->insert(_MassedNodeInfo(_point.point, _point.mass)))
            {
                continue; // todo: something else?
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

    inline const OctTree<_MassedNodeInfo> *getOctTree() const{return this->octTree;};

    inline double getTheta() const{return this->theta;};
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
        
        // the mass should be the accumulative mass
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

#include <mpi.h> // todo remove

template<typename T>
T GravityTree<T>::gravityHelper(const T &point, const Node *node) const
{
    int rank; MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    T gravity;
    if(node == nullptr)
    {
        return gravity;
    }
    if(this->shouldOpenBox(point, node))
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
        const T temp = node->value.CM - point;
        gravity_result_t sizeOfForce = 1 / (std::pow(abs(temp), 3));
        gravity = (temp * sizeOfForce) * node->value.mass; // will create a vector in the direction of `temp`, which is in length 1/|temp|^2
    }
    return gravity;
}


#endif // _GRAVITY_TREE_HPP