#ifndef _GRAVITY_TREE_HPP
#define _GRAVITY_TREE_HPP

#include <vectorclass.h>
#include <vector>
#include "GravityTypes.h"
#include "ds/OctTree/OctTree.hpp"

template<typename T, typename BB>
double GetAngleBoxPoint(const T &point, const BB &boundingBox, const T &centerOfMass)
{
    typename T::coord_type width = boundingBox.getWidthSquared(); // std::max(boundingBox.ur[0] - boundingBox.ll[0], std::max(boundingBox.ur[1] - boundingBox.ll[1], boundingBox.ur[2] - boundingBox.ll[2]));
    Vec4d diff(point[0] - centerOfMass[0], point[1] - centerOfMass[1], point[2] - centerOfMass[2], 0);
    Vec4d squared = diff * diff;
    typename T::coord_type distance = squared[0] + squared[1] + squared[2];
    return width / distance; // todo: avoid division!
    /*
    T dist = point - centerOfMass;
    Vec8d vec1(boundingBox.ur[0], boundingBox.ur[1], boundingBox.ur[2], dist.x, dist.y, dist.z, 0, 0);
    Vec8d vec2(boundingBox.ll[0], boundingBox.ll[1], boundingBox.ll[2], 0, 0, 0, 0, 0);
    Vec8d diff = vec1 - vec2;
    Vec8d squared = diff * diff;
    typename T::coord_type L = squared[0] + squared[1] + squared[2];
    typename T::coord_type D = squared[3] + squared[4] + squared[5];
    return L / D;
    */

    /*
    T urMll = boundingBox.ur - boundingBox.ll;
    typename T::coord_type L = (urMll.x * urMll.x) + (urMll.y * urMll.y) + (urMll.z * urMll.z);
    T dist = point - centerOfMass;
    typename T::coord_type D = (dist.x * dist.x) + (dist.y * dist.y) + (dist.z * dist.z);
    return L / D;
    */
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
        explicit inline _MassedNodeInfo(const T &value, gravity_result_t mass): value(value), CM(value), mass(mass){};
        explicit inline _MassedNodeInfo(const T &value): _MassedNodeInfo(value, 0){};
        explicit inline _MassedNodeInfo(): _MassedNodeInfo(T(), 0){};
    };

    using Node = typename OctTree<_MassedNodeInfo>::OctTreeNode;


private:
    void calculateMassHelper(Node *node);
    /*inline*/ bool shouldOpenBox(const T &point, const Node *node) const
    {
        if(node == nullptr)
        {
            return false;
        }
        return /* boundingBox.contains(point) or  */(std::abs(GetAngleBoxPoint(point, node->boundingBox, node->value.CM)) >= this->thetaSquared);
    }

    OctTree<_MassedNodeInfo> *octTree;
    mutable std::vector<std::pair<const Node*, bool>> nodes_stack_;
    double theta, thetaSquared;

public:
    GravityTree(const T &ll, const T &ur, double theta): octTree(new OctTree<_MassedNodeInfo>(_MassedNodeInfo(ll), _MassedNodeInfo(ur))), theta(theta), thetaSquared(theta * theta){};

    ~GravityTree(){delete this->octTree;};

    inline void calculateMasses(){this->calculateMassHelper(this->octTree->getRoot());};

    inline bool build(const std::vector<MassedPoint<T>> &points)
    {
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
            // always push the child that contains the node
            if(!node->isValue and (containsPoint or this->shouldOpenBox(point, node)))
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
                // do not open the box
                const T temp = node->value.CM - point;
                gravity_result_t length = fastabs(temp);
                if(length >= EPSILON)
                {
                    // otherwise, this leaf is the point itself. gravity sould not be calculated
                    gravity_result_t sizeOfForce = 1 / (length * length * length);
                    gravity += (temp * sizeOfForce) * node->value.mass; // will create a vector in the direction of `temp`, which is in length 1/|temp|^2
                }
            }
        }
        return gravity;
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

#endif // _GRAVITY_TREE_HPP