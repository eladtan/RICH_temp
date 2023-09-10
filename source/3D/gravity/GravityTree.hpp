#ifndef _GRAVITY_TREE_HPP
#define _GRAVITY_TREE_HPP

#include "GravityTypes.h"
#include "3D/elementary/Vector3D.hpp"
#include "ds/OctTree/OctTree.hpp"

namespace
{
    double GetAngleBoxPoint(const Vector3D &point, const _BoundingBox<Vector3D> boundingBox, const Vector3D &centerOfMass)
    {
        typename Vector3D::coord_type L = ScalarProd(boundingBox.ur - boundingBox.ll, boundingBox.ur - boundingBox.ll);
        typename Vector3D::coord_type D = ScalarProd(point - centerOfMass, point - centerOfMass);
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

struct _Massed3DPoint
{
    Vector3D point;
    gravity_result_t mass;

    explicit _Massed3DPoint(const Vector3D &point, gravity_result_t mass): point(point), mass(mass){};
};

class GravityTree : public OctTree<Vector3D>
{
private:
    class GravityTreeNode : public OctTree<Vector3D>::OctTreeNode
    {
    public:
        inline GravityTreeNode(const Vector3D &ll, const Vector3D &ur): OctTreeNode(ll, ur), mass(0), centerOfMass(Vector3D(0, 0, 0)){};

        inline GravityTreeNode(const Vector3D &point, gravity_result_t mass): OctTreeNode(point), mass(mass), centerOfMass(point){};

        inline GravityTreeNode(const Vector3D &point): GravityTreeNode(point, 0){};

        GravityTreeNode(GravityTreeNode *parent, int childNumber): OctTreeNode(parent, childNumber), mass(0), centerOfMass(Vector3D(0, 0, 0)){};

        inline OctTreeNode *addLeafChild(int childIndex, const Vector3D &point) override
        {
            GravityTreeNode *node = new GravityTreeNode(point);
            node->parent = this;
            node->fixHeightsRecursively();
            this->children[childIndex] = node;
            return node;
        }

        inline OctTreeNode *createChild(int childIndex) override
        {
            assert(this->children[childIndex] == nullptr);
            this->children[childIndex] = new GravityTreeNode(this, childIndex);
            return this->children[childIndex];
        }

        inline void setMass(gravity_result_t mass)
        {
            assert(this->isValue);
            this->mass = mass;
            this->centerOfMass = this->value;
            this->updateGravityFieldsRecursively();
        };

        gravity_result_t mass;
        Vector3D centerOfMass; // CM
    
        void updateGravityFieldsRecursively();

        inline void print() const override
        {
            std::cout << "BB: " << this->boundingBox.ll << ", " << this->boundingBox.ur << " (depth: " << this->depth << ", height: " << this->height << "), (Mass: " << this->mass << ", CM: " << this->centerOfMass << ")" << std::endl;
        }
    };

    GravityTreeNode *newRoot;
    double theta;

public:
    GravityTree(const Vector3D &ll, const Vector3D &ur, double theta): newRoot(nullptr), theta(theta){this->setBounds(ll, ur);};

    ~GravityTree() override{this->deleteSubtree(this->getRoot());};

    bool insert(const Vector3D &point, gravity_result_t mass)
    {
        GravityTreeNode *node = dynamic_cast<GravityTreeNode*>(this->tryInsert(point));
        if(node == nullptr)
        {
            std::cerr << "Error, could not insert node" << std::endl;
            return false;
        }
        this->treeSize++;
        node->setMass(mass);
        return true;
    }

    inline bool insert(const _Massed3DPoint &massedPoint){return this->insert(massedPoint.point, massedPoint.mass);};

    inline bool find(const Vector3D &point){return OctTree<Vector3D>::find(point);}

    Vector3D gravityHelper(const Vector3D &point, const GravityTreeNode *node) const;

    inline Vector3D gravity(gravity_result_t mass, const Vector3D &point, const std::vector<int> &directions) const
    {
        return mass * this->gravityHelper(point, dynamic_cast<const GravityTreeNode*>(this->getNodeByDirections(directions)));
    }

    void setBounds(const Vector3D &ll, const Vector3D &ur) override
    {
        assert(this->getRoot() == nullptr);
        this->setRoot(new GravityTreeNode(ll, ur));
        this->getRoot()->parent = nullptr;
    }

    OctTreeNode *getRoot() override{return this->newRoot;};
    const OctTreeNode *getRoot() const override{return this->newRoot;};
    void setRoot(OctTreeNode *newRoot) override{this->newRoot = dynamic_cast<GravityTreeNode*>(newRoot);};
};

void GravityTree::GravityTreeNode::updateGravityFieldsRecursively()
{
    if(!this->isValue)
    {
        // mass is the accumulative mass
        this->mass = 0;
        this->centerOfMass = Vector3D(0, 0, 0);
        for(int i = 0; i < CHILDREN; i++)
        {
            const GravityTreeNode *node = dynamic_cast<const GravityTreeNode*>(this->children[i]);
            if(node != nullptr)
            {
                this->centerOfMass += (node->mass) * (node->centerOfMass);
                this->mass += node->mass;
            }
        }
        this->centerOfMass = this->centerOfMass  / this->mass;
    }
    if(this->parent != nullptr)
    {
        dynamic_cast<GravityTreeNode*>(parent)->updateGravityFieldsRecursively();
    }
}

Vector3D GravityTree::gravityHelper(const Vector3D &point, const GravityTreeNode *node) const
{
    Vector3D gravity(0, 0, 0);
    if(node == nullptr)
    {
        return gravity;
    }
    if((!node->isValue) and (node->boundingBox.contains(point) or (GetAngleBoxPoint(point, node->boundingBox, node->centerOfMass) >= this->theta)))
    {
        // open the box
        for(int i = 0; i < CHILDREN; i++)
        {
            const GravityTreeNode *child = dynamic_cast<const GravityTreeNode*>(node->children[i]);
            gravity += this->gravityHelper(point, child);
        }
    }
    else
    {
        // do not open the box
        const Vector3D temp(node->centerOfMass - point);
        gravity_result_t sizeOfForce = 1 / (std::pow(abs(temp), 3));
        gravity = (temp * sizeOfForce) * node->mass; // will create a vector in the direction of `temp`, which is in length 1/|temp|^2
    }
    return gravity;
}


#endif // _GRAVITY_TREE_HPP