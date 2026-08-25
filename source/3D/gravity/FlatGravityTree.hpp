#ifndef FLAT_GRAVITY_TREE_HPP
#define FLAT_GRAVITY_TREE_HPP

#include <vector>
#include <array>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include "3D/elementary/Vector3D.hpp"
#include "GravityTree.hpp"

struct FlatGravityNode
{
    double widthSquared;
    Vector3D CM;
    Vector3D center;
    int32_t firstChild;
    uint8_t childMask;
    uint8_t _pad[3];

    double mass;
    std::array<double, 6> Q;
};

class FlatGravityTree
{
public:
    using TraversalStack = std::vector<std::pair<int32_t, bool>>;

    explicit FlatGravityTree(const GravityTree<Vector3D> &tree)
        : thetaSquared(tree.getTheta() * tree.getTheta()),
          quadrupole(tree.getQuadrupole())
    {
        const auto *root = tree.getOctTree()->getRoot();
        if(root == nullptr)
            return;
        const std::size_t nodeCount = this->countNodes(root);
        if(nodeCount > static_cast<std::size_t>(
                std::numeric_limits<int32_t>::max()))
            throw std::overflow_error(
                "Flat gravity tree exceeds 32-bit node indexing");
        this->nodes.resize(nodeCount);
        std::size_t nextFree = 1;
        this->compileNode(root, 0, nextFree);
        if(nextFree != nodeCount)
            throw std::logic_error(
                "Flat gravity tree node count changed while compiling");
    }

    std::size_t nodeCount() const
    {
        return this->nodes.size();
    }

    Vector3D gravity(const Vector3D &point, bool rootContainsPoint) const
    {
        TraversalStack scratch;
        scratch.reserve(128);
        return this->gravity(point, rootContainsPoint, scratch);
    }

    Vector3D gravity(const Vector3D &point, bool rootContainsPoint,
                     TraversalStack &scratch) const
    {
        Vector3D result;
        if(this->nodes.empty())
            return result;

        scratch.clear();
        scratch.push_back({0, rootContainsPoint});

        while(!scratch.empty())
        {
            auto [idx, containsPoint] = scratch.back();
            scratch.pop_back();

            const FlatGravityNode &n = this->nodes[idx];
            bool isLeaf = (n.firstChild < 0);

            bool shouldOpen = false;
            if(!isLeaf)
            {
                if(containsPoint)
                {
                    shouldOpen = true;
                }
                else
                {
                    double dx = point[0] - n.CM[0];
                    double dy = point[1] - n.CM[1];
                    double dz = point[2] - n.CM[2];
                    double dist2 = dx * dx + dy * dy + dz * dz;
                    shouldOpen = (n.widthSquared >= dist2 * this->thetaSquared);
                }
            }

            if(shouldOpen)
            {
                int containsOctant = -1;
                if(containsPoint)
                {
                    containsOctant = 0;
                    for(int i = 0; i < 3; i++)
                        containsOctant = (containsOctant << 1) | ((n.center[i] < point[i]) ? 1 : 0);

                    if(n.childMask & (1 << containsOctant))
                    {
                        int childIdx = n.firstChild + __builtin_popcount(n.childMask & ((1 << containsOctant) - 1));
                        scratch.push_back({childIdx, true});
                    }
                }

                int childOffset = 0;
                for(int i = 0; i < 8; i++)
                {
                    if(n.childMask & (1 << i))
                    {
                        if(i != containsOctant)
                            scratch.push_back(
                                {n.firstChild + childOffset, false});
                        childOffset++;
                    }
                }
            }
            else
            {
                result += CalculateLeafGravityContribution(n, point, this->quadrupole);
            }
        }
        return result;
    }

private:
    std::vector<FlatGravityNode> nodes;
    double thetaSquared;
    bool quadrupole;

    using PtrNode = typename GravityTree<Vector3D>::Node;

    std::size_t countNodes(const PtrNode *node) const
    {
        if(node == nullptr)
            return 0;
        std::size_t result = 1;
        if(!node->isLeaf)
            for(int i = 0; i < 8; ++i)
                result += this->countNodes(node->children[i]);
        return result;
    }

    void compileNode(const PtrNode *node, int32_t slotIdx,
                     std::size_t &nextFree)
    {
        nodes[slotIdx].widthSquared = node->boundingBox.getWidthSquared();
        nodes[slotIdx].CM = node->value.CM;
        nodes[slotIdx].center = Vector3D(node->value[0], node->value[1], node->value[2]);
        nodes[slotIdx].mass = node->value.mass;
        nodes[slotIdx].Q = node->value.Q;

        if(node->isLeaf)
        {
            nodes[slotIdx].firstChild = -1;
            nodes[slotIdx].childMask = 0;
            return;
        }

        uint8_t mask = 0;
        int numChildren = 0;
        for(int i = 0; i < 8; i++)
        {
            if(node->children[i] != nullptr)
            {
                mask |= static_cast<uint8_t>(1 << i);
                numChildren++;
            }
        }

        int32_t firstChild = static_cast<int32_t>(nextFree);
        nodes[slotIdx].firstChild = firstChild;
        nodes[slotIdx].childMask = mask;
        nextFree += static_cast<std::size_t>(numChildren);

        int childOffset = 0;
        for(int i = 0; i < 8; i++)
        {
            if(node->children[i] != nullptr)
            {
                this->compileNode(node->children[i],
                                  firstChild + childOffset, nextFree);
                childOffset++;
            }
        }
    }
};

#endif // FLAT_GRAVITY_TREE_HPP
