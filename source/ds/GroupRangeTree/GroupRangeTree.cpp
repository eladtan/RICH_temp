#include "GroupRangeTree.h"

template<typename T, int N>
void GroupRangeTree<T, N>::recreateSubtree(GroupRangeNode *node)
{
    if(node == nullptr)
    {
        return;
    }
    if(this->currentDim == this->dim - 1)
    {
        return;
    }
    delete node->subtree;
    node->subtree = new GroupRangeTree<T, N>(this->currentDim + 1, this->dim);
    node->subtree->build(this->getAllDecendants(node));
    
    // rebuild also my children:
    this->recreateSubtree(dynamic_cast<GroupRangeNode*>(node->left));
    this->recreateSubtree(dynamic_cast<GroupRangeNode*>(node->right));
}

template<typename T, int N>
template<typename RandomAccessIterator>
typename GroupRangeTree<T, N>::GroupRangeNode *GroupRangeTree<T, N>::fastBuildHelper(const RandomAccessIterator &first, const RandomAccessIterator &last)
{
    if(first == last)
    {
        return nullptr;
    }

    GroupRangeNode *node;
    if(last - first <= N)
    {
        node = new GroupRangeNode();
        node->numValues = last - first;
        int j = 0;
        for(RandomAccessIterator it = first; it != last; it++)
        {
            node->values[j++] = *it;
        }
    }
    else
    {
        RandomAccessIterator mid = first + (last - first) / 2;
        node = new GroupRangeNode(*mid);
        GroupRangeNode *left = this->fastBuildHelper(first, mid);
        node->left = left;
        GroupRangeNode *right = this->fastBuildHelper(mid + 1, last); 
        node->right = right;
    }
    if(this->currentDim == this->dim - 1)
    {
        node->subtree = nullptr;
    }
    else
    {
        node->subtree = new GroupRangeTree<T, N>(this->currentDim + 1, this->dim);
        node->subtree->build(this->getAllDecendants(node));
    }
    this->treeSize += node->numValues;
    
    if(node->left != nullptr)
    {
        node->left->parent = node;
        this->updateNodeInfo(node->left);
    }
    if(node->right != nullptr)
    {
        node->right->parent = node;
        this->updateNodeInfo(node->right);
    }
    return node;
}

template<typename T, int N>
void GroupRangeTree<T, N>::splitNode(typename GroupTree<T, N>::Node *node)
{
    if(node == nullptr)
    {
        return;
    }
    // assert(node->numValues > 1);
    
    std::sort(node->values.begin(), node->values.begin() + node->numValues, this->compare);
    int numLeft = node->numValues / 2;
    int numRight = node->numValues - numLeft - 1;
    std::array<T, N> left;
    std::array<T, N> right;
    for(int i = 0; i < numLeft; i++)
    {
        left[i] = node->values[i];
    }
    for(int i = numLeft + 1; i < node->numValues; i++)
    {
        right[i - (numLeft + 1)] = node->values[i];
    }

    if(numLeft > 0 and node->left == nullptr)
    {
        node->left = new GroupRangeNode(left);
        node->left->numValues = numLeft;
        node->left->parent = node;
    }
    if(numRight > 0 and node->right == nullptr)
    {
        node->right = new GroupRangeNode(right);
        node->right->numValues = numRight;
        node->right->parent = node;
    }

    node->numValues = 1;
    node->values[0] = node->values[numLeft];
    node->minInSubtree = node->maxInSubtree = node->values[0];

    this->updateNodeInfo(node->left);
    this->updateNodeInfo(node->right);
    this->recreateSubtree(dynamic_cast<GroupRangeNode*>(node)); // rebuild the sub-trees
}

template<typename T, int N>
typename GroupRangeTree<T, N>::GroupRangeNode *GroupRangeTree<T, N>::tryInsert(const T &value)
{
    if(this->getRoot() == nullptr)
    {
        this->setRoot(new GroupRangeNode(value));
        if(this->currentDim != this->dim - 1)
        {
            this->getRoot()->subtree = new GroupRangeTree<T, N>(this->currentDim + 1, this->dim);
            this->getRoot()->subtree->insert(value);
        }
        return this->getRoot();
    }
    GroupRangeNode *current = this->getRoot();
    while(true)
    {
        if(this->compare(value, current->values[0]))
        {
            if(current->left == nullptr)
            {
                break;
            }
            current = dynamic_cast<GroupRangeNode*>(current->left);
        }
        else
        {
            if(current->right == nullptr)
            {
                break;
            }
            current = dynamic_cast<GroupRangeNode*>(current->right);
        }
    }

    // check if value already exists:
    for(int i = 0; i < current->numValues; i++)
    {
        if(value == current->values[i])
        {
            return current;
        }
    }

    GroupRangeNode *addTo;
    if(current->numValues < N and current->left == nullptr and current->right == nullptr)
    {
        addTo = current;
    }
    else
    {
        if(current->numValues < N)
        {
            this->splitNode(current);
        }
        if(this->compare(value, current->values[0]))
        {
            if(current->left == nullptr)
            {
                // not a leaf, so we need to create it a child
                current->left = new GroupRangeNode(value);
                current->left->parent = current;
                current->left->numValues = 0;
            }
            addTo = dynamic_cast<GroupRangeNode*>(current->left);
        }
        else
        {
            if(current->right == nullptr)
            {
                // not a leaf, so we need to create it a child
                current->right = new GroupRangeNode(value);
                current->right->parent = current;
                current->right->numValues = 0;
            }
            addTo = dynamic_cast<GroupRangeNode*>(current->right);
        }
    }
    addTo->values[addTo->numValues++] = value;

    // add the new value to its ancestors subtrees:
    if(this->currentDim != this->dim - 1)
    {
        GroupRangeNode *node = addTo;
        while(node != nullptr)
        {
            if(node->subtree == nullptr)
            {
                assert(node == addTo);
                // should not happen, unless `node` is `addTo` (because its ancesors' subtrees should exist)
                node->subtree = new GroupRangeTree<T, N>(this->currentDim + 1, this->dim);
            }
            node->subtree->insert(value);
            node = dynamic_cast<GroupRangeNode*>(node->parent);
        }
    }
    return addTo;
}

template<typename T, int N>
void GroupRangeTree<T, N>::rangeHelper(const std::vector<std::pair<typename T::coord_type, typename T::coord_type>> &range, const typename GroupTree<T, N>::Node *node, int coord, std::vector<T> &result) const
{
    if(node == nullptr or coord >= this->dim)
    {
        return;
    }
    if((node->minInSubtree[coord] > range[coord].second) or (node->maxInSubtree[coord] < range[coord].first))
    {
        return;
    }
    if(coord == this->dim - 1)
    {
        // search over the tree to find the maching points
        this->rangeHelper(range, node->left, coord, result);
        for(int i = 0; i < node->numValues; i++)
        {
            const T &value = node->values[i];
            if((value[coord] >= range[coord].first) and (value[coord] <= range[coord].second))
            {
                result.push_back(value);
            }
        }
        this->rangeHelper(range, node->right, coord, result);
    }
    else
    {
        if(node->minInSubtree[coord] >= range[coord].first and node->maxInSubtree[coord] <= range[coord].second)
        {
            // the whole subtree matches this coordinate! move on to the next one
            assert(dynamic_cast<const GroupRangeNode*>(node)->subtree != nullptr); // if the subtree was nullptr, then coord was dim-1.
            dynamic_cast<const GroupRangeNode*>(node)->subtree->rangeHelper(range, dynamic_cast<const GroupRangeNode*>(node)->subtree->getRoot(), coord + 1, result);
        }
        else
        {
            for(int i = 0; i < node->numValues; i++)
            {
                const T &value = node->values[i];
                int j;
                for(j = coord; j < this->dim; j++) if(value[j] < range[j].first or value[j] > range[j].second) break;
                if(j == this->dim) result.push_back(value);
            }
            this->rangeHelper(range, node->right, coord, result);
            this->rangeHelper(range, node->left, coord, result);
        }
    }
}

template<typename T, int N>
std::vector<T> GroupRangeTree<T, N>::circularRange(const T &center, typename T::coord_type radius) const
{
    std::vector<std::pair<typename T::coord_type, typename T::coord_type>> range;
    for(int i = 0; i < this->dim; i++)
    {
        range.push_back(std::make_pair(center[i] - radius, center[i] + radius));
    }
    std::vector<T> result;
    for(const T &possible_value : this->range(range))
    {
        typename T::coord_type distanceSquared = 0;
        for(int i = 0; i < this->dim; i++)
        {
            distanceSquared += (possible_value[i] - center[i]) * (possible_value[i] - center[i]);
        }
        if(distanceSquared <= (radius * radius))
        {
            result.push_back(possible_value);
        }
    }
    return result;
}