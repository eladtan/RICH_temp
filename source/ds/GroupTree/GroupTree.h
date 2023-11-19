#ifndef _GROUP_TREE_RICH_H
#define _GROUP_TREE_RICH_H

#ifdef DEBUG_MODE
#include <iostream>
#endif

#include <utility>
#include <vector>
#include <array>
#include <functional>
#include <algorithm>
#include <assert.h>


template<typename T, int N>
class GroupTree
{
protected:
    using Compare = std::function<bool(const T&, const T&)>; 

    class Node
    {
    public:
        inline Node(const T &value): numValues(1), right(nullptr), left(nullptr), parent(nullptr), minInSubtree(T()), maxInSubtree(T()), height(0), depth(0){this->values[0] = value;};
        inline explicit Node(): Node(T()){};
        template<typename Container>
        Node(const Container &container);
        Node(const std::array<T, N> &values, int numValues);
        virtual ~Node() = default;

        std::array<T, N> values;
        int numValues;
        Node *left, *right, *parent;
        T minInSubtree;
        T maxInSubtree;
        int height, depth;

    protected:
        void getAllDecendantsHelper(std::vector<T> &vec);
    
    friend class GroupTree<T, N>;
    };

    Node *root;
    size_t treeSize;
    Compare compare;

    virtual void splitNode(Node *node);

    #ifdef DEBUG_MODE
    void printHelper(const Node *node, int indent) const;
    #endif // DEBUG_MODE

    virtual inline void deleteHelper(Node *node){if(node == nullptr) return; deleteHelper(node->left); deleteHelper(node->right); delete node;};
    virtual inline void deleteTree(){this->deleteHelper(this->getRoot()); this->setRoot(nullptr); this->treeSize = 0;};

    virtual inline const Node *getRoot() const{return this->root;};
    virtual inline Node *getRoot(){return this->root;};
    virtual inline void setRoot(Node *other){this->root = other;};

    inline std::vector<T> getAllDecendants(Node *node){std::vector<T> vec; if(node != nullptr) node->getAllDecendantsHelper(vec); return vec;};

    virtual void updateNodeInfo(Node *node);
    inline virtual void recursiveUpdateNodeInfo(Node *node){while(node != nullptr){this->updateNodeInfo(node); node = node->parent;}};
    
private:
    template<typename RandomAccessIterator>
    Node *fastBuildHelper(const RandomAccessIterator &first, const RandomAccessIterator &last);

public:
    inline GroupTree(Node *root, const Compare &compare): root(root), compare(compare), treeSize(0){};
    inline GroupTree(const Compare &compare): GroupTree(nullptr, compare){};
    inline GroupTree(): GroupTree([](const T &a, const T &b){return a <= b;}){};
    virtual ~GroupTree(){this->deleteTree();};

    const typename GroupTree<T, N>::Node *tryFind(const T &value) const;
    typename GroupTree<T, N>::Node *tryFind(const T &value){return const_cast<Node*>((std::as_const(*this)).tryFind(value));};
    virtual typename GroupTree<T, N>::Node *tryInsert(const T &value);

    inline bool find(const T &value) const{return this->tryFind(value) != nullptr;};
    bool insert(const T &value);

    #ifdef DEBUG_MODE
    inline void print() const{this->printHelper(this->getRoot(), 0);};
    #endif // DEBUG_MODE
};

#endif // _GROUP_TREE_RICH_H