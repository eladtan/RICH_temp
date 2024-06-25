#ifndef _AVL_HPP
#define _AVL_HPP

#include "../BinaryTree.hpp"

template<typename T>
class AVLTree : public BinaryTree<T>
{
public:
    inline AVLTree(typename BinaryTree<T>::Node *root, const typename BinaryTree<T>::Compare &compare): BinaryTree<T>(root, compare){};
    inline AVLTree(const typename BinaryTree<T>::Compare &compare): BinaryTree<T>(compare){};
    inline AVLTree(): BinaryTree<T>(){};
    template<typename InputIterator>
    inline AVLTree(const InputIterator &begin, const InputIterator &end): AVLTree<T>(){for(InputIterator it = begin; it < end; it++) this->insert(*it);};
    virtual ~AVLTree() = default;

    virtual bool insert(const T &value) override;
    virtual bool remove(const T &value) override;

protected:
    inline int getBalance(const typename BinaryTree<T>::Node *node) const
    {if(node == nullptr) return 0; int leftHeight = (node->left == nullptr)? -1 : node->left->height; int rightHeight = (node->right == nullptr)? -1 : node->right->height; return leftHeight - rightHeight;}
    void recursiveUpdateNodeInfo(typename BinaryTree<T>::Node *node) override;
    long int validateHelper(const typename AVLTree<T>::Node *node) const override;
};

template<typename T>
void AVLTree<T>::recursiveUpdateNodeInfo(typename BinaryTree<T>::Node *node)
{
    while(node != nullptr)
    {
        this->updateNodeInfo(node);
        typename BinaryTree<T>::Node *continueFrom = node->parent;

        if(this->getBalance(node) >= 2)
        {
            assert(node->left != nullptr);
            continueFrom = node;
            if(this->getBalance(node->left) == -1)
            {
                continueFrom = node->left;
                this->fastRotateLeft(node->left);
            }
            this->fastRotateRight(node);
        }
        else if(this->getBalance(node) <= -2)
        {
            assert(node->right != nullptr);
            continueFrom = node;
            if(this->getBalance(node->right) == 1)
            {
                continueFrom = node->right;
                this->fastRotateRight(node->right);
            }
            this->fastRotateLeft(node);
        }
        node = continueFrom;
    }
}

template<typename T>
bool AVLTree<T>::insert(const T &value)
{
    typename BinaryTree<T>::Node *node = this->_insert(value);
    ++this->treeSize;
    this->recursiveUpdateNodeInfo(node);
    return (node != nullptr);
}

template<typename T>
bool AVLTree<T>::remove(const T &value)
{
    typename BinaryTree<T>::Node *node = this->findNode(value);
    if(node == nullptr)
    {
        return false; // value is not in the tree
    }
    --this->treeSize;
    if(node->duplications > 1)
    {
        --node->duplications;
        return true;
    }
    typename BinaryTree<T>::Node *fixFrom = this->_remove(node);
    assert(node != nullptr);
    typename BinaryTree<T>::Node *fixFromParent = fixFrom->parent;
    this->recursiveUpdateNodeInfo(fixFromParent);
    delete fixFrom;
    if(this->treeSize == 0)
    {
        this->setRoot(nullptr);
    }
    return true;
}

#ifdef DEBUG_MODE
template<typename T>
long int AVLTree<T>::validateHelper(const typename AVLTree<T>::Node *node) const
{
    if(node == nullptr)
    {
        return 0;
    }
    long int leftSize = validateHelper(node->left);
    long int rightSize = validateHelper(node->right);
    int leftHeight = (node->left == nullptr)? -1 : node->left->height;
    int rightHeight = (node->right == nullptr)? -1 : node->right->height;
    if(node->left != nullptr)
    {
        assert(node->left->parent == node);
        assert(node->left != node);
        assert(node->left != node->right);
        assert(node->parent != node->left->parent); // cycle!!
    }
    if(node->right != nullptr)
    {
        assert(node->right->parent == node);
        assert(node->right != node);
        assert(node->left != node->right);
        assert(node->parent != node->right->parent); // cycle!!
    }
    assert(node->height == std::max(leftHeight, rightHeight) + 1);
    assert(node->leftSize == leftSize);
    assert(node->rightSize == rightSize);
    int balance = this->getBalance(node);
    assert(balance >= -1 and balance <= 1);
    return leftSize + rightSize + node->duplications;
}
#endif 

#endif // _AVL_HPP