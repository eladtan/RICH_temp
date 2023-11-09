#include "HilbertTree2D.hpp"

HilbertTree2D::HilbertTree2D(const Vector2D &ll, const Vector2D &ur, size_t order): root(nullptr)
{
    coord_t realWidth = ur.x - ll.x;
    coord_t realHeight = ur.y - ll.y;

    // calculate divisions number in x axis and y axis
    DirectionVector2D div;
    div.x = std::ceil(std::sqrt(realWidth / realHeight) * std::pow(2, order));
    div.y = std::ceil(std::pow(4, order) / div.x);
    
    this->step = Vector2D(realWidth / div.x, realHeight / div.y);
    
    this->root = new Node();
    hilbert_index_t d = 0;
    this->buildHelper(this->root, {0, 0}, {div.x, 0}, {0, div.y}, d);
}

void HilbertTree2D::deleteTreeHelper(Node *root)
{
    if(root == nullptr)
    {
        return;
    }
    for(Node *child : root->children)
    {
        this->deleteTreeHelper(child);
    }
    delete root;
}

Vector2D HilbertTree2D::WidthHeightToXY(int width, int height) const
{
    coord_t x, y;
    x = this->ll[0] + width * this->step[0];
    y = this->ll[1] + height * this->step[1];
    // std::cout << "translating (" << width << ", " << height << ") to (" << x << ", " << y << ")" << std::endl;
    return Vector2D(x, y);
}

void HilbertTree2D::Node::calculateLLUR()
{
    this->ll = Vector2D(std::numeric_limits<coord_t>::max(), std::numeric_limits<coord_t>::max());
    this->ur = Vector2D(std::numeric_limits<coord_t>::min(), std::numeric_limits<coord_t>::min());
    for(const Node *child : this->children)
    {
        this->ll.x = std::min(this->ll.x, child->ll.x);
        this->ll.y = std::min(this->ll.y, child->ll.y);
        this->ur.x = std::max(this->ur.x, child->ur.x);
        this->ur.y = std::max(this->ur.y, child->ur.y);
    }
}

void HilbertTree2D::buildBaseStep(Node *node, const DirectionVector2D &startPoint, int length, const DirectionVector2D &direction, hilbert_index_t &d)
{
    //std::cout << "in trivial row fill" << std::endl;
    coord_t curr_x = startPoint.x, curr_y = startPoint.y;
    node->ll = this->WidthHeightToXY(curr_x, curr_y);
    node->ur = this->WidthHeightToXY(curr_x, curr_y);

    for(int i = 0; i < length; i++)
    {
        Vector2D real_ll = this->WidthHeightToXY(curr_x, curr_y);
        Vector2D real_ur = this->WidthHeightToXY(curr_x + 1, curr_y + 1);
        node->children.push_back(new Node(real_ll, real_ur));
        node->children.back()->d_start = d;
        node->children.back()->d_end = d + 1;
        node->children.back()->isLeaf = true;

        curr_x += direction.x;
        curr_y += direction.y;
        d++;
    }
    node->calculateLLUR();
    node->d_end = d;
    return;
}

/**
 * see here the algorithm: https://github.com/jakubcerveny/gilbert
*/
void HilbertTree2D::buildHelper(Node *node, const DirectionVector2D &startPoint, const DirectionVector2D &a, const DirectionVector2D &b, hilbert_index_t &d)
{
    node->isLeaf = false;
    node->d_start = d;

    int width = std::abs(a.x + a.y);
    int height = std::abs(b.x + b.y);

    int dax = SIGN(a.x), day = SIGN(a.y);
    int dbx = SIGN(b.x), dby = SIGN(b.y);

    // base cases
    if(height == 1)
    {
        buildBaseStep(node, startPoint, width, {dax, day}, d);
        return;
    }

    if(width == 1)
    {
        buildBaseStep(node, startPoint, height, {dbx, dby}, d);
        return;
    }

    DirectionVector2D a2 = {a.x >> 1, a.y >> 1}; /* (a.x//2, a.y//2) */
    DirectionVector2D b2 = {b.x >> 1, b.y >> 1}; /* (b.x//2, b.y//2) */

    int width2 = std::abs(a2.x + a2.y);
    int height2 = std::abs(b2.x + b2.y);

    if(2 * width > 3 * height)
    {
        if((width2 % 2) and (width > 2))
        {
            // prefer even steps
            a2.x = a2.x + dax;
            a2.y = a2.y + day;
        }
        
        node->children.push_back(new Node());
        this->buildHelper(node->children.back(), startPoint, a2, b, d);
        node->children.push_back(new Node());
        this->buildHelper(node->children.back(), {startPoint.x + a2.x, startPoint.y + a2.y}, {a.x - a2.x, a.y - a2.y}, b, d);
    }
    else
    {
        // std::cout << "here" << std::endl;
        if((height2 % 2) and (height > 2))
        {
            // prefer even steps
            b2.x = b2.x + dbx;
            b2.y = b2.y + dby;
        }
        // up
        node->children.push_back(new Node());
        this->buildHelper(node->children.back(), startPoint, b2, a2, d);
        // long horizontal
        node->children.push_back(new Node());
        this->buildHelper(node->children.back(), {startPoint.x + b2.x, startPoint.y + b2.y}, a, {b.x - b2.x, b.y - b2.y}, d);
        // down
        node->children.push_back(new Node());
        this->buildHelper(node->children.back(),
                        {startPoint.x + (a.x - dax) + (b2.x - dbx), startPoint.y + (a.y - day) + (b2.y - dby)}, 
                        {-b2.x, -b2.y}, {-(a.x - a2.x), -(a.y - a2.y)}, d);
    }

    node->calculateLLUR();
    node->d_end = d;
}

hilbert_index_t HilbertTree2D::xy2d(coord_t x, coord_t y) const
{
    if(this->root == nullptr)
    {
        throw UniversalError("Tree has not been built yet");
    }
    const Node *node = this->root;
    
    while(!node->isLeaf)
    {
        bool found = false;
        // find the child that contains `(x, y)`
        for(const Node *child : node->children)
        {
            if(child->contains(x, y))
            {
                node = child;
                found = true;
                break;
            }
        }
        if(not found)
        {
           throw UniversalError("Node does not contain a child that contains the given point (in HilbertTree2D)"); // should not reach here
        }
    }
    return node->d_start;
}

Vector2D HilbertTree2D::d2xy(hilbert_index_t d) const
{
    if(this->root == nullptr)
    {
        throw UniversalError("Tree has not been built yet");
    }
    const Node *node = this->root;
    
    while(!node->isLeaf)
    {
        bool found = false;
        // find the child that contains `d`
        for(const Node *child : node->children)
        {
            if(child->d_start <= d and d < child->d_end)
            {
                node = child;
                found = true;
                break;
            }
        }
        if(not found)
        {
           throw UniversalError("Node does not contain a child that contains the given order d (in HilbertTree2D)"); // should not reach here
        }
    }
    return (node->ll + node->ur) / 2;
}