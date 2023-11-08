#include "HilbertTree2D.hpp"


HilbertTree2D::HilbertTree2D(const Vector2D &ll, const Vector2D &ur, size_t order): root(nullptr)
{
    coord_t realWidth = ur[0] - ll[0];
    coord_t realHeight = ur[1] - ll[1];

    // calculate divisions number in x axis and y axis
    this->div[0] = std::ceil(std::sqrt(realWidth / realHeight) * std::pow(2, order));
    this->div[1] = std::ceil(std::pow(4, order) / this->div[0]);
    
    this->step = Vector2D(realWidth / this->div[0], realHeight / this->div[1]);
    
    this->root = new Node();
    this->buildHelper(this->root, {0, 0}, {this->div[0], 0}, {0, this->div[1]}, 0);
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

void HilbertTree2D::buildBaseStep(Node *node, const std::pair<int, int> &startPoint, int length, const std::pair<int, int> &direction, hilbert_index_t d_start)
{
    //std::cout << "in trivial row fill" << std::endl;
    coord_t curr_x = startPoint.first, curr_y = startPoint.second;
    node->ll = this->WidthHeightToXY(curr_x, curr_y);
    node->ur = this->WidthHeightToXY(curr_x, curr_y);

    for(int i = 0; i < length; i++)
    {
        Vector2D real_ll = this->WidthHeightToXY(curr_x, curr_y);
        Vector2D real_ur = this->WidthHeightToXY(curr_x + 1, curr_y + 1);
        node->children.push_back(new Node(real_ll, real_ur));
        node->children.back()->d_start = d_start;
        node->children.back()->d_end = d_start + 1;
        node->children.back()->isLeaf = true;

        curr_x += direction.first;
        curr_y += direction.second;
        d_start++;
    }
    node->calculateLLUR();
    node->d_end = d_start;
    return;
}

/**
 * see here the algorithm: https://github.com/jakubcerveny/gilbert
*/
void HilbertTree2D::buildHelper(Node *node, const std::pair<int, int> &startPoint, const std::pair<int, int> &a, const std::pair<int, int> &b, hilbert_index_t d_start)
{
    node->isLeaf = false;
    node->d_start = d_start;

    int width = std::abs(a.first + a.second);
    int height = std::abs(b.first + b.second);

    int dax = SIGN(a.first), day = SIGN(a.second);
    int dbx = SIGN(b.first), dby = SIGN(b.second);

    // base cases
    if(height == 1)
    {
        buildBaseStep(node, startPoint, width, {dax, day}, d_start);
        return;
    }

    if(width == 1)
    {
        buildBaseStep(node, startPoint, height, {dbx, dby}, d_start);
        return;
    }

    std::pair<int, int> a2 = {a.first >> 1, a.second >> 1}; /* (a.x//2, a.y//2) */
    std::pair<int, int> b2 = {b.first >> 1, b.second >> 1}; /* (b.x//2, b.y//2) */

    int width2 = std::abs(a2.first + a2.second);
    int height2 = std::abs(b2.first + b2.second);

    if(2 * width > 3 * height)
    {
        if((width2 % 2) and (width > 2))
        {
            // prefer even steps
            a2.first = a2.first + dax;
            a2.second = a2.second + day;
        }
        
        node->children.push_back(new Node());
        this->buildHelper(node->children.back(), startPoint, a2, b, d_start);
        d_start = node->children.back()->d_end;
        node->children.push_back(new Node());
        this->buildHelper(node->children.back(), {startPoint.first + a2.first, startPoint.second + a2.second}, {a.first - a2.first, a.second - a2.second}, b, d_start);
        d_start = node->children.back()->d_end;
    }
    else
    {
        // std::cout << "here" << std::endl;
        if((height2 % 2) and (height > 2))
        {
            // prefer even steps
            b2.first = b2.first + dbx;
            b2.second = b2.second + dby;
        }
        // up
        node->children.push_back(new Node());
        this->buildHelper(node->children.back(), startPoint, b2, a2, d_start);
        d_start = node->children.back()->d_end;
        // long horizontal
        node->children.push_back(new Node());
        this->buildHelper(node->children.back(), {startPoint.first + b2.first, startPoint.second + b2.second}, a, {b.first - b2.first, b.second - b2.second}, d_start);
        d_start = node->children.back()->d_end;
        // down
        node->children.push_back(new Node());
        this->buildHelper(node->children.back(),
                        {startPoint.first + (a.first - dax) + (b2.first - dbx), startPoint.second + (a.second - day) + (b2.second - dby)}, 
                        {-b2.first, -b2.second}, {-(a.first - a2.first), -(a.second - a2.second)}, d_start);
        d_start = node->children.back()->d_end;
    }

    node->calculateLLUR();
    node->d_end = d_start;
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
        // find the child that contains `(x, y)`
        for(const Node *child : node->children)
        {
            if(child->contains(x, y))
            {
                node = child;
                break;
            }
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
        // find the child that contains `d`
        for(const Node *child : node->children)
        {
            if(child->d_start <= d and d < child->d_end)
            {
                node = child;
                break;
            }
        }
    }
    return (node->ll + node->ur) / 2;
}