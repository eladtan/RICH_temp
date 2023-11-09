#include "HilbertTree3D.hpp"

HilbertTree3D::HilbertTree3D(const Vector3D &ll, const Vector3D &ur, size_t order): root(nullptr)
{
    coord_t realWidth = ur.x - ll.x;
    coord_t realHeight = ur.y - ll.y;
    coord_t realDepth = ur.z - ll.z;

    // calculate divisions number in x axis and y axis
    DirectionVector3D div;
    div.x = std::ceil(std::pow((realWidth * realWidth) / (realHeight * realDepth), 0.333333333) * std::pow(2, order));
    div.y = std::ceil(std::pow((realHeight * realHeight) / (realWidth * realDepth), 0.333333333) * std::pow(2, order));
    div.z = std::ceil(std::pow(8, order) / (div.x * div.y));
    
    this->step = Vector3D(realWidth / div.x, realHeight / div.y, realDepth / div.z);
    std::cout << "step is " << this->step << ", div is (" << div.x << ", " << div.y << ", " << div.z << ")" << std::endl;
    this->root = new Node();
    hilbert_index_t d = 0;
    this->buildHelper(this->root, {0, 0, 0}, {div.x, 0, 0}, {0, div.y, 0}, {0, 0, div.z}, d);
}

void HilbertTree3D::deleteTreeHelper(Node *root)
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

Vector3D HilbertTree3D::WidthHeightDepthToXYZ(int width, int height, int depth) const
{
    coord_t x, y, z;
    x = this->ll[0] + width * this->step[0];
    y = this->ll[1] + height * this->step[1];
    z = this->ll[2] + depth * this->step[2];
    // std::cout << "translating (" << width << ", " << height << ") to (" << x << ", " << y << ")" << std::endl;
    return Vector3D(x, y, z);
}

void HilbertTree3D::Node::calculateLLUR()
{
    this->ll = Vector3D(std::numeric_limits<coord_t>::max(), std::numeric_limits<coord_t>::max(), std::numeric_limits<coord_t>::max());
    this->ur = Vector3D(std::numeric_limits<coord_t>::min(), std::numeric_limits<coord_t>::min(), std::numeric_limits<coord_t>::min());
    for(const Node *child : this->children)
    {
        this->ll.x = std::min(this->ll.x, child->ll.x);
        this->ll.y = std::min(this->ll.y, child->ll.y);
        this->ll.z = std::min(this->ll.z, child->ll.z);
        this->ur.x = std::max(this->ur.x, child->ur.x);
        this->ur.y = std::max(this->ur.y, child->ur.y);
        this->ur.z = std::max(this->ur.z, child->ur.z);
    }
}

void HilbertTree3D::buildBaseStep(Node *node, const DirectionVector3D &startPoint, int length, const DirectionVector3D &direction, hilbert_index_t &d)
{
    //std::cout << "in trivial row fill" << std::endl;
    coord_t x = startPoint.x, y = startPoint.y, z = startPoint.z;
    node->ll = node->ur = this->WidthHeightDepthToXYZ(x, y, z);

    for(int i = 0; i < length; i++)
    {
        Vector3D real_ll = this->WidthHeightDepthToXYZ(x, y, z);
        Vector3D real_ur = this->WidthHeightDepthToXYZ(x + 1, y + 1, z + 1);
        node->children.push_back(new Node(real_ll, real_ur));
        node->children.back()->d_start = d;
        node->children.back()->d_end = d + 1;
        node->children.back()->isLeaf = true;

        x += direction.x;
        y += direction.y;
        z += direction.z;
        d++;
    }
    node->calculateLLUR();
    node->d_end = d;
    return;
}

/**
 * see here the algorithm: https://github.com/jakubcerveny/gilbert
*/
void HilbertTree3D::buildHelper(Node *node, const DirectionVector3D &startPoint, const DirectionVector3D &a, const DirectionVector3D &b, const DirectionVector3D &c, hilbert_index_t &d)
{
    node->isLeaf = false;
    node->d_start = d;

    int width = std::abs(a.x + a.y + a.z);
    int height = std::abs(b.x + b.y + b.z);
    int depth = std::abs(c.x + c.y + c.z);

    int dax = SIGN(a.x), day = SIGN(a.y), daz = SIGN(a.z);
    int dbx = SIGN(b.x), dby = SIGN(b.y), dbz = SIGN(b.z);
    int dcx = SIGN(c.x), dcy = SIGN(c.y), dcz = SIGN(c.z);

    // base cases
    if(height == 1 and depth == 1)
    {
        buildBaseStep(node, startPoint, width, {dax, day, daz}, d);
        return;
    }

    if(width == 1 and depth == 1)
    {
        buildBaseStep(node, startPoint, height, {dbx, dby, dbz}, d);
        return;
    }

    if(width == 1 and height == 1)
    {
        buildBaseStep(node, startPoint, depth, {dcx, dcy, dcz}, d);
        return;
    }

    DirectionVector3D a2 = {a.x >> 1, a.y >> 1, a.z >> 1}; /* (a.x//2, a.y//2, a.z//2) */
    DirectionVector3D b2 = {b.x >> 1, b.y >> 1, b.z >> 1}; /* (b.x//2, b.y//2, b.z//2) */
    DirectionVector3D c2 = {c.x >> 1, c.y >> 1, c.z >> 1}; /* (c.x//2, c.y//2, c.z//2) */

    int width2 = std::abs(a2.x + a2.y + a2.z);
    int height2 = std::abs(b2.x + b2.y + b2.z);
    int depth2 = std::abs(c2.x + c2.y + c2.z);

    // prefer even steps
    if((width2 % 2) and (width > 2))
    {
        a2.x = a2.x + dax;
        a2.y = a2.y + day;
        a2.z = a2.z + daz;
    }

    if((height2 % 2) and (height > 2))
    {
        b2.x = b2.x + dbx;
        b2.y = b2.y + dby;
        b2.z = b2.z + dbz;
    }

    if((depth2 % 2) and (depth > 2))
    {
        c2.x = c2.x + dcx;
        c2.y = c2.y + dcy;
        c2.z = c2.z + dcz;
    }

    const int &x = startPoint.x;
    const int &y = startPoint.y;
    const int &z = startPoint.z;

    if((2 * width > 3 * height) and (2 * width > 3 * depth))
    {
        node->children.push_back(new Node());
        this->buildHelper(node->children.back(), startPoint, a2, b, c, d);
        node->children.push_back(new Node());
        this->buildHelper(node->children.back(), {x + a2.x, y + a2.y, z + a2.z}, {a.x - a2.x, a.y - a2.y, a.z - a2.z}, b, c, d);
    }
    else if(3 * height > 4 * depth)
    {
        node->children.push_back(new Node());
        this->buildHelper(node->children.back(), startPoint, b2, c, a2, d);
        node->children.push_back(new Node());
        this->buildHelper(node->children.back(), {x + b2.x, y + b2.y, z + b2.z}, a, {b.x - b2.x, b.y - b2.y, b.z - b2.z}, c, d);
        node->children.push_back(new Node());
        this->buildHelper(node->children.back(), {x + (a.x - dax) + (b2.x - dbx), y + (a.y - day) + (b2.y - dby), z + (a.z - daz) + (b2.z - dbz)}, {-b2.x, -b2.y, -b2.z}, c, {-(a.x - a2.x), -(a.y - a2.y), -(a.z - a2.z)}, d);
    }
    else if(3 * depth > 4 * height)
    {
        node->children.push_back(new Node());
        this->buildHelper(node->children.back(), startPoint, c2, a2, b, d);
        node->children.push_back(new Node());
        this->buildHelper(node->children.back(), {x + c2.x, y + c2.y, z + c2.z}, a, b, {c.x - c2.x, c.y - c2.y, c.z - c2.z}, d);
        node->children.push_back(new Node());
        this->buildHelper(node->children.back(), {x + (a.x - dax) + (c2.x - dcx), y + (a.y - day) + (c2.y - dcy), z + (a.z - daz) + (c2.z - dcz)}, {-c2.x, -c2.y, -c2.z}, {-(a.x - a2.x), -(a.y - a2.y), -(a.z - a2.z)}, b, d);
    }
    else
    {
        node->children.push_back(new Node());
        this->buildHelper(node->children.back(), startPoint, b2, c2, a2, d);
        node->children.push_back(new Node());
        this->buildHelper(node->children.back(), {x + b2.x, y + b2.y, z + b2.z}, c, a2, {b.x - b2.x, b.y - b2.y, b.z - b2.z}, d);
        node->children.push_back(new Node());
        this->buildHelper(node->children.back(), {x + (b2.x - dbx) + (c.x - dcx), y + (b2.y - dby) + (c.y - dcy), z + (b2.z - dbz) + (c.z - dcz)}, a, {-b2.x, -b2.y, -b2.z}, {-(c.x - c2.x), -(c.y - c2.y), -(c.z - c2.z)}, d);
        node->children.push_back(new Node());
        this->buildHelper(node->children.back(), {x + (a.x - dax) + b2.x + (c.x - dcx), y + (a.y - day) + b2.y + (c.y - dcy), z + (a.z - daz) + b2.z + (c.z - dcz)}, {-c.x, -c.y, -c.z}, {-(a.x - a2.x), -(a.y - a2.y), -(a.z - a2.z)}, {b.x - b2.x, b.y - b2.y, b.z - b2.z}, d);
        node->children.push_back(new Node());
        this->buildHelper(node->children.back(), {x + (a.x - dax) + (b2.x - dbx), y + (a.y - day) + (b2.y - dby), z + (a.z - daz) + (b2.z - dbz)}, {-b2.x, -b2.y, -b2.z}, c2, {-(a.x - a2.x), -(a.y - a2.y), -(a.z - a2.z)}, d);
    }

    node->calculateLLUR();
    node->d_end = d;
}

hilbert_index_t HilbertTree3D::xyz2d(coord_t x, coord_t y, coord_t z) const
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
            if(child->contains(x, y, z))
            {
                node = child;
                found = true;
                break;
            }
        }
        if(not found)
        {
           throw UniversalError("Node does not contain a child that contains the given point (in HilbertTree3D)"); // should not reach here
        }
    }
    return node->d_start;
}

Vector3D HilbertTree3D::d2xyz(hilbert_index_t d) const
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
           throw UniversalError("Node does not contain a child that contains the given order d (in HilbertTree3D)"); // should not reach here
        }
    }
    return (node->ll + node->ur) / 2;
}