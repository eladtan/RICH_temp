#ifndef HILBERT_RECTANGULAR_CONVERTOR_3D_HPP
#define HILBERT_RECTANGULAR_CONVERTOR_3D_HPP

#include <array>
#include "../HilbertConvertor3D.hpp"

#define MAX_HILBERT_DEPTH 54

/**
 * see here the algorithm: https://github.com/jakubcerveny/gilbert
*/
class HilbertRectangularConvertor3D : public HilbertConvertor3D
{
    template<int max_leaf_ranks>
    friend class HilbertTree3D;

private:
    using direction_t = long int;

    struct DirectionVector3D
    {
        direction_t x, y, z;

        #ifdef DEBUG_MODE
        friend std::ostream &operator<<(std::ostream &os, const DirectionVector3D &args)
        {
            return os << "(" << args.x << ", " << args.y << ", " << args.z << ")";
        }
        #endif // DEBUG_MODE
    };

    struct RecursionArguments
    {
        DirectionVector3D startPoint;
        DirectionVector3D a;
        DirectionVector3D b;
        DirectionVector3D c;

        #ifdef DEBUG_MODE
        friend std::ostream &operator<<(std::ostream &os, const RecursionArguments &args)
        {
            return os << "startPoint = " << args.startPoint << ", a = " << args.a << ", b = " << args.b << ", c = " << args.c;
        }
        #endif // DEBUG_MODE
    };

    Vector3D step;
    BoundingBox<Vector3D> spaceBoundingBox;
    DirectionVector3D div;
    hilbert_index_t total_points_num;

    mutable std::array<std::vector<RecursionArguments>, MAX_HILBERT_DEPTH> argumentsBuffer;

public:
    explicit HilbertRectangularConvertor3D(const Vector3D &ll, const Vector3D &ur, size_t order);
    
    ~HilbertRectangularConvertor3D() override = default;
    
    inline hilbert_index_t getHilbertSize() const{return this->total_points_num;};
    
    void changeOrder(size_t order) override;
    
    hilbert_index_t xyz2d(coord_t x, coord_t y, coord_t z) const override;
    
    Vector3D d2xyz(hilbert_index_t d) const override;

    inline std::shared_ptr<HilbertConvertor3D> clone(void) const override
    {
        return std::make_shared<HilbertRectangularConvertor3D>(this->ll, this->ur, this->order);
    }
        
private:
    void setRecursionArguments(const RecursionArguments &args, size_t currentDepth) const;
    
    bool d2xyz_helper(const RecursionArguments &args, size_t currentDepth, hilbert_index_t requested_d, hilbert_index_t &current_d, Vector3D &result) const;
    
    bool xyz2d_helper_base(const DirectionVector3D &startPoint, size_t steps, const DirectionVector3D &direction, const DirectionVector3D &requested_point, hilbert_index_t &current_d) const;
    
    bool xyz2d_helper(const RecursionArguments &args, size_t currentDepth, const DirectionVector3D &requested_point, hilbert_index_t &current_d) const;
    
    std::pair<DirectionVector3D, DirectionVector3D> getBoundingBox(const RecursionArguments &args) const;
    
    Vector3D WidthHeightDepthToXYZ(direction_t width, direction_t height, direction_t depth) const;
};

#endif // HILBERT_RECTANGULAR_CONVERTOR_3D_HPP