#ifndef HILBERT_ORDINARY_CONVERTOR_3D_HPP
#define HILBERT_ORDINARY_CONVERTOR_3D_HPP

#include "../HilbertConvertor3D.hpp"
#include "../HilbertOrder3D.hpp"

class HilbertOrdinaryConvertor3D : public HilbertConvertor3D
{
public:
    explicit HilbertOrdinaryConvertor3D(const Vector3D &ll, const Vector3D &ur, size_t order);
    
    ~HilbertOrdinaryConvertor3D() override = default;
    
    void changeOrder(size_t order) override
    {
        this->order = order;
    }
    
    hilbert_index_t xyz2d(coord_t x, coord_t y, coord_t z) const override;
        
    Vector3D d2xyz(hilbert_index_t d) const override;
    

    inline std::shared_ptr<HilbertConvertor3D> clone(void) const override
    {
        return std::make_shared<HilbertOrdinaryConvertor3D>(this->ll, this->ur, this->order);
    }

private:
    HilbertCurve3D curve;
    Vector3D length;
};

inline HilbertOrdinaryConvertor3D::HilbertOrdinaryConvertor3D(const Vector3D &ll, const Vector3D &ur, size_t order)
    : HilbertConvertor3D(ll, ur, order)
{
    this->length = this->ur - this->ll; 
}

inline Vector3D HilbertOrdinaryConvertor3D::d2xyz(hilbert_index_t d) const 
{
    throw UniversalError("HilbertOrdinaryConvertor3D::d2xyz: not implemented yet");
}

inline hilbert_index_t HilbertOrdinaryConvertor3D::xyz2d(coord_t x, coord_t y, coord_t z) const
{
    Vector3D vec;
    vec.x = (x - this->ll[0]) / this->length[0];
    vec.y = (y - this->ll[1]) / this->length[1];
    vec.z = (z - this->ll[2]) / this->length[2];
    return this->curve.Hilbert3D_xyz2d(vec, this->order);
}


#endif // HILBERT_ORDINARY_CONVERTOR_3D_HPP