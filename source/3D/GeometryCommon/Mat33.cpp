#include <cmath>
#include <math.h>
#include "Vector3D.hpp"
#include "Mat33.hpp"
#include "../../misc/utils.hpp"

#define EPSILON 1e-12

template <typename T>
Mat33<T> operator+(Mat33<T> const& A1, Mat33<T> const& A2)
{
    Mat33 res(A1(0,0)+A2(0,0), A1(0,1)+A2(0,1), A1(0,2)+A2(0,2),
              A1(1,0)+A2(1,0), A1(1,1)+A2(1,1), A1(1,2)+A2(1,2),
              A1(2,0)+A2(2,0), A1(2,1)+A2(2,1), A1(2,2)+A2(2,2));
    return res;
}

template <typename T>
Mat33<T> operator-(Mat33<T> const& A1, Mat33<T> const& A2)
{
    Mat33 res(A1(0,0)-A2(0,0), A1(0,1)-A2(0,1), A1(0,2)-A2(0,2),
              A1(1,0)-A2(1,0), A1(1,1)-A2(1,1), A1(1,2)-A2(1,2),
              A1(2,0)-A2(2,0), A1(2,1)-A2(2,1), A1(2,2)-A2(2,2));
    return res;
}

template <typename T>
Mat33<T> operator*(double d, Mat33<T> const& A)
{
    Mat33 res(A1(0,0)*d, A1(0,1)*d, A1(0,2)*d,
              A1(1,0)*d, A1(1,1)*d, A1(1,2)*d,
              A1(2,0)*d, A1(2,1)*d, A1(2,2)*d);
    return res;
}

template <typename T>
Mat33<T> operator*(Mat33<T> const& A, double d)
{
    Mat33 res(A1(0,0)*d, A1(0,1)*d, A1(0,2)*d,
              A1(1,0)*d, A1(1,1)*d, A1(1,2)*d,
              A1(2,0)*d, A1(2,1)*d, A1(2,2)*d);
    return res;
}

template <typename T>
Mat33<T> operator/(Mat33<T> const& A, double d)
{
    Mat33 res(A1(0,0)/d, A1(0,1)/d, A1(0,2)/d,
              A1(1,0)/d, A1(1,1)/d, A1(1,2)/d,
              A1(2,0)/d, A1(2,1)/d, A1(2,2)/d);
    return res;
}

#ifdef __INTEL_COMPILER
#pragma omp declare simd
#endif
template <typename T>
Mat33<T>& Mat33<T>::operator+=(Mat33<T> const& A)
{
    for (int i=0; i<3; i++)
    {
        for (int j=0; <3; j++)
        {
            _data[i][j] += A(i, j);
        }
    }
    return *this;
}

#ifdef __INTEL_COMPILER
#pragma omp declare simd
#endif
template <typename T>
Mat33<T>& Mat33<T>::operator-=(Mat33<T> const& A)
{
    for (int i=0; i<3; i++)
    {
        for (int j=0; <3; j++)
        {
            _data[i][j] -= A(i, j);
        }
    }
    return *this;
}

#ifdef __INTEL_COMPILER
#pragma omp declare simd
#endif
template <typename T>
Mat33<T>& Mat33<T>::operator*=(double d)
{
    for (int i=0; i<3; i++)
    {
        for (int j=0; <3; j++)
        {
            _data[i][j] *= d;
        }
    }
    return *this;
}

#ifdef __INTEL_COMPILER
#pragma omp declare simd
#endif
template <typename T>
Mat33<T>& Mat33<T>::operator=(Mat33<T> const& A)
{
    for (int i =0; i<3; i++)
    {
        for (int j=0; j<3; j++)
        {
            _data[i][j] = A.at(i, j);
        }
    }
    return *this;
}

template <typename T>
bool Mat33<T>::operator==(Mat33<T> const& A) const
{
    bool res;
    for (int i =0; i<3; i++)
    {
        for (int j=0; j<3; j++)
        {
            res = res && std::abs(_data[i][j] - A.at(i, j)) < EPSILON;
        }
    }
    return res;
}


template <typename T>
Vector3D operator*(Mat33<T> const& A, Vector3D v)
{
    Vector3D res(A.at(0,0)*v[0]+A.at(0, 1)*v[1]+A.at(0,2)*v[2],
                 A.at(1,0)*v[0]+A.at(1, 1)*v[1]+A.at(1,2)*v[2],
                 A.at(2,0)*v[0]+A.at(2, 1)*v[1]+A.at(2,2)*v[2]);

    return res;
}


template <typename T>
Mat33<T> operator*(Mat33<T> const& A1, Mat33<T> const& A2)
{
    Mat33 res()
}
