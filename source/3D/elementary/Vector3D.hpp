/*! \file Vector3D.hpp
\brief 3D Geometrical calculations
\author Itai Linial
*/

#ifndef Vector3D_HPP
#define Vector3D_HPP 1

#include <vector>
#include <limits>
#include <cmath>
#include <cmath>
#include <math.h>
#include "misc/utils.hpp"
#include "misc/serializable.hpp"

using std::vector;

#define SIGN(x) ((x > 0) - (x < 0))

#define EPSILON 1e-12

namespace
{
	static inline double my_round(double val)
	{    
		return floor(val + 0.5);
	}
}

//! \brief 3D Mathematical vector
class Vector3D : public Serializable
{
public:
    using coord_type = double;

	//! \brief Component in the x direction
	double x;

	//! \brief Component in the y direction
	double y;

	//! \brief Component in the z direction
	double z;

	/*! \brief Class constructor
	\param ix x Component
	\param iy y Component
	\param iz z Component
	*/
	#ifdef __INTEL_COMPILER
	#pragma omp declare simd
	#endif
	inline Vector3D(double ix, double iy, double iz): x(ix), y(iy), z(iz) {};

	/*! \brief Null constructor
	\details Sets all components to 0
	*/
	explicit inline Vector3D(void): Vector3D(0, 0, 0){};

	/*! \brief Class copy constructor
	\param other Other vector
	*/
	template<typename VectorType>
	inline Vector3D(const VectorType &other): Vector3D(other[0], other[1], other[2]){}

	/*! \brief Set vector components
	\param ix x Component
	\param iy y Component
	\param iz z Component
	*/
	inline void Set(double ix, double iy, double iz) 
	{
		x = ix;
		y = iy;
		z = iz;
	}

  /*! \brief Indexed access to member
    \param index Member index
    \return Reference to member
   */
	double& operator[](size_t index)
	{
		if (index == 0)
			return x;
		if (index == 1)
			return y;
		if (index == 2)
			return z;
		UniversalError eo("Trying to access illegal index of vector");
		eo.addEntry("index", index);
		throw eo;
	}

  /*! \brief Indexed access to member
    \param index Member index
    \return Value of member
   */
	double operator[](size_t index) const
	{
		if (index == 0)
			return x;
		if (index == 1)
			return y;
		if (index == 2)
			return z;
		throw UniversalError("Trying to access illegal index of vector (index " + std::to_string(index) + ")");
	}

	/*! \brief Addition
	\param v Vector to be added
	\return Reference to sum
	*/
	#ifdef __INTEL_COMPILER
	#pragma omp declare simd
	#endif
	inline Vector3D& operator+=(Vector3D const& v)
	{
		x += v.x;
		y += v.y;
		z += v.z;
		return *this;
	}

	/*! \brief Subtraction
	\param v Vector to be subtracted
	\return Difference
	*/
	#ifdef __INTEL_COMPILER
	#pragma omp declare simd
	#endif
	inline Vector3D& operator-=(Vector3D const& v)
	{
		x -= v.x;
		y -= v.y;
		z -= v.z;
		return *this;
	}

	/*! \brief Assignment operator
	\param v Vector to be copied
	\return The assigned value
	*/
	template<typename VectorType>
	#ifdef __INTEL_COMPILER
	#pragma omp declare simd
	#endif
	inline Vector3D& operator=(const VectorType& v)
	{
		x = v[0];
		y = v[1];
		z = v[2];
		return *this;
	}

	/*! \brief Scalar product
	\param s Scalar
	\return Reference to the vector multiplied by scalar
	*/
	#ifdef __INTEL_COMPILER
	#pragma omp declare simd
	#endif
	inline Vector3D& operator*=(double s)
	{
		x *= s;
		y *= s;
		z *= s;
		return *this;
	}

	/*! \brief Compare 3D-Vectors (up to an arbitrary precision)
	\param v Vector to be compared to
	\return True/False - according to the comparison results.
	*/
	bool operator==(Vector3D const& v) const
	{
		// Note - since working with double precision, two vectors are assumed to be "equal",
		// if their coordinates agree up to precision EPSILON
		return (std::abs(x - v.x) < EPSILON) && (std::abs(y - v.y) < EPSILON) && (std::abs(z - v.z) < EPSILON);
	}

	inline bool operator!=(Vector3D const& v) const{return !this->operator==(v);}

	/*! \brief Rotates the vector around the X axes
	\param a Angle of rotation (in radians)
	*/
	inline void RotateX(double a)
	{
		Vector3D v;
		v.x = x;
		v.y = y*cos(a) - z*sin(a);
		v.z = y*sin(a) + z*cos(a);

		*this = v;
	}

	/*! \brief Rotates the vector around the Y axes
	\param a Angle of rotation (in radians)
	*/
	inline void RotateY(double a)
	{
		Vector3D v;
		v.x = x*cos(a) + z*sin(a);
		v.y = y;
		v.z = -x*sin(a) + z*cos(a);

		*this = v;
	}

	/*! \brief Rotates the vector around the Z axes
	\param a Angle of rotation (in radians)
	*/
	inline void RotateZ(double a)
	{
		Vector3D v;
		v.x = x*cos(a) - y*sin(a);
		v.y = x*sin(a) + y*cos(a);
		v.z = z;

		*this = v;
	}

	/*! \brief Integer round of the vector's entries
	*/
	inline void Round()
	{
		x = my_round(x);
		y = my_round(y);
		z = my_round(z);
	}

	inline size_t getChunkSize(void) const
	{
		return 3;
	}

	inline vector<double> serialize(void) const override
	{
		vector<double> res(3);
		res[0] = x;
		res[1] = y;
		res[2] = z;
		return res;
	}

	inline void unserialize(const vector<double>& data)
	{
		assert(data.size() == 3);
		x = data[0];
		y = data[1];
		z = data[2];
	}

	friend std::ostream &operator<<(std::ostream &stream, const Vector3D &vec)
	{
		stream << "(" << vec.x << ", " << vec.y << ", " << vec.z << ")";
		return stream;
	}

	friend std::istream &operator>>(std::istream &stream, Vector3D &vec)
	{
		std::string str;
		std::getline(stream, str, '(');
		std::getline(stream, str, ',');
		vec.x = std::stod(str);
		std::getline(stream, str, ',');
		vec.y = std::stod(str);
		std::getline(stream, str, ')');
		vec.z = std::stod(str);
		return stream;
	}

#ifdef __INTEL_COMPILER
#pragma omp declare simd
#endif
	~Vector3D(void) override {}
};

/*! \brief Norm of a vector
\param v Three dimensional vector
\return Norm of v
*/
#ifdef __INTEL_COMPILER
#pragma omp declare simd
#endif
inline double abs(Vector3D const& v)
{
	return std::sqrt(v.x*v.x + v.y*v.y + v.z*v.z);
}

/*! \brief Norm of a vector, less accurate
\param v Three dimensional vector
\return Norm of v
*/
#ifdef __INTEL_COMPILER
#pragma omp declare simd
#endif
inline double fastabs(Vector3D const& v)
{
	return fastsqrt(v.x*v.x + v.y*v.y + v.z*v.z);
}

/*! \brief Term by term addition
\param v1 First vector
\param v2 Second vector
\return Sum
*/
inline Vector3D operator+(Vector3D const& v1, Vector3D const& v2)
{
	Vector3D res;
	res.x = v1.x + v2.x;
	res.y = v1.y + v2.y;
	res.z = v1.z + v2.z;
	return res;
}

/*! \brief Term by term subtraction
\param v1 First vector
\param v2 Second vector
\return Difference
*/
inline Vector3D operator-(Vector3D const& v1,
	Vector3D const& v2)
{
	Vector3D res;
	res.x = v1.x - v2.x;
	res.y = v1.y - v2.y;
	res.z = v1.z - v2.z;
	return res;
}

/*! \brief Scalar product
\param d Scalar
\param v Vector
\return Three dimensional vector
*/
inline Vector3D operator*(double d, Vector3D const& v)
{
	Vector3D res;
	res.x = v.x * d;
	res.y = v.y * d;
	res.z = v.z * d;
	return res;
}

/*! \brief Scalar product
\param v Vector
\param d Scalar
\return Three dimensional vector
*/
inline Vector3D operator*(Vector3D const& v, double d)
{
	return d*v;
}

/*! \brief Scalar division
\param v Vector
\param d Scalar
\return Three dimensional vector
*/
inline Vector3D operator/(Vector3D const& v, double d)
{
	Vector3D res;
	res.x = v.x / d;
	res.y = v.y / d;
	res.z = v.z / d;
	return res;
}

/*! \brief Scalar product of two vectors
\param v1 3D vector
\param v2 3D vector
\return Scalar product of v1 and v2
*/
#ifdef __INTEL_COMPILER
#pragma omp declare simd
#endif
inline double ScalarProd(Vector3D const& v1, Vector3D const& v2)
{
	return v1.x*v2.x + v1.y*v2.y + v1.z*v2.z;
}

/*! \brief Returns the angle between two vectors (in radians)
\param v1 First vector
\param v2 Second vector
\return Angle (radians)
*/
inline double CalcAngle(Vector3D const& v1, Vector3D const& v2)
{
	return std::acos(std::max(-1.0, std::min(-1.0, ScalarProd(v1, v2) / (abs(v1) * abs(v2)))));
}

/*! \brief Calculates the projection of one vector in the direction of the second
\param v1 First vector
\param v2 Direction of the projection
\return Component of v1 in the direction of v2
*/
inline double Projection(Vector3D const& v1, Vector3D const& v2)
{
	return ScalarProd(v1, v2) / abs(v2);
}

/*! \brief Rotates a 3D-vector around the X axis
\param v Vector
\param a  (in radians)
\return Rotated vector
*/
inline Vector3D RotateX(Vector3D const& v, double a)
{
	Vector3D res;
	res.x = v.x;
	res.y = v.y*cos(a) - v.z*sin(a);
	res.z = v.y*sin(a) + v.z*cos(a);
	return res;
}

/*! \brief Rotates a 3D-vector around the Y axis
\param v Vector
\param a  (in radians)
\return Rotated vector
*/
inline Vector3D RotateY(Vector3D const& v, double a)
{
	Vector3D res;
	res.x = v.x*cos(a) + v.z*sin(a);
	res.y = v.y;
	res.z = -v.x*sin(a) + v.z*cos(a);
	return res;
}

/*! \brief Rotates a 3D-vector around the Z axis
\param v Vector
\param a  (in radians)
\return Rotated vector
*/
inline Vector3D RotateZ(Vector3D const& v, double a)
{
	Vector3D res;
	res.x = v.x*cos(a) - v.y*sin(a);
	res.y = v.x*sin(a) + v.y*cos(a);
	res.z = v.z;
	return res;
}

/*! \brief Reflect vector
\param v Vector
\param normal Normal to the reflection plane
\return Reflection of v about axis
*/
inline Vector3D Reflect(Vector3D const& v, Vector3D const& normal)
{
	return v - 2 * ScalarProd(v, normal)*normal / ScalarProd(normal,normal);
}

/*! \brief Calculates the distance between two vectors
\param v1 First vector
\param v2 Second vector
\return distance between v1 and v2
*/
inline double distance(Vector3D const& v1, Vector3D const& v2)
{
	return abs(v1 - v2);
}

/*! \brief Returns the cross product of two vectors
\param v1 First vector
\param v2 Second vector
\return Cross product between v1 and v2
*/
inline Vector3D CrossProduct(Vector3D const& v1, Vector3D const& v2)
{
	return Vector3D(v1.y*v2.z - v1.z*v2.y, v1.z*v2.x - v1.x*v2.z, v1.x*v2.y - v1.y*v2.x);
}

/*! \brief Cross product
  \param v1 First vector
  \param v2 Second vector
  \param res result
 */
inline void CrossProduct(Vector3D const& v1, Vector3D const& v2,Vector3D &res)
{
	res.x = v1.y*v2.z - v1.z*v2.y;
	res.y = v1.z*v2.x - v1.x*v2.z;
	res.z = v1.x*v2.y - v1.y*v2.x;
}

/*! \brief Normalise vector
  \param vec Vector
  \return Normalised vector
 */
inline Vector3D normalize(Vector3D const& vec)
{
	double l = abs(vec);
	return vec / l;
}

/*! \brief Splits a vector of 3D points to components
\param vIn Input vector of 3D points
\param vX Vector of x coordinates (out)
\param vY Vector of y coordinates (out)
\param vZ Vector of z coordinates (out)
*/
inline void Split(vector<Vector3D> const & vIn, vector<double> & vX, vector<double> & vY, vector<double> & vZ)
{
	vX.resize(vIn.size());
	vY.resize(vIn.size());
	vZ.resize(vIn.size());

	for (std::size_t ii = 0; ii < vIn.size(); ++ii)
	{
		vX[ii] = vIn[ii].x;
		vY[ii] = vIn[ii].y;
		vZ[ii] = vIn[ii].z;
	}
	return;
}

template<>
#ifdef __INTEL_COMPILER
#pragma omp declare simd
#endif
inline Vector3D::Vector3D(const Vector3D &v): Vector3D(v.x, v.y, v.z)
{}

/*! \brief Assignment operator
\param v Vector to be copied
\return The assigned value
*/
template<>
#ifdef __INTEL_COMPILER
#pragma omp declare simd
#endif
inline Vector3D& Vector3D::operator=<Vector3D>(const Vector3D& v)
{
	x = v.x;
	y = v.y;
	z = v.z;
	return *this;
}

#endif // Vector3D_HPP

