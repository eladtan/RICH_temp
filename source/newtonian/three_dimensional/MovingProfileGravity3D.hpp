/*! \file MovingProfileGravity3D.hpp
  \brief Fixed spherical gravity from a tabulated enclosed-mass profile M(r), centred on a prescribed path.
*/
#ifndef MOVING_PROFILE_GRAVITY3D_HPP
#define MOVING_PROFILE_GRAVITY3D_HPP 1

#include <vector>
#include "ConservativeForce3D.hpp"

/*! \brief Acceleration -G M(<r) / r^2 towards Centre(t), with M(<r) from a table and M(<r) = M(<RTable.back())
  outside it. The centre is at rest at Centre0 until StartTime and then moves with constant Velocity.
  The acceleration is evaluated at the cell centre of mass. ZeroZ removes the z component (wedge runs).
*/
class MovingProfileGravity3D : public Acceleration3D {
public:
    /*!
      \param RTable Increasing radii, RTable[0] > 0
      \param MTable Enclosed mass at each radius
      \param G Gravitational constant in code units
      \param ZeroZ Drop the z component of the acceleration
    */
    MovingProfileGravity3D(std::vector<double> const& RTable, std::vector<double> const& MTable, double G, bool ZeroZ);

    void operator()(const Tessellation3D& Tess, const vector<ComputationalCell3D>& Cells, const vector<Conserved3D>& Fluxes, const double Time,
        vector<Vector3D>& Acc) const override;

    //! \brief Set the path: at rest at Centre0 until StartTime, then moving with Velocity
    void SetPath(Vector3D const& Centre0, Vector3D const& Velocity, double StartTime);

    Vector3D GetCentre(double Time) const;

    Vector3D GetVelocity(double Time) const;

    //! \brief Enclosed mass at distance R from the centre
    double EnclosedMass(double R) const;

    //! \brief Gravitational potential at distance R from the centre (zero at infinity)
    double Potential(double R) const;

private:
    std::vector<double> RTable_, MTable_, PhiTable_;
    double G_;
    bool ZeroZ_;
    Vector3D Centre0_, Velocity_;
    double StartTime_;
};

#endif // MOVING_PROFILE_GRAVITY3D_HPP
