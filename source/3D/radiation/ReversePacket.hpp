#ifndef REVERSE_PACKET_HPP
#define REVERSE_PACKET_HPP

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include "3D/elementary/Vector3D.hpp"

struct MuellerResponse3
{
    std::array<std::array<double, 3>, 3> m;

    MuellerResponse3()
    {
        m[0] = {1.0, 0.0, 0.0};
        m[1] = {0.0, 1.0, 0.0};
        m[2] = {0.0, 0.0, 1.0};
    }

    static MuellerResponse3 identity() { return MuellerResponse3(); }

    static MuellerResponse3 zero()
    {
        MuellerResponse3 r;
        r.m[0] = {0.0, 0.0, 0.0};
        r.m[1] = {0.0, 0.0, 0.0};
        r.m[2] = {0.0, 0.0, 0.0};
        return r;
    }

    MuellerResponse3 operator*(MuellerResponse3 const &rhs) const
    {
        MuellerResponse3 result = zero();
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                for (int k = 0; k < 3; ++k)
                    result.m[i][j] += m[i][k] * rhs.m[k][j];
        return result;
    }

    MuellerResponse3 &operator*=(MuellerResponse3 const &rhs)
    {
        *this = *this * rhs;
        return *this;
    }

    void dampPolarizationRows(double factor)
    {
        for (int j = 0; j < 3; ++j)
        {
            m[1][j] *= factor;
            m[2][j] *= factor;
        }
    }

    void resetToUnpolarizedSource()
    {
        m[0] = {1.0, 0.0, 0.0};
        m[1] = {0.0, 0.0, 0.0};
        m[2] = {0.0, 0.0, 0.0};
    }

    double frobeniusNorm() const
    {
        double sum = 0.0;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                sum += m[i][j] * m[i][j];
        return std::sqrt(sum);
    }

    void applyToSourceI(double srcI, double &outI, double &outQ, double &outU) const
    {
        outI = m[0][0] * srcI;
        outQ = m[1][0] * srcI;
        outU = m[2][0] * srcI;
    }
};

struct ReverseAdjointPacket
{
    Vector3D xLab;
    size_t cellIndex = std::numeric_limits<size_t>::max();
    Vector3D kForwardLab;
    Vector3D kReverseLab;

    double nuLab = 0.0;
    double nuCo = 0.0;
    size_t observedGroup = 0;
    size_t currentCoGroup = 0;

    double scalarWeight = 1.0;
    MuellerResponse3 M_obs_from_src;
    double logWeight = 0.0;

    Vector3D sourceBasisLab;
    Vector3D observerSkyE1;
    bool basisInitialized = false;

    double tCoAccumulated = 0.0;
    double tLabAccumulated = 0.0;
    double pathLabAccumulated = 0.0;

    size_t observerIndex = 0;
    size_t scatterCountExplicit = 0;
    size_t scatterCountSynthetic = 0;
    size_t ddmcStepCount = 0;
    size_t ddmcLeakCount = 0;
    size_t resetCount = 0;
    size_t faceCrossingCount = 0;
    size_t eventCount = 0;
    bool usedDDMC = false;
    bool alive = true;
};

#endif // REVERSE_PACKET_HPP
