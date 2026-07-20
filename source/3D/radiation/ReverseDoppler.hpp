#ifndef REVERSE_DOPPLER_HPP
#define REVERSE_DOPPLER_HPP

#include <cmath>
#include "3D/elementary/Vector3D.hpp"
#include "CMMC/src/units/units.hpp"
#include "ReversePacket.hpp"
#include "ReversePolarizationMueller.hpp"

namespace ReverseDoppler
{

struct FrameState
{
    bool valid = false;
    double dopplerFactor = 1.0;
    double gamma = 1.0;
    Vector3D kForwardCo;
    double nuCo = 0.0;
    double frameWeightFactor = 1.0;
};

// Doppler factor: D = gamma * (1 - v . n_lab / c)
// Matches the forward DopplerShift convention from LorentzTransformation.
inline double computeDopplerFactor(Vector3D const &velocity,
                                   Vector3D const &kForwardLab)
{
    double v2 = ScalarProd(velocity, velocity);
    if (v2 < 1e-30)
        return 1.0;
    double gamma = 1.0 / std::sqrt(1.0 - units::inv_clight2 * v2);
    Vector3D nLab = kForwardLab;
    double nLen = abs(nLab);
    if (nLen > 1e-30)
        nLab = nLab * (1.0 / nLen);
    return gamma * (1.0 - ScalarProd(velocity, nLab) * units::inv_clight);
}

// Transform packet quantities to comoving frame for a given cell velocity.
// Full Lorentz transform matching the forward LorentzTransformation:
//   - nuCo = D * nuLab
//   - kForwardCo = aberrated direction (full relativistic)
//   - frameWeightFactor = 1/D (forward emission divides by D)
//   - Lab-frame opacity = sigma_co * D (caller must apply D scaling)
inline FrameState toComoving(ReverseAdjointPacket const &pkt,
                             Vector3D const &cellVelocity)
{
    FrameState fs;
    double v2 = ScalarProd(cellVelocity, cellVelocity);
    if (v2 < 1e-30)
    {
        fs.valid = true;
        fs.dopplerFactor = 1.0;
        fs.gamma = 1.0;
        fs.kForwardCo = pkt.kForwardLab;
        fs.nuCo = pkt.nuLab;
        fs.frameWeightFactor = 1.0;
        return fs;
    }

    fs.gamma = 1.0 / std::sqrt(1.0 - units::inv_clight2 * v2);
    if (!std::isfinite(fs.gamma) || fs.gamma > 1e6)
    {
        fs.valid = false;
        return fs;
    }

    fs.dopplerFactor = computeDopplerFactor(cellVelocity, pkt.kForwardLab);
    if (!std::isfinite(fs.dopplerFactor) || fs.dopplerFactor <= 0.0)
    {
        fs.valid = false;
        return fs;
    }

    fs.nuCo = fs.dopplerFactor * pkt.nuLab;

    // Full Lorentz aberration: u' = u + v * [(gamma-1)*dot(u,v)/v² - gamma]
    // where u = c * kForwardLab (photon velocity vector).
    // Matches forward LorentzTransformation.cpp line 19.
    Vector3D uLab = pkt.kForwardLab * units::clight;
    double udotv = ScalarProd(uLab, cellVelocity);
    Vector3D uCo = uLab + cellVelocity * ((fs.gamma - 1.0) * udotv / v2 - fs.gamma);
    double uCoMag = abs(uCo);
    if (uCoMag > 1e-30)
        fs.kForwardCo = uCo * (1.0 / uCoMag);
    else
        fs.kForwardCo = pkt.kForwardLab;

    // Frame weight: forward code divides emission weight by D, so reverse
    // source scoring must apply 1/D to match.
    fs.frameWeightFactor = 1.0 / fs.dopplerFactor;
    fs.valid = true;
    return fs;
}

// Transform a comoving-frame direction back to lab frame (inverse aberration).
// Uses the inverse Lorentz boost: lab velocity is -v in the comoving frame.
inline Vector3D comovingDirToLab(Vector3D const &kCo, Vector3D const &cellVelocity,
                                 double gamma)
{
    double v2 = ScalarProd(cellVelocity, cellVelocity);
    if (v2 < 1e-30)
        return kCo;

    // Inverse boost: replace v with -v in the aberration formula.
    // u_lab = u_co + (-v) * [(gamma-1)*dot(u_co,-v)/v² - gamma]
    //       = u_co - v * [(gamma-1)*(-dot(u_co,v))/v² - gamma]
    //       = u_co + v * [(gamma-1)*dot(u_co,v)/v² + gamma]
    Vector3D uCo = kCo * units::clight;
    double udotv = ScalarProd(uCo, cellVelocity);
    Vector3D uLab = uCo + cellVelocity * ((gamma - 1.0) * udotv / v2 + gamma);
    double uLabMag = abs(uLab);
    if (uLabMag > 1e-30)
        return uLab * (1.0 / uLabMag);
    return kCo;
}

// Compute lab frequency from comoving frequency (inverse Doppler)
inline double toLabFrequency(double nuCo, double dopplerFactor)
{
    if (dopplerFactor > 1e-30)
        return nuCo / dopplerFactor;
    return nuCo;
}

// Lorentz-transform a polarization basis vector from lab screen to comoving screen.
// Uses the electric-field boost: E_co = gamma*(E + beta x B) - (gamma²/(gamma+1)) beta (beta·E)
// where B = k x E for a transverse wave.
inline Vector3D labBasisToComovingScreen(
    Vector3D const &basisLab,
    Vector3D const &kLab,
    Vector3D const &kCo,
    Vector3D const &cellVelocity,
    double gamma)
{
    double v2 = ScalarProd(cellVelocity, cellVelocity);
    if (v2 < 1e-30 || gamma <= 1.0 + 1e-14)
        return ReverseMueller::projectBasisToDirection(basisLab, kCo);

    Vector3D beta = cellVelocity * units::inv_clight;
    Vector3D B_lab = CrossProduct(kLab, basisLab);
    Vector3D betaCrossB = CrossProduct(beta, B_lab);
    double betaDotE = ScalarProd(beta, basisLab);
    double coeff = gamma * gamma / (gamma + 1.0);

    Vector3D E_co = basisLab * gamma + betaCrossB * gamma
                  - beta * (coeff * betaDotE);

    return ReverseMueller::projectBasisToDirection(E_co, kCo);
}

// Lorentz-transform a polarization basis vector from comoving screen to lab screen.
// Uses inverse boost (replace beta with -beta):
// E_lab = gamma*(E_co - beta x B_co) - (gamma²/(gamma+1)) beta (beta·E_co)
inline Vector3D comovingBasisToLabScreen(
    Vector3D const &basisCo,
    Vector3D const &kCo,
    Vector3D const &kLab,
    Vector3D const &cellVelocity,
    double gamma)
{
    double v2 = ScalarProd(cellVelocity, cellVelocity);
    if (v2 < 1e-30 || gamma <= 1.0 + 1e-14)
        return ReverseMueller::projectBasisToDirection(basisCo, kLab);

    Vector3D beta = cellVelocity * units::inv_clight;
    Vector3D B_co = CrossProduct(kCo, basisCo);
    Vector3D betaCrossB = CrossProduct(beta, B_co);
    double betaDotE = ScalarProd(beta, basisCo);
    double coeff = gamma * gamma / (gamma + 1.0);

    Vector3D E_lab = basisCo * gamma - betaCrossB * gamma
                   - beta * (coeff * betaDotE);

    return ReverseMueller::projectBasisToDirection(E_lab, kLab);
}

} // namespace ReverseDoppler

#endif // REVERSE_DOPPLER_HPP
