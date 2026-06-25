#include "ReversePolarizationMueller.hpp"

namespace ReverseMueller
{

MuellerResponse3 basisRotationBetween(Vector3D const &kDir,
                                      Vector3D const &oldBasis,
                                      Vector3D const &newBasis)
{
    Vector3D k = safeNormalize(kDir, Vector3D(0.0, 0.0, 1.0));
    Vector3D eOld = projectBasisToDirection(oldBasis, k);
    Vector3D eNew = projectBasisToDirection(newBasis, k);

    double c = std::clamp(ScalarProd(eOld, eNew), -1.0, 1.0);
    double s = ScalarProd(k, CrossProduct(eOld, eNew));
    return basisRotation(c, s);
}

MuellerResponse3 reverseThomsonUpdate(
    Vector3D const &kIn, Vector3D const &kOut,
    Vector3D const &basisIn, Vector3D const &basisOut)
{
    Vector3D planeNormal = CrossProduct(kIn, kOut);
    double planeNorm = abs(planeNormal);

    if (planeNorm <= 1e-10)
    {
        double mu = std::clamp(ScalarProd(
            safeNormalize(kIn, Vector3D(0, 0, 1)),
            safeNormalize(kOut, Vector3D(0, 0, 1))), -1.0, 1.0);
        MuellerResponse3 T = thomsonMatrix(mu);
        double norm = 1.0 + mu * mu;
        if (norm > POL_EPS)
        {
            double invNorm = 1.0 / norm;
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j)
                    T.m[i][j] *= invNorm;
        }
        return T;
    }

    planeNormal = planeNormal * (1.0 / planeNorm);
    Vector3D scatBasisIn = planeNormal;
    Vector3D scatBasisOut = planeNormal;

    double mu = std::clamp(ScalarProd(
        safeNormalize(kIn, Vector3D(0, 0, 1)),
        safeNormalize(kOut, Vector3D(0, 0, 1))), -1.0, 1.0);

    MuellerResponse3 Bin = basisRotationBetween(kIn, basisIn, scatBasisIn);
    MuellerResponse3 T = thomsonMatrix(mu);
    MuellerResponse3 Bout = basisRotationBetween(kOut, scatBasisOut, basisOut);

    double norm = 1.0 + mu * mu;
    if (norm > POL_EPS)
    {
        double invNorm = 1.0 / norm;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                T.m[i][j] *= invNorm;
    }

    return Bout * T * Bin;
}

} // namespace ReverseMueller
