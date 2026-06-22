#ifndef REVERSE_OBSERVER_TALLIES_HPP
#define REVERSE_OBSERVER_TALLIES_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include "3D/elementary/Vector3D.hpp"
#include "ReversePacket.hpp"

#ifdef RICH_MPI
#include <mpi.h>
#endif

class HDF5Writer;

struct ReverseTallyDiagnostics
{
    uint64_t packetsLaunched = 0;
    uint64_t packetsCutoff = 0;
    uint64_t packetsMaxEvents = 0;
    uint64_t packetsEscaped = 0;
    uint64_t packetsNonFinite = 0;
    uint64_t timeCensusCount = 0;
    uint64_t ordinarySteps = 0;
    uint64_t ddmcSteps = 0;
    uint64_t ddmcFallbacks = 0;
    uint64_t ddmcTimeLimited = 0;
    uint64_t nonFiniteScoreCount = 0;
    uint64_t negativeICount = 0;
    uint64_t thermalSamplerFallbackCount = 0;
    uint64_t thermalSamplerFailureCount = 0;
    uint64_t thermalBoundaryFallbackCount = 0;
    double maxMuellerNorm = 0.0;
    double maxTLabAccumulated = 0.0;
    std::vector<uint64_t> sourceGroupScoreCount;
};

class ReverseObserverTallies
{
public:
    ReverseObserverTallies(size_t numObservers, size_t numGroups,
                           size_t maxScatterOrders = 0)
        : nObs_(numObservers), nGroups_(numGroups),
          maxScatterOrders_(maxScatterOrders)
    {
        obsI_.assign(nObs_, 0.0);
        obsQ_.assign(nObs_, 0.0);
        obsU_.assign(nObs_, 0.0);
        diag_.sourceGroupScoreCount.assign(nGroups_, 0);
        obsI2_.assign(nObs_, 0.0);
        obsQ2_.assign(nObs_, 0.0);
        obsU2_.assign(nObs_, 0.0);

        grpI_.assign(nObs_, std::vector<double>(nGroups_, 0.0));
        grpQ_.assign(nObs_, std::vector<double>(nGroups_, 0.0));
        grpU_.assign(nObs_, std::vector<double>(nGroups_, 0.0));
        grpI2_.assign(nObs_, std::vector<double>(nGroups_, 0.0));
        grpQ2_.assign(nObs_, std::vector<double>(nGroups_, 0.0));
        grpU2_.assign(nObs_, std::vector<double>(nGroups_, 0.0));

        if (maxScatterOrders_ > 0)
        {
            scatI_.assign(nObs_, std::vector<double>(maxScatterOrders_, 0.0));
            scatQ_.assign(nObs_, std::vector<double>(maxScatterOrders_, 0.0));
            scatU_.assign(nObs_, std::vector<double>(maxScatterOrders_, 0.0));
        }

        ddmcAbsIContrib_.assign(nObs_, 0.0);
        maxMuellerNorm_.assign(nObs_, 0.0);
        negICount_.assign(nObs_, 0);
        nonFiniteCount_.assign(nObs_, 0);

        collapsedI_.assign(nObs_, std::vector<double>(nGroups_, 0.0));
        collapsedQ_.assign(nObs_, std::vector<double>(nGroups_, 0.0));
        collapsedU_.assign(nObs_, std::vector<double>(nGroups_, 0.0));

        pktI_.assign(nObs_, 0.0);
        pktQ_.assign(nObs_, 0.0);
        pktU_.assign(nObs_, 0.0);
        pktGrpI_.assign(nObs_, std::vector<double>(nGroups_, 0.0));
        pktGrpQ_.assign(nObs_, std::vector<double>(nGroups_, 0.0));
        pktGrpU_.assign(nObs_, std::vector<double>(nGroups_, 0.0));
        pktCollapsedI_.assign(nObs_, std::vector<double>(nGroups_, 0.0));
        pktCollapsedQ_.assign(nObs_, std::vector<double>(nGroups_, 0.0));
        pktCollapsedU_.assign(nObs_, std::vector<double>(nGroups_, 0.0));
        totalGrpI2_.assign(nObs_, std::vector<double>(nGroups_, 0.0));
        totalGrpQ2_.assign(nObs_, std::vector<double>(nGroups_, 0.0));
        totalGrpU2_.assign(nObs_, std::vector<double>(nGroups_, 0.0));
        packetCount_.assign(nObs_, 0);
    }

    void beginPacket()
    {
        std::fill(pktI_.begin(), pktI_.end(), 0.0);
        std::fill(pktQ_.begin(), pktQ_.end(), 0.0);
        std::fill(pktU_.begin(), pktU_.end(), 0.0);
        for (size_t p = 0; p < nObs_; ++p)
        {
            std::fill(pktGrpI_[p].begin(), pktGrpI_[p].end(), 0.0);
            std::fill(pktGrpQ_[p].begin(), pktGrpQ_[p].end(), 0.0);
            std::fill(pktGrpU_[p].begin(), pktGrpU_[p].end(), 0.0);
            std::fill(pktCollapsedI_[p].begin(), pktCollapsedI_[p].end(), 0.0);
            std::fill(pktCollapsedQ_[p].begin(), pktCollapsedQ_[p].end(), 0.0);
            std::fill(pktCollapsedU_[p].begin(), pktCollapsedU_[p].end(), 0.0);
        }
    }

    void endPacket()
    {
        for (size_t p = 0; p < nObs_; ++p)
        {
            obsI2_[p] += pktI_[p] * pktI_[p];
            obsQ2_[p] += pktQ_[p] * pktQ_[p];
            obsU2_[p] += pktU_[p] * pktU_[p];
            if (pktI_[p] != 0.0 || pktQ_[p] != 0.0 || pktU_[p] != 0.0)
                ++packetCount_[p];
            for (size_t g = 0; g < nGroups_; ++g)
            {
                grpI2_[p][g] += pktGrpI_[p][g] * pktGrpI_[p][g];
                grpQ2_[p][g] += pktGrpQ_[p][g] * pktGrpQ_[p][g];
                grpU2_[p][g] += pktGrpU_[p][g] * pktGrpU_[p][g];
                double pktTotalI = pktGrpI_[p][g] + pktCollapsedI_[p][g];
                double pktTotalQ = pktGrpQ_[p][g] + pktCollapsedQ_[p][g];
                double pktTotalU = pktGrpU_[p][g] + pktCollapsedU_[p][g];
                totalGrpI2_[p][g] += pktTotalI * pktTotalI;
                totalGrpQ2_[p][g] += pktTotalQ * pktTotalQ;
                totalGrpU2_[p][g] += pktTotalU * pktTotalU;
            }
        }
    }

    // Read accessors for testing
    double getObsI(size_t obs) const { return (obs < nObs_) ? obsI_[obs] : 0.0; }
    double getObsQ(size_t obs) const { return (obs < nObs_) ? obsQ_[obs] : 0.0; }
    double getObsU(size_t obs) const { return (obs < nObs_) ? obsU_[obs] : 0.0; }
    double getObsI2(size_t obs) const { return (obs < nObs_) ? obsI2_[obs] : 0.0; }
    uint64_t getPacketCount(size_t obs) const { return (obs < nObs_) ? packetCount_[obs] : 0; }
    double getGrpI(size_t obs, size_t grp) const
    {
        return (obs < nObs_ && grp < nGroups_) ? grpI_[obs][grp] : 0.0;
    }

    // attenuationOpacity = f*sigmaA, the continuous Fleck decay.
    void scoreOrdinarySegment(ReverseAdjointPacket const &pkt,
                              size_t /*cellIndex*/, double segmentLength,
                              double cellVolume, double sourceLuminosity,
                              double patchAreaOverN, double frameWeight,
                              double attenuationOpacity = 0.0)
    {
        // Attenuation-integrated track-length: integral of exp(-kappa*s) ds over [0,ds]
        double effectiveLength = segmentLength;
        double tau = attenuationOpacity * segmentLength;
        if (tau > 1e-6)
            effectiveLength = (1.0 - std::exp(-tau)) / attenuationOpacity;

        double scoreFactor = pkt.scalarWeight * patchAreaOverN *
                             (effectiveLength / cellVolume) * frameWeight;

        double dI, dQ, dU;
        pkt.M_obs_from_src.applyToSourceI(sourceLuminosity * scoreFactor, dI, dQ, dU);

        double mNorm = pkt.M_obs_from_src.frobeniusNorm();
        if (pkt.observerIndex < maxMuellerNorm_.size() && mNorm > maxMuellerNorm_[pkt.observerIndex])
            maxMuellerNorm_[pkt.observerIndex] = mNorm;
        diag_.maxMuellerNorm = std::max(diag_.maxMuellerNorm, mNorm);

        accumulateScore(pkt.observerIndex, pkt.observedGroup, dI, dQ, dU,
                        pkt.scatterCountExplicit + pkt.scatterCountSynthetic);
    }

    // absDecayRate = c * f * sigmaA (continuous absorption rate).
    // w0 = pre-decay weight (before exp(-absDecayRate*dtCo) was applied).
    void scoreDDMCResidence(ReverseAdjointPacket const &pkt,
                            size_t /*cellIndex*/, double dtCo,
                            double cellVolume, double sourceLuminosity,
                            double patchAreaOverN, double clight,
                            double frameWeight,
                            double absDecayRate = 0.0,
                            double w0 = -1.0)
    {
        double weight = (w0 >= 0.0) ? w0 : pkt.scalarWeight;

        // Attenuation-integrated residence: integral of exp(-lambda*t) * c * dt
        double effectivePath = clight * dtCo;
        double tau = absDecayRate * dtCo;
        if (tau > 1e-6 && absDecayRate > 0.0)
            effectivePath = clight * (1.0 - std::exp(-tau)) / absDecayRate;

        double scoreFactor = weight * patchAreaOverN *
                             (effectivePath / cellVolume) * frameWeight;

        double dI, dQ, dU;
        pkt.M_obs_from_src.applyToSourceI(sourceLuminosity * scoreFactor, dI, dQ, dU);

        double mNorm = pkt.M_obs_from_src.frobeniusNorm();
        if (pkt.observerIndex < maxMuellerNorm_.size() && mNorm > maxMuellerNorm_[pkt.observerIndex])
            maxMuellerNorm_[pkt.observerIndex] = mNorm;
        diag_.maxMuellerNorm = std::max(diag_.maxMuellerNorm, mNorm);

        if (pkt.observerIndex < ddmcAbsIContrib_.size())
            ddmcAbsIContrib_[pkt.observerIndex] += std::abs(dI);

        accumulateScore(pkt.observerIndex, pkt.observedGroup, dI, dQ, dU,
                        pkt.scatterCountExplicit + pkt.scatterCountSynthetic);
        ++diag_.ddmcSteps;
    }

    void scoreDDMCResidenceCollapsed(ReverseAdjointPacket const &pkt,
                                     size_t /*cellIndex*/, double dtCo,
                                     double cellVolume, double sourceLuminosity,
                                     double patchAreaOverN, double clight,
                                     double frameWeight,
                                     double absDecayRate = 0.0,
                                     double w0 = -1.0)
    {
        double weight = (w0 >= 0.0) ? w0 : pkt.scalarWeight;
        double effectivePath = clight * dtCo;
        double tau = absDecayRate * dtCo;
        if (tau > 1e-6 && absDecayRate > 0.0)
            effectivePath = clight * (1.0 - std::exp(-tau)) / absDecayRate;

        double scoreFactor = weight * patchAreaOverN *
                             (effectivePath / cellVolume) * frameWeight;

        double dI, dQ, dU;
        pkt.M_obs_from_src.applyToSourceI(sourceLuminosity * scoreFactor, dI, dQ, dU);

        size_t obs = pkt.observerIndex;
        size_t grp = pkt.observedGroup;
        if (obs < nObs_ && grp < nGroups_)
        {
            collapsedI_[obs][grp] += dI;
            collapsedQ_[obs][grp] += dQ;
            collapsedU_[obs][grp] += dU;
            pktCollapsedI_[obs][grp] += dI;
            pktCollapsedQ_[obs][grp] += dQ;
            pktCollapsedU_[obs][grp] += dU;
        }
        obsI_[obs] += dI;
        obsQ_[obs] += dQ;
        obsU_[obs] += dU;
        pktI_[obs] += dI;
        pktQ_[obs] += dQ;
        pktU_[obs] += dU;
        ++diag_.ddmcSteps;
    }

    std::vector<std::vector<double>> const &collapsedI() const { return collapsedI_; }
    std::vector<std::vector<double>> const &collapsedQ() const { return collapsedQ_; }
    std::vector<std::vector<double>> const &collapsedU() const { return collapsedU_; }

    void mpiReduceToRank0()
    {
#ifdef RICH_MPI
        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

        reduceVector(obsI_);
        reduceVector(obsQ_);
        reduceVector(obsU_);
        reduceVector(obsI2_);
        reduceVector(obsQ2_);
        reduceVector(obsU2_);

        for (size_t p = 0; p < nObs_; ++p)
        {
            reduceVector(grpI_[p]);
            reduceVector(grpQ_[p]);
            reduceVector(grpU_[p]);
            reduceVector(grpI2_[p]);
            reduceVector(grpQ2_[p]);
            reduceVector(grpU2_[p]);
            reduceVector(totalGrpI2_[p]);
            reduceVector(totalGrpQ2_[p]);
            reduceVector(totalGrpU2_[p]);
        }

        for (size_t p = 0; p < nObs_ && maxScatterOrders_ > 0; ++p)
        {
            reduceVector(scatI_[p]);
            reduceVector(scatQ_[p]);
            reduceVector(scatU_[p]);
        }

        reduceVector(ddmcAbsIContrib_);

        for (size_t p = 0; p < nObs_; ++p)
        {
            reduceVector(collapsedI_[p]);
            reduceVector(collapsedQ_[p]);
            reduceVector(collapsedU_[p]);
        }

        {
            std::vector<uint64_t> tmpPkt(nObs_, 0);
            MPI_Reduce(packetCount_.data(), tmpPkt.data(),
                       static_cast<int>(nObs_), MPI_UINT64_T, MPI_SUM,
                       0, MPI_COMM_WORLD);
            if (rank == 0) packetCount_ = tmpPkt;
        }

        {
            std::vector<double> tmp(nObs_);
            MPI_Reduce(maxMuellerNorm_.data(), tmp.data(),
                       static_cast<int>(nObs_), MPI_DOUBLE, MPI_MAX,
                       0, MPI_COMM_WORLD);
            if (rank == 0) maxMuellerNorm_ = tmp;
        }

        reduceDiagnostics();
#endif
    }

    void writeHDF5(HDF5Writer &writer, std::string const &prefix) const;

    // Tallies store luminosity directly [erg/s]. Do not divide by sourceDt.
    void finalizeDerived(std::vector<double> &obsLumI,
                         std::vector<double> &obsQ, std::vector<double> &obsU,
                         std::vector<double> &obsPolDeg,
                         std::vector<double> &obsPolAngle) const
    {
        obsLumI.resize(nObs_);
        obsQ.resize(nObs_);
        obsU.resize(nObs_);
        obsPolDeg.resize(nObs_);
        obsPolAngle.resize(nObs_);

        for (size_t p = 0; p < nObs_; ++p)
        {
            double I = obsI_[p];
            double Q = obsQ_[p];
            double U = obsU_[p];
            obsLumI[p] = I;
            if (std::abs(I) > 1e-300)
            {
                obsQ[p] = Q / I;
                obsU[p] = U / I;
            }
            else
            {
                obsQ[p] = 0.0;
                obsU[p] = 0.0;
            }
            obsPolDeg[p] = std::sqrt(obsQ[p] * obsQ[p] + obsU[p] * obsU[p]);
            obsPolAngle[p] = 0.5 * std::atan2(obsU[p], obsQ[p]);
        }
    }

    ReverseTallyDiagnostics const &diagnostics() const { return diag_; }
    ReverseTallyDiagnostics &diagnostics() { return diag_; }

    std::vector<double> const &observerI() const { return obsI_; }
    std::vector<double> const &observerQ() const { return obsQ_; }
    std::vector<double> const &observerU() const { return obsU_; }
    std::vector<double> const &observerI2() const { return obsI2_; }
    std::vector<double> const &observerQ2() const { return obsQ2_; }
    std::vector<double> const &observerU2() const { return obsU2_; }
    std::vector<std::vector<double>> const &groupI() const { return grpI_; }
    std::vector<std::vector<double>> const &groupQ() const { return grpQ_; }
    std::vector<std::vector<double>> const &groupU() const { return grpU_; }
    std::vector<std::vector<double>> const &groupI2() const { return grpI2_; }
    std::vector<std::vector<double>> const &groupQ2() const { return grpQ2_; }
    std::vector<std::vector<double>> const &groupU2() const { return grpU2_; }
    std::vector<std::vector<double>> const &totalGroupI2() const { return totalGrpI2_; }
    std::vector<std::vector<double>> const &totalGroupQ2() const { return totalGrpQ2_; }
    std::vector<std::vector<double>> const &totalGroupU2() const { return totalGrpU2_; }
    std::vector<uint64_t> const &packetCounts() const { return packetCount_; }
    size_t numObservers() const { return nObs_; }
    size_t numGroups() const { return nGroups_; }

private:
    size_t nObs_;
    size_t nGroups_;
    size_t maxScatterOrders_;
    ReverseTallyDiagnostics diag_;

    std::vector<double> obsI_, obsQ_, obsU_;
    std::vector<double> obsI2_, obsQ2_, obsU2_;

    // Group tallies are resolved-only. Collapsed-PGRW contributions are stored
    // separately in collapsedI_/collapsedQ_/collapsedU_. HDF5 total group output
    // is computed as resolved + collapsed.
    std::vector<std::vector<double>> grpI_, grpQ_, grpU_;

    // Per-packet second moments for resolved-only group contributions.
    std::vector<std::vector<double>> grpI2_, grpQ2_, grpU2_;
    std::vector<std::vector<double>> scatI_, scatQ_, scatU_;

    std::vector<double> ddmcAbsIContrib_;
    std::vector<double> maxMuellerNorm_;
    std::vector<uint64_t> negICount_;
    std::vector<uint64_t> nonFiniteCount_;

    std::vector<std::vector<double>> collapsedI_, collapsedQ_, collapsedU_;

    // Per-packet accumulators for correct variance estimation.
    std::vector<double> pktI_, pktQ_, pktU_;
    std::vector<std::vector<double>> pktGrpI_, pktGrpQ_, pktGrpU_;
    std::vector<std::vector<double>> pktCollapsedI_, pktCollapsedQ_, pktCollapsedU_;

    // Per-packet second moments for total (resolved + collapsed) group contributions.
    std::vector<std::vector<double>> totalGrpI2_, totalGrpQ2_, totalGrpU2_;

    // Packet count per observer for standard error computation.
    std::vector<uint64_t> packetCount_;

    void accumulateScore(size_t obs, size_t group,
                         double dI, double dQ, double dU,
                         size_t scatterOrder)
    {
        if (!std::isfinite(dI) || !std::isfinite(dQ) || !std::isfinite(dU))
        {
            ++nonFiniteCount_[obs];
            ++diag_.nonFiniteScoreCount;
            return;
        }
        if (dI < 0.0)
        {
            ++negICount_[obs];
            ++diag_.negativeICount;
        }

        obsI_[obs] += dI;
        obsQ_[obs] += dQ;
        obsU_[obs] += dU;
        pktI_[obs] += dI;
        pktQ_[obs] += dQ;
        pktU_[obs] += dU;

        if (group < nGroups_)
        {
            grpI_[obs][group] += dI;
            grpQ_[obs][group] += dQ;
            grpU_[obs][group] += dU;
            pktGrpI_[obs][group] += dI;
            pktGrpQ_[obs][group] += dQ;
            pktGrpU_[obs][group] += dU;
        }

        if (maxScatterOrders_ > 0 && scatterOrder < maxScatterOrders_)
        {
            scatI_[obs][scatterOrder] += dI;
            scatQ_[obs][scatterOrder] += dQ;
            scatU_[obs][scatterOrder] += dU;
        }
    }

#ifdef RICH_MPI
    static void reduceVector(std::vector<double> &v)
    {
        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        std::vector<double> tmp(v.size(), 0.0);
        MPI_Reduce(v.data(), tmp.data(), static_cast<int>(v.size()),
                   MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        if (rank == 0) v = tmp;
    }

    void reduceDiagnostics()
    {
        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

        auto reduceU64 = [&](uint64_t &val)
        {
            uint64_t tmp = 0;
            MPI_Reduce(&val, &tmp, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
            if (rank == 0) val = tmp;
        };
        auto reduceMaxD = [&](double &val)
        {
            double tmp = 0.0;
            MPI_Reduce(&val, &tmp, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            if (rank == 0) val = tmp;
        };

        reduceU64(diag_.packetsLaunched);
        reduceU64(diag_.packetsCutoff);
        reduceU64(diag_.packetsMaxEvents);
        reduceU64(diag_.packetsEscaped);
        reduceU64(diag_.packetsNonFinite);
        reduceU64(diag_.timeCensusCount);
        reduceU64(diag_.ordinarySteps);
        reduceU64(diag_.ddmcSteps);
        reduceU64(diag_.ddmcFallbacks);
        reduceU64(diag_.ddmcTimeLimited);
        reduceU64(diag_.nonFiniteScoreCount);
        reduceU64(diag_.negativeICount);
        reduceU64(diag_.thermalSamplerFallbackCount);
        reduceU64(diag_.thermalSamplerFailureCount);
        reduceU64(diag_.thermalBoundaryFallbackCount);
        for (auto &v : diag_.sourceGroupScoreCount)
            reduceU64(v);
        reduceMaxD(diag_.maxMuellerNorm);
        reduceMaxD(diag_.maxTLabAccumulated);
    }
#endif
};

#endif // REVERSE_OBSERVER_TALLIES_HPP
