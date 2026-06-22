#include "ReverseObserverTallies.hpp"
#include "ReverseEstimatorConfig.hpp"
#include "utils/hdf5/HDF5Writer.hpp"
#include <cmath>
#include <string>

namespace
{
std::vector<double> addSameSizeVectors(
    std::vector<double> const &a,
    std::vector<double> const &b)
{
    std::vector<double> out(a.size(), 0.0);
    for (size_t i = 0; i < out.size(); ++i)
        out[i] = a[i] + ((i < b.size()) ? b[i] : 0.0);
    return out;
}
} // anonymous namespace

void ReverseObserverTallies::writeHDF5(HDF5Writer &writer,
                                       std::string const &prefix) const
{
    writer.WriteElement(prefix + "/observer_I", obsI_);
    writer.WriteElement(prefix + "/observer_Q", obsQ_);
    writer.WriteElement(prefix + "/observer_U", obsU_);
    writer.WriteElement(prefix + "/observer_I2", obsI2_);
    writer.WriteElement(prefix + "/observer_Q2", obsQ2_);
    writer.WriteElement(prefix + "/observer_U2", obsU2_);

    for (size_t p = 0; p < nObs_; ++p)
    {
        std::string gPrefix = prefix + "/group_" + std::to_string(p);
        writer.WriteElement(gPrefix + "/I", grpI_[p]);
        writer.WriteElement(gPrefix + "/Q", grpQ_[p]);
        writer.WriteElement(gPrefix + "/U", grpU_[p]);
        writer.WriteElement(gPrefix + "/I_resolved", grpI_[p]);
        writer.WriteElement(gPrefix + "/Q_resolved", grpQ_[p]);
        writer.WriteElement(gPrefix + "/U_resolved", grpU_[p]);
        auto totalI = addSameSizeVectors(grpI_[p], collapsedI_[p]);
        auto totalQ = addSameSizeVectors(grpQ_[p], collapsedQ_[p]);
        auto totalU = addSameSizeVectors(grpU_[p], collapsedU_[p]);
        writer.WriteElement(gPrefix + "/I_total", totalI);
        writer.WriteElement(gPrefix + "/Q_total", totalQ);
        writer.WriteElement(gPrefix + "/U_total", totalU);
        writer.WriteElement(gPrefix + "/I_collapsed_pgrw", collapsedI_[p]);
        writer.WriteElement(gPrefix + "/Q_collapsed_pgrw", collapsedQ_[p]);
        writer.WriteElement(gPrefix + "/U_collapsed_pgrw", collapsedU_[p]);
        writer.WriteElement(gPrefix + "/metadata/I_semantics",
            std::string(ReverseOutputStrings::ResolvedOnlyLegacyAlias));
    }
    writer.WriteElement(prefix + "/metadata/group_I_semantics",
        std::string(ReverseOutputStrings::GroupISemantics));

    if (maxScatterOrders_ > 0)
    {
        for (size_t p = 0; p < nObs_; ++p)
        {
            std::string sPrefix = prefix + "/scatter_order_" + std::to_string(p);
            writer.WriteElement(sPrefix + "/I", scatI_[p]);
            writer.WriteElement(sPrefix + "/Q", scatQ_[p]);
            writer.WriteElement(sPrefix + "/U", scatU_[p]);
        }
    }

    writer.WriteElement(prefix + "/ddmc_abs_I_contribution_by_observer", ddmcAbsIContrib_);
    writer.WriteElement(prefix + "/max_mueller_norm_by_observer", maxMuellerNorm_);

    // Resolved/collapsed/total split
    bool hasCollapsed = false;
    for (size_t p = 0; p < nObs_ && !hasCollapsed; ++p)
        for (size_t g = 0; g < nGroups_ && !hasCollapsed; ++g)
            if (collapsedI_[p][g] != 0.0 || collapsedQ_[p][g] != 0.0 || collapsedU_[p][g] != 0.0)
                hasCollapsed = true;

    std::vector<double> resolvedObsI(nObs_), collapsedObsI(nObs_);
    for (size_t p = 0; p < nObs_; ++p)
    {
        double collSum = 0.0;
        for (size_t g = 0; g < nGroups_; ++g)
            collSum += collapsedI_[p][g];
        collapsedObsI[p] = collSum;
        resolvedObsI[p] = obsI_[p] - collSum;
    }
    writer.WriteElement(prefix + "/total_observer_I", obsI_);
    writer.WriteElement(prefix + "/resolved_observer_I", resolvedObsI);
    writer.WriteElement(prefix + "/collapsed_observer_I", collapsedObsI);

    if (hasCollapsed)
    {
        for (size_t p = 0; p < nObs_; ++p)
        {
            std::string cPrefix = prefix + "/collapsed_pgrw_" + std::to_string(p);
            writer.WriteElement(cPrefix + "/I", collapsedI_[p]);
            writer.WriteElement(cPrefix + "/Q", collapsedQ_[p]);
            writer.WriteElement(cPrefix + "/U", collapsedU_[p]);
        }

        // grpI_ is resolved-only; total = resolved + collapsed
        for (size_t p = 0; p < nObs_; ++p)
        {
            auto totalGrpI = addSameSizeVectors(grpI_[p], collapsedI_[p]);
            auto totalGrpQ = addSameSizeVectors(grpQ_[p], collapsedQ_[p]);
            auto totalGrpU = addSameSizeVectors(grpU_[p], collapsedU_[p]);
            std::string rPrefix = prefix + "/resolved_group_" + std::to_string(p);
            writer.WriteElement(rPrefix + "/I", grpI_[p]);
            writer.WriteElement(rPrefix + "/Q", grpQ_[p]);
            writer.WriteElement(rPrefix + "/U", grpU_[p]);
            std::string tPrefix = prefix + "/total_group_" + std::to_string(p);
            writer.WriteElement(tPrefix + "/I", totalGrpI);
            writer.WriteElement(tPrefix + "/Q", totalGrpQ);
            writer.WriteElement(tPrefix + "/U", totalGrpU);
        }

        writer.WriteElement(prefix + "/metadata/group_luminosity_below_cutoff_valid", 0.0);
        writer.WriteElement(prefix + "/metadata/group_luminosity_contains_collapsed_pgrw", 1.0);
        writer.WriteElement(prefix + "/metadata/resolved_collapsed_split_available", 1.0);
        writer.WriteElement(prefix + "/metadata/pgrw_output_semantics",
            std::string("total_contains_collapsed; resolved_split_available"));
    }
    else
    {
        writer.WriteElement(prefix + "/metadata/group_luminosity_contains_collapsed_pgrw", 0.0);
        writer.WriteElement(prefix + "/metadata/resolved_collapsed_split_available", 0.0);
    }

    std::vector<double> pktCountDbl(packetCount_.begin(), packetCount_.end());
    writer.WriteElement(prefix + "/packet_count_by_observer", pktCountDbl);
}
