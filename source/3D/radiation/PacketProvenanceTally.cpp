#include "PacketProvenanceTally.hpp"

#include <algorithm>
#include <cmath>

#include "source/misc/universal_error.hpp"
#include "monte/particle/PacketProvenance.hpp"


PacketProvenanceTally::PacketProvenanceTally(size_t groups)
    : groups_(groups), reference_(groups, 0.0)
{
    size_t const classCells = kCandidates * groups * kXBins * kClasses;
    size_t const birthCells = groups * (groups + 1) * kXBins;
    size_t const channelCells = groups * kChannelMasks * kXBins;
    size_t const kindCells = groups * kBirthKinds * kXBins;
    size_t const refusedCells = groups * kXBins;
    birthOffset_ = classCells * kStats;
    channelOffset_ = birthOffset_ + birthCells * kStats;
    kindOffset_ = channelOffset_ + channelCells * kStats;
    refusedOffset_ = kindOffset_ + kindCells * kStats;
    data_.assign(refusedOffset_ + refusedCells * kStats, 0.0);
}

void PacketProvenanceTally::setReference(std::vector<double> const& referenceEnergy)
{
    if (referenceEnergy.size() != groups_)
        throw UniversalError("PacketProvenanceTally: reference energy has the wrong number of groups");
    for (double const e : referenceEnergy)
        if (!(e >= 0.0) || !std::isfinite(e))
            throw UniversalError("PacketProvenanceTally: negative or non-finite reference energy");
    reference_ = referenceEnergy;
}

size_t PacketProvenanceTally::xBin(double weight, double reference)
{
    if (!(reference > 0.0))
        return kMissingBin;
    double const x = std::abs(weight) / reference;
    // Bins closed on the right, (e_k-1, e_k]: the tail x > 0.03 starts strictly above 0.03.
    return static_cast<size_t>(std::lower_bound(kXEdges.begin(), kXEdges.end(), x) - kXEdges.begin());
}

size_t PacketProvenanceTally::classIndex(size_t candidate, size_t group, size_t bin, size_t cls) const
{
    return (((candidate * groups_ + group) * kXBins + bin) * kClasses + cls) * kStats;
}

size_t PacketProvenanceTally::birthIndex(size_t group, size_t birthGroup, size_t bin) const
{
    return birthOffset_ + ((group * (groups_ + 1) + birthGroup) * kXBins + bin) * kStats;
}

size_t PacketProvenanceTally::channelIndex(size_t group, size_t mask, size_t bin) const
{
    return channelOffset_ + ((group * kChannelMasks + mask) * kXBins + bin) * kStats;
}

size_t PacketProvenanceTally::kindIndex(size_t group, size_t kind, size_t bin) const
{
    return kindOffset_ + ((group * kBirthKinds + kind) * kXBins + bin) * kStats;
}

size_t PacketProvenanceTally::refusedIndex(size_t group, size_t bin) const
{
    return refusedOffset_ + (group * kXBins + bin) * kStats;
}

void PacketProvenanceTally::record(size_t escapeGroup, double weight, double splitFactor,
                                   std::uint8_t birthGroup, std::uint8_t provenance0,
                                   std::uint8_t provenance1)
{
    if (escapeGroup >= groups_ || weight == 0.0 || !std::isfinite(weight))
        return;
    double const w = std::abs(weight);
    double const w2 = w * w;
    double const kw2 = std::max(1.0, splitFactor) * w2;
    size_t const bin = xBin(w, reference_[escapeGroup]);
    auto add = [&](size_t index) {
        data_[index] += 1.0;
        data_[index + 1] += w;
        data_[index + 2] += w2;
        data_[index + 3] += kw2;
    };
    for (size_t c = 0; c < kCandidates; ++c) {
        size_t const cls = static_cast<size_t>(
            STORM::provenance::classify(provenance0, provenance1, static_cast<unsigned>(c)));
        add(classIndex(c, escapeGroup, bin, cls));
    }
    size_t const born = birthGroup < groups_ ? birthGroup : groups_;
    add(birthIndex(escapeGroup, born, bin));
    add(channelIndex(escapeGroup, STORM::provenance::channelMask(provenance1), bin));
    size_t const kind = (provenance0 & STORM::provenance::kBirthKindMask) +
        ((provenance0 & STORM::provenance::kExploration) ? 4u : 0u);
    add(kindIndex(escapeGroup, kind, bin));
    if (provenance0 & STORM::provenance::kRefusedAncestor)
        add(refusedIndex(escapeGroup, bin));
}

void PacketProvenanceTally::clear()
{
    std::fill(data_.begin(), data_.end(), 0.0);
}

void PacketProvenanceTally::add(PacketProvenanceTally const& other)
{
    if (other.data_.size() != data_.size())
        throw UniversalError("PacketProvenanceTally: adding tallies of different shapes");
    for (size_t k = 0; k < data_.size(); ++k)
        data_[k] += other.data_[k];
}

std::vector<PacketProvenanceTally::BandTail> PacketProvenanceTally::summarize() const
{
    std::vector<BandTail> bands(groups_);
    for (size_t g = 0; g < groups_; ++g) {
        BandTail& band = bands[g];
        for (size_t bin = kTailFirstBin; bin < kMissingBin; ++bin) {
            for (size_t cls = 0; cls < kClasses; ++cls) {
                size_t const i0 = classIndex(0, g, bin, cls);
                band.tailW2 += data_[i0 + 2];
                band.crossings += data_[i0];
                for (size_t c = 0; c < kCandidates; ++c)
                    band.classW2[c][cls] += data_[classIndex(c, g, bin, cls) + 2];
            }
            for (size_t born = 0; born <= groups_; ++born) {
                double const w2 = data_[birthIndex(g, born, bin) + 2];
                if (born == groups_)
                    band.unknownBirthW2 += w2;
                else if (born != g)
                    band.bornElsewhereW2 += w2;
            }
            band.refusedW2 += data_[refusedIndex(g, bin) + 2];
        }
    }
    return bands;
}

std::vector<PacketProvenanceTally::Section> PacketProvenanceTally::sections() const
{
    return {
        {"class", 0, {kCandidates, groups_, kXBins, kClasses, kStats}},
        {"birth_group", birthOffset_, {groups_, groups_ + 1, kXBins, kStats}},
        {"channel", channelOffset_, {groups_, kChannelMasks, kXBins, kStats}},
        {"birth_kind", kindOffset_, {groups_, kBirthKinds, kXBins, kStats}},
        {"refused", refusedOffset_, {groups_, kXBins, kStats}},
    };
}
