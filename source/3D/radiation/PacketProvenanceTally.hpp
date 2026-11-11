#ifndef RICH_PACKET_PROVENANCE_TALLY_HPP
#define RICH_PACKET_PROVENANCE_TALLY_HPP

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

// Escape tallies of the packet provenance diagnostic (STORM
// particle/PacketProvenance.hpp; diagnostics.packet-provenance). Every
// outward crossing of escape group g adds (count, sum w, sum w^2, sum k w^2)
// to cells indexed by x = w / e_g, the packet weight over the band's
// per-observer escaping energy of the previous generation (bins with edges
// kXEdges, plus a "missing reference" bin when e_g = 0), crossed with:
//   class   [candidate][g][x][TailClass]      mutually exclusive tail classes
//   birth   [g][birth group or unknown][x]
//   channel [g][4-bit group-change channel mask][x]
//   kind    [g][birth kind + 4 x exploration][x]
//   refused [g][x]                             descended from a refused split
// The tally is a flat vector of doubles so ranks can reduce it in one call.
class PacketProvenanceTally
{
public:
    static constexpr std::array<double, 7> kXEdges = {1e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0, 10.0};
    static constexpr size_t kXBins = kXEdges.size() + 2;   // 8 ranges + missing reference
    static constexpr size_t kMissingBin = kXBins - 1;
    // Bins are closed on the right: [0, 1e-3], (1e-3, 1e-2], ..., (10, inf).
    // First bin of the escaping tail x > 0.03 (bins 3..7).
    static constexpr size_t kTailFirstBin = 3;
    static constexpr size_t kStats = 4;     // count, sum w, sum w^2, sum k w^2
    static constexpr size_t kCandidates = 2;
    static constexpr size_t kClasses = 5;   // STORM::provenance::TailClass
    static constexpr size_t kChannelMasks = 16;
    static constexpr size_t kBirthKinds = 8;

    explicit PacketProvenanceTally(size_t groups = 0);

    size_t groups() const { return groups_; }
    void setReference(std::vector<double> const& referenceEnergy);
    std::vector<double> const& reference() const { return reference_; }

    static size_t xBin(double weight, double reference);

    void record(size_t escapeGroup, double weight, double splitFactor,
                std::uint8_t birthGroup, std::uint8_t provenance0, std::uint8_t provenance1);
    void clear();
    void add(PacketProvenanceTally const& other);

    std::vector<double>& data() { return data_; }
    std::vector<double> const& data() const { return data_; }

    size_t classIndex(size_t candidate, size_t group, size_t bin, size_t cls) const;
    size_t birthIndex(size_t group, size_t birthGroup, size_t bin) const;
    size_t channelIndex(size_t group, size_t mask, size_t bin) const;
    size_t kindIndex(size_t group, size_t kind, size_t bin) const;
    size_t refusedIndex(size_t group, size_t bin) const;

    // Per band, from the tail bins (x > 0.03): sum w^2 in total and by class
    // for each candidate, sum w^2 of packets born in another group, and of
    // packets descended from a refused split.
    struct BandTail
    {
        double tailW2 = 0.0;
        std::array<std::array<double, kClasses>, kCandidates> classW2{};
        double bornElsewhereW2 = 0.0;
        double unknownBirthW2 = 0.0;
        double refusedW2 = 0.0;
        double crossings = 0.0;
    };
    std::vector<BandTail> summarize() const;

    // Flat sections with their shapes (for HDF5 output).
    struct Section
    {
        std::string name;
        size_t offset;
        std::vector<size_t> shape;   // including the trailing kStats
    };
    std::vector<Section> sections() const;

private:
    size_t groups_ = 0;
    std::vector<double> reference_;
    std::vector<double> data_;
    size_t birthOffset_ = 0;
    size_t channelOffset_ = 0;
    size_t kindOffset_ = 0;
    size_t refusedOffset_ = 0;
};

#endif // RICH_PACKET_PROVENANCE_TALLY_HPP
