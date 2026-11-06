// Deterministic checks of the per-group weight window and the packet
// provenance diagnostic (band-SNR step 4), without transport:
//   - ComputeGroupWindows: exact windows for a known spectrum, the share
//     floor, the scale, rejection of invalid input and of underflow;
//   - CaptureWeightSplitSpectrum: lagged spectrum, an empty generation keeps
//     the previous one, a negative entry throws;
//   - provenance state machine (STORM PacketProvenance.hpp): birth bits,
//     opportunity then stochastic event, a ballistic history, a packet that
//     qualifies only for the second candidate, class priority, channels;
//   - PacketProvenanceTally: x bins at exact edges, the missing-reference
//     bin, class/birth/channel/kind/refused tallies, the tail summary,
//     section layout covering the whole table;
//   - ApplySeedOffset: offset 0 is the identity; no seed collides within or
//     between the scheduled seed sets of offsets 0, 1, 2 for a production
//     layout (1536 ranks, MG and grey, fresh and restart schedules);
//   - replaceWithEditedCopy (STORM): stale copies refused, equal, advanced
//     and other-key copies taken.
// Writes imc_postprocess_band_window_unit_metrics.txt with "pass 1|0".

#include <cmath>
#include <cstdint>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "source/3D/radiation/PacketProvenanceTally.hpp"
#include "source/3D/radiation/postprocess/IMCPostProcess.hpp"
#include "source/3D/radiation/postprocess/PostProcessCommandLine.hpp"
#include "source/3D/elementary/Vector3D.hpp"
#include "monte/particle/Particle.hpp"
#include "monte/radiation/transport/PostProcessWeightSplit.hpp"
#include "source/3D/radiation/postprocess/PostProcessRestart.hpp"
#include "source/3D/radiation/postprocess/WeightSplit.hpp"
#include "source/misc/universal_error.hpp"
#include "monte/particle/PacketProvenance.hpp"
#include "monte/radiation/imc/IMCComponentBase.hpp"

namespace
{
int gFailures = 0;
std::ofstream gMetrics;

void check(bool ok, std::string const& what)
{
    std::cout << (ok ? "PASS " : "FAIL ") << what << std::endl;
    if (!ok)
        ++gFailures;
}

bool throws(std::function<void()> const& f)
{
    try {
        f();
    } catch (UniversalError const&) {
        return true;
    }
    return false;
}

bool close(double a, double b, double rel = 1e-14)
{
    return std::abs(a - b) <= rel * std::max(std::abs(a), std::abs(b));
}

void TestGroupWindows()
{
    using imc_postprocess_tde::ComputeGroupWindows;
    std::vector<double> const spectrum = {1.0, 3.0, 6.0, 0.0};   // shares 0.1, 0.3, 0.6, 0
    double const w = 2.0e40;
    auto const windows = ComputeGroupWindows(w, 0.5, 1e-3, spectrum);
    check(windows.size() == 4 && close(windows[0], w * 0.5 * 0.1) && close(windows[1], w * 0.5 * 0.3) &&
              close(windows[2], w * 0.5 * 0.6) && close(windows[3], w * 0.5 * 1e-3),
          "group windows = W x scale x max(floor, share)");
    auto const unit = ComputeGroupWindows(w, 1.0, 1.0, spectrum);
    bool equal = true;
    for (double const x : unit)
        equal = equal && x == w;
    check(equal, "scale 1 and floor 1 reproduce the scalar window in every group");
    bool bounded = true;
    for (double const x : ComputeGroupWindows(w, 1.0, 1e-3, {5.0, 0.0, 0.0, 0.0}))
        bounded = bounded && x <= w;
    check(bounded, "no group window exceeds the scalar window");
    check(throws([&] { ComputeGroupWindows(0.0, 1.0, 1e-3, spectrum); }), "zero scalar window rejected");
    check(throws([&] { ComputeGroupWindows(w, 0.0, 1e-3, spectrum); }), "scale 0 rejected");
    check(throws([&] { ComputeGroupWindows(w, 1.5, 1e-3, spectrum); }), "scale > 1 rejected");
    check(throws([&] { ComputeGroupWindows(w, 1.0, 0.0, spectrum); }), "floor 0 rejected");
    check(throws([&] { ComputeGroupWindows(w, 1.0, 1e-3, {0.0, 0.0, 0.0, 0.0}); }), "empty spectrum rejected");
    check(throws([&] { ComputeGroupWindows(w, 1.0, 1e-3, {1.0, -1.0, 0.0, 0.0}); }), "negative spectrum rejected");
    check(throws([&] { ComputeGroupWindows(w, 1.0, 1e-3, {1.0, std::nan(""), 0.0, 0.0}); }),
          "non-finite spectrum rejected");
    check(throws([&] { ComputeGroupWindows(1e-320, 1e-3, 1e-3, spectrum); }), "underflowed window rejected");
}

void TestImportanceFloorGate()
{
    using imc_postprocess_tde::EffectiveImportanceFloor;
    check(EffectiveImportanceFloor(0.01, false) == 0.0 && EffectiveImportanceFloor(0.01, true) == 0.01,
          "importance floor inactive until the full-emission importance statistics start");
}

void TestSpectrumCapture()
{
    using imc_postprocess_tde::CaptureWeightSplitSpectrum;
    imc_postprocess_tde::WeightSplitState state;
    check(!state.spectrumValid && state.spectrumGeneration == -1, "no spectrum before the first capture");
    CaptureWeightSplitSpectrum({1.0, 2.0, 3.0, 4.0}, 21, state);
    check(state.spectrumValid && state.spectrumGeneration == 21 && state.groupSpectrum[3] == 4.0,
          "capture stores the spectrum and its generation");
    CaptureWeightSplitSpectrum({0.0, 0.0, 0.0, 0.0}, 22, state);
    check(state.spectrumValid && state.spectrumGeneration == 21 && state.groupSpectrum[0] == 1.0,
          "a generation with no escaping energy keeps the previous spectrum");
    check(throws([&] { CaptureWeightSplitSpectrum({1.0, -2.0, 0.0, 0.0}, 23, state); }),
          "negative escaping energy throws");
}

void TestProvenanceStateMachine()
{
    namespace pv = STORM::provenance;
    using pv::TailClass;
    std::uint8_t p0 = pv::birthBits(3, true);
    check((p0 & pv::kBirthKindMask) == 3 && (p0 & pv::kExploration) != 0 && (p0 & ~0x07u) == 0,
          "birth bits: kind and exploration only");
    // Never overweight anywhere: class None for both candidates.
    std::uint8_t p1 = 0;
    check(pv::classify(0, 0, 0) == TailClass::None && pv::classify(0, 0, 1) == TailClass::None,
          "no history: class none");
    // A stochastic event before any opportunity does not count.
    pv::markStochastic(p1);
    check(p1 == 0, "stochastic event without an opportunity sets nothing");
    // Ballistic: opportunity for candidate 0, nothing stochastic after it.
    p1 = pv::kOpportunity[0];
    check(pv::classify(0, p1, 0) == TailClass::Ballistic, "opportunity without a later event: ballistic");
    pv::markStochastic(p1);
    check(pv::classify(0, p1, 0) == TailClass::Useful, "opportunity then a stochastic event: useful");
    // Only the tighter candidate (1) qualified.
    p1 = pv::kOpportunity[1];
    pv::markStochastic(p1);
    check(pv::classify(0, p1, 1) == TailClass::Useful && pv::classify(0, p1, 0) == TailClass::None,
          "a packet over only the beta=0.2 window is useful for candidate 1 and none for candidate 0");
    // Class priority: opportunity beats zero-importance beats DDMC.
    std::uint8_t const zero0 = pv::kZeroImportanceOverweight[0];
    check(pv::classify(zero0 | pv::kEverDDMC, 0, 0) == TailClass::ZeroImportance,
          "zero importance before DDMC in the class order");
    check(pv::classify(zero0, pv::kOpportunity[0], 0) == TailClass::Ballistic,
          "an opportunity outranks the zero-importance flag");
    check(pv::classify(pv::kEverPendingFlux, 0, 0) == TailClass::DDMC && pv::classify(zero0, 0, 1) == TailClass::None,
          "pending flux is DDMC class; candidate flags are per candidate");
    // Channels are independent of the candidate bits.
    p1 = static_cast<std::uint8_t>(pv::kChannelThermal | pv::kChannelFace);
    check(pv::channelMask(p1) == ((pv::kChannelThermal | pv::kChannelFace) >> pv::kChannelShift) &&
              pv::classify(0, p1, 0) == TailClass::None,
          "channel bits form the 4-bit mask and leave the class unchanged");
}

void TestProvenanceTally()
{
    namespace pv = STORM::provenance;
    using T = PacketProvenanceTally;
    check(T::xBin(1.0, 0.0) == T::kMissingBin, "zero reference: missing-reference bin");
    check(T::xBin(0.03, 1.0) == 2 && T::xBin(std::nextafter(0.03, 1.0), 1.0) == 3,
          "the tail is x > 0.03: x = 0.03 stays out, the next double is in");
    check(T::xBin(1e-4, 1.0) == 0 && T::xBin(100.0, 1.0) == 7, "x below 1e-3 and above 10");

    T tally(4);
    tally.setReference({1.0, 1.0, 1.0, 0.0});
    // Group 1: a useful tail packet (x=0.5, born in group 2, refused ancestor,
    // split twice), a ballistic one (x=0.05), a light one (x=0.001, not tail).
    std::uint8_t useful = pv::kOpportunity[0];
    pv::markStochastic(useful);
    tally.record(1, 0.5, 2.0, 2, pv::birthBits(1, false) | pv::kRefusedAncestor, useful);
    tally.record(1, 0.05, 1.0, 1, pv::birthBits(0, true), pv::kOpportunity[0]);
    tally.record(1, 0.001, 1.0, 1, 0, 0);
    // Group 3 has no reference: missing bin, never tail.
    tally.record(3, 5.0, 1.0, pv::kUnknownGroup, 0, 0);
    auto const bands = tally.summarize();
    double const w2 = 0.25 + 0.0025;
    check(close(bands[1].tailW2, w2) && close(bands[1].classW2[0][0], 0.25) &&
              close(bands[1].classW2[0][1], 0.0025) && close(bands[1].classW2[1][4], w2),
          "tail w^2 by class per candidate");
    check(close(bands[1].bornElsewhereW2, 0.25) && close(bands[1].refusedW2, 0.25) && bands[1].crossings == 2.0,
          "born-elsewhere and refused-ancestor tail shares");
    check(bands[3].tailW2 == 0.0 && tally.data()[tally.classIndex(0, 3, T::kMissingBin, 4)] == 1.0,
          "no reference: counted in the missing bin, not the tail");
    std::vector<double> const& d = tally.data();
    size_t const i = tally.classIndex(0, 1, T::xBin(0.5, 1.0), 0);
    check(d[i] == 1.0 && d[i + 1] == 0.5 && d[i + 2] == 0.25 && d[i + 3] == 0.5,
          "count, sum w, sum w^2 and sum k w^2 of one cell");
    size_t covered = 0;
    for (auto const& s : tally.sections()) {
        size_t n = 1;
        for (size_t const x : s.shape)
            n *= x;
        check(s.offset == covered, "section " + s.name + " starts where the previous ends");
        covered += n;
    }
    check(covered == d.size(), "sections cover the whole table");
    T other(4);
    other.setReference({1.0, 1.0, 1.0, 0.0});
    other.add(tally);
    check(other.data() == tally.data(), "add() of a tally into an empty one copies it");
}

void TestSeedOffsets()
{
    using imc_postprocess_tde::ApplySeedOffset;
    using imc_postprocess_tde::RestartGenerationSeed;
    using imc_postprocess_tde::RestartPass;
    bool identity = true;
    for (uint64_t s : {uint64_t(0), uint64_t(1), uint64_t(12345678) * 232, ~uint64_t(0)})
        identity = identity && ApplySeedOffset(s, 0) == s;
    check(identity, "seed offset 0 keeps every seed");
    // Production layout: 1536 ranks; MG 232 transported generations, grey
    // 230; restart schedules over 400 final generations.
    int const ranks = 1536;
    size_t const mgGenerations = 232, greyGenerations = 230, restartFinal = 400;
    std::unordered_set<uint64_t> seen;
    seen.reserve(3 * static_cast<size_t>(ranks) * (mgGenerations + greyGenerations + 2 * restartFinal));
    size_t inserted = 0;
    bool unique = true;
    for (uint64_t offset = 0; offset < 3; ++offset) {
        for (int r = 0; r < ranks; ++r) {
            auto add = [&](uint64_t legacy) {
                ++inserted;
                unique = seen.insert(ApplySeedOffset(legacy, offset)).second && unique;
            };
            for (size_t g = 0; g < mgGenerations; ++g)
                add(static_cast<uint64_t>(r + 12345678) * mgGenerations + g);
            for (size_t g = 0; g < greyGenerations; ++g)
                add(static_cast<uint64_t>(r + 87654321) * greyGenerations + g);
            for (size_t g = 0; g < restartFinal; ++g) {
                add(RestartGenerationSeed(RestartPass::Forward, r, g));
                add(RestartGenerationSeed(RestartPass::Grey, r, g));
            }
        }
    }
    gMetrics << "seed_sets inserted " << inserted << " unique " << seen.size() << "\n";
    // Downstream: STORM seeds rank r with S + 104729 r (particle keys also
    // carry the creation rank), so the seeds must be distinct per rank.
    bool downstreamUnique = true;
    for (int r = 0; r < ranks && downstreamUnique; ++r) {
        std::unordered_set<uint64_t> perRank;
        for (uint64_t offset = 0; offset < 3; ++offset) {
            auto add = [&](uint64_t legacy) {
                uint64_t const downstream = ApplySeedOffset(legacy, offset) + static_cast<uint64_t>(r) * 104729ULL;
                downstreamUnique = perRank.insert(downstream).second && downstreamUnique;
            };
            for (size_t g = 0; g < mgGenerations; ++g)
                add(static_cast<uint64_t>(r + 12345678) * mgGenerations + g);
            for (size_t g = 0; g < greyGenerations; ++g)
                add(static_cast<uint64_t>(r + 87654321) * greyGenerations + g);
            for (size_t g = 0; g < restartFinal; ++g) {
                add(RestartGenerationSeed(RestartPass::Forward, r, g));
                add(RestartGenerationSeed(RestartPass::Grey, r, g));
            }
        }
    }
    check(downstreamUnique, "downstream rank seeds (S + 104729 r) are distinct on every rank");
    // Burn-in volume-emission subsample seeds (generation + 1): offset 0 keeps
    // them, offsets 1 and 2 give disjoint sets.
    std::unordered_set<uint64_t> subsample;
    bool subsampleOk = true;
    for (uint64_t offset = 0; offset < 3; ++offset)
        for (uint64_t g = 0; g < mgGenerations; ++g) {
            uint64_t const s = ApplySeedOffset(g + 1, offset);
            subsampleOk = subsample.insert(s).second && subsampleOk && (offset != 0 || s == g + 1);
        }
    check(subsampleOk, "burn-in subsample seeds follow the offset (identity at 0, disjoint otherwise)");
    check(unique && seen.size() == inserted,
          "no seed collides within or between offsets 0, 1, 2 (" + std::to_string(inserted) + " seeds)");
}
using Packet = STORM::Particle<Vector3D>;
constexpr std::size_t kG = 4;
using Window = STORM::postprocess_split::GroupWindow<kG>;
using Inputs = STORM::postprocess_split::Inputs<kG>;
using ScalarCounters = STORM::postprocess_split::ScalarCounters;
using GroupCounters = STORM::postprocess_split::GroupCounters<kG>;

// Lab group of a frequency for boundaries 0, 1, 2, 3, 4 ([b_g, b_g+1)).
std::array<double, kG + 1> const kBoundaries = {0.0, 1.0, 2.0, 3.0, 4.0};
size_t gGroupLookups = 0;
size_t LabGroup(double frequency)
{
    ++gGroupLookups;
    for (size_t g = 0; g < kG; ++g)
        if (frequency < kBoundaries[g + 1])
            return g;
    return kG - 1;
}

uint64_t gNextKey = 1000;
void InitRng(Packet& p)
{
    p.rngKey = gNextKey++;
    p.rngCounter = 0;
}

Packet MakePacket(double weight, double frequency, size_t cell)
{
    Packet p;
    p.weight = weight;
    p.initialWeight = 2.0 * weight;
    p.frequency = frequency;
    p.cellIndex = cell;
    p.rngKey = 7;
    p.rngCounter = 3;
    p.splitFactor = 1.0f;
#ifdef MONTECARLO_POLARIZATION
    p.stokesQ = 0.25;
    p.stokesU = -0.125;
#endif
    return p;
}

void TestSplitter()
{
    namespace ps = STORM::postprocess_split;
    namespace pv = STORM::provenance;
    std::vector<double> importance = {1.0, 0.5, 0.0, 0.01};
    Inputs in;
    in.scalarWindow = 10.0;
    in.importance = &importance;
    in.maxCopies = 1000;
    in.copyBudget = 1000000;
    auto run = [&](Packet& p, std::vector<Packet>& out, ScalarCounters& c, GroupCounters& gc,
                   Window const* w = nullptr) {
        Inputs local = in;
        local.groupWindow = w;
        return ps::apply<kG>(p, out, local, c, gc, LabGroup, InitRng);
    };
    {   // finalize
        Window w;
        w.active = true;
        w.maxWeight = {1.0, 2.0, 3.0, 4.0};
        w.importanceFloor = 0.25;
        Window const f = ps::finalize(w, 10.0);
        check(f.minMaxWeight == 1.0 && f.importanceFloor == 0.25, "finalize: smallest window and floor kept");
        Window inactive = w;
        inactive.active = false;
        check(ps::finalize(inactive, 10.0).importanceFloor == 0.0, "finalize: floor only with the active window");
        bool threw = false;
        try { ps::finalize(w, 0.0); } catch (STORM::StormError const&) { threw = true; }
        check(threw, "finalize: no scalar window rejected");
        Window bad = w;
        bad.maxWeight[2] = 0.0;
        threw = false;
        try { ps::finalize(bad, 10.0); } catch (STORM::StormError const&) { threw = true; }
        check(threw, "finalize: zero group window rejected");
        bad = w;
        bad.importanceFloor = 1.5;
        threw = false;
        try { ps::finalize(bad, 10.0); } catch (STORM::StormError const&) { threw = true; }
        check(threw, "finalize: importance floor > 1 rejected");
    }
    {   // threshold: |w| = window does not split, the next double does
        ScalarCounters c;
        GroupCounters gc;
        std::vector<Packet> out;
        Packet p = MakePacket(20.0, 0.5, 1);   // importance 0.5: window 20
        check(run(p, out, c, gc) == 0 && out.empty() && p.weight == 20.0, "weight equal to the window: no split");
        Packet q = MakePacket(std::nextafter(20.0, 30.0), 0.5, 1);
        check(run(q, out, c, gc) == 2 && out.size() == 1, "weight just above the window: two copies");
    }
    {   // conservation, k, RNG, Stokes
        ScalarCounters c;
        GroupCounters gc;
        std::vector<Packet> out;
        Packet p = MakePacket(55.0, 0.5, 0);   // window 10: ceil(5.5) = 6 copies
        Packet const original = p;
        size_t const copies = run(p, out, c, gc);
        double weight = p.weight, initial = p.initialWeight, kw2 = p.splitFactor * p.weight * p.weight;
        bool sameState = true, freshKeys = true;
        std::unordered_set<uint64_t> keys{p.rngKey};
        for (Packet const& e : out) {
            weight += e.weight;
            initial += e.initialWeight;
            kw2 += e.splitFactor * e.weight * e.weight;
            sameState = sameState && e.weight == p.weight && e.splitFactor == p.splitFactor &&
                e.frequency == p.frequency && e.cellIndex == p.cellIndex;
#ifdef MONTECARLO_POLARIZATION
            sameState = sameState && e.stokesQ == original.stokesQ && e.stokesU == original.stokesU;
#endif
            freshKeys = keys.insert(e.rngKey).second && freshKeys && e.rngCounter == 0;
        }
        check(copies == 6 && out.size() == 5 && p.splitFactor == 6.0f, "6 equal copies, split factor 6");
        check(close(weight, original.weight) && close(initial, original.initialWeight),
              "weight and initial weight conserved by the split");
        check(close(kw2, original.weight * original.weight), "sum of k w^2 equals the unsplit w^2");
        check(sameState, "copies share the state and the (normalized) Stokes parameters");
        check(freshKeys && p.rngKey == original.rngKey && p.rngCounter == original.rngCounter,
              "copies get distinct fresh RNG streams, the original keeps its own");
        // A second, repeated split of a copy multiplies k again.
        Packet again = out[0];
        again.weight *= 30.0;
        std::vector<Packet> out2;
        run(again, out2, c, gc);
        check(again.splitFactor > 6.0f && keys.count(out2.at(0).rngKey) == 0, "repeated splits: k multiplies, new streams");
        check(c.events == 2 && c.copies == 5 + out2.size(), "scalar counters count events and copies");
    }
    {   // cap and budget
        ScalarCounters c;
        GroupCounters gc;
        std::vector<Packet> out;
        Inputs saved = in;
        in.maxCopies = 4;
        Packet p = MakePacket(1000.0, 0.5, 0);
        run(p, out, c, gc);
        check(out.size() == 3 && c.capped == 1, "maxCopies caps a split (counted)");
        in = saved;
        in.copyBudget = 4;
        ScalarCounters exact;
        std::vector<Packet> outExact;
        Packet e = MakePacket(50.0, 0.5, 0);   // wants 5 copies: 4 extra, exactly the budget
        run(e, outExact, exact, gc);
        check(outExact.size() == 4 && exact.refused == 0 && exact.copies == 4,
              "a split that uses the budget exactly is not a refusal");
        in.copyBudget = 2;
        ScalarCounters c2;
        std::vector<Packet> out2;
        Packet q = MakePacket(50.0, 0.5, 0);   // wants 5 copies, room for 2 extra
        run(q, out2, c2, gc);
        check(out2.size() == 2 && c2.refused == 1 && close(q.weight * 3.0, 50.0), "partial budget: reduced to room + 1");
        Window prov;
        prov.provenance = true;
        prov = ps::finalize(prov, in.scalarWindow);
        Packet r = MakePacket(50.0, 0.5, 0);
        std::vector<Packet> out3;
        run(r, out3, c2, gc, &prov);
        check(out3.empty() && c2.refused == 2 && (r.provenance0 & pv::kRefusedAncestor) && r.weight == 50.0,
              "exhausted budget: no split, refusal counted, ancestor flag set");
        in = saved;
    }
    {   // DDMC, pending flux, zero importance, missing cell
        ScalarCounters c;
        GroupCounters gc;
        std::vector<Packet> out;
        Window prov;
        prov.provenance = true;
        prov = ps::finalize(prov, in.scalarWindow);
        Packet d = MakePacket(500.0, 0.5, 0);
        d.radiationState.set(STORM::RadiationTransportState<Vector3D>::DDMCMode);
        size_t const lookups = gGroupLookups;
        run(d, out, c, gc, &prov);
        check(out.empty() && c.skippedResident == 1 && (d.provenance0 & pv::kEverDDMC) && gGroupLookups == lookups,
              "DDMC resident: no split, counted, history set, no lab-group lookup");
        Packet f = MakePacket(500.0, 0.5, 0);
        f.radiationState.set(STORM::RadiationTransportState<Vector3D>::PendingFlux);
        run(f, out, c, gc, &prov);
        check(out.empty() && c.skippedResident == 2 && (f.provenance0 & pv::kEverPendingFlux),
              "pending flux: no split, counted, history set");
        Packet z = MakePacket(500.0, 0.5, 2);   // importance 0
        Packet m = MakePacket(500.0, 0.5, 9);   // no importance entry
        run(z, out, c, gc);
        run(m, out, c, gc);
        check(out.empty() && c.skippedNoImportance == 2, "zero importance and cells without an entry never split");
    }
    {   // active per-group window, exact group edges, importance floor
        Window w;
        w.active = true;
        w.maxWeight = {1.0, 2.0, 4.0, 8.0};
        w.importanceFloor = 0.1;
        w = ps::finalize(w, in.scalarWindow);
        ScalarCounters c;
        GroupCounters gc;
        std::vector<Packet> out;
        Packet edge = MakePacket(3.0, 1.0, 0);   // frequency on b_1: group 1, window 2
        run(edge, out, c, gc, &w);
        check(out.size() == 1 && gc.events[1] == 1 && gc.copies[1] == 1, "frequency on a boundary belongs to the upper group");
        Packet below = MakePacket(3.0, std::nextafter(1.0, 0.0), 0);   // group 0, window 1: 3 copies
        out.clear();
        run(below, out, c, gc, &w);
        check(out.size() == 2 && gc.events[0] == 1, "frequency just below a boundary belongs to the lower group");
        Packet floor = MakePacket(30.0, 3.5, 2);   // importance 0 -> floor 0.1: window 80
        out.clear();
        run(floor, out, c, gc, &w);
        Packet floor2 = MakePacket(81.0, 3.5, 2);
        run(floor2, out, c, gc, &w);
        check(out.size() == 1 && gc.events[3] == 1, "importance floor makes zero-importance cells split above W_g / floor");
        Packet huge = MakePacket(30.0, 3.5, 0);
        huge.splitFactor = 16777216.0f;
        bool threw = false;
        try { run(huge, out, c, gc, &w); } catch (STORM::StormError const&) { threw = true; }
        check(threw, "split factor beyond 2^24 stops the run");
    }
    {   // provenance candidates and fast-reject ordering
        Window w;
        w.active = true;
        w.maxWeight = {10.0, 10.0, 10.0, 10.0};
        w.provenance = true;
        w.shadowValid = true;
        w.shadowMaxWeight[0] = {10.0, 10.0, 10.0, 10.0};
        w.shadowMaxWeight[1] = {2.0, 2.0, 2.0, 2.0};
        w = ps::finalize(w, in.scalarWindow);
        ScalarCounters c;
        GroupCounters gc;
        std::vector<Packet> out;
        Packet p = MakePacket(5.0, 0.5, 0);   // below the active window, above candidate 1
        run(p, out, c, gc, &w);
        check(out.empty() && !(p.provenance1 & pv::kOpportunity[0]) && (p.provenance1 & pv::kOpportunity[1]),
              "fast reject keeps the shadow check: only candidate 1 records an opportunity");
        Packet z = MakePacket(5.0, 0.5, 2);   // zero importance, active floor 0
        run(z, out, c, gc, &w);
        check(!(z.provenance0 & pv::kZeroImportanceOverweight[0]) && (z.provenance0 & pv::kZeroImportanceOverweight[1]),
              "zero-importance overweight flag per candidate");
        Packet light = MakePacket(1.0, 0.5, 0);
        size_t const lookups = gGroupLookups;
        run(light, out, c, gc, &w);
        check(light.provenance1 == 0 && gGroupLookups == lookups, "a packet below every window returns before the group lookup");
    }
}

// Ballistic fixture: identical copies that never meet a stochastic event
// cut the raw sum of w^2 but not the generation-to-generation variance, and
// are classified ballistic, not useful.
void TestBallisticFixture()
{
    namespace ps = STORM::postprocess_split;
    namespace pv = STORM::provenance;
    std::vector<double> importance = {1.0};
    Inputs in;
    in.scalarWindow = 10.0;
    in.importance = &importance;
    in.maxCopies = 1000;
    in.copyBudget = 1000000;
    Window w;
    w.provenance = true;
    w.shadowValid = true;
    w.shadowMaxWeight[0] = {10.0, 10.0, 10.0, 10.0};
    w.shadowMaxWeight[1] = {2.0, 2.0, 2.0, 2.0};
    w = ps::finalize(w, in.scalarWindow);
    in.groupWindow = &w;
    size_t const generations = 200;
    std::vector<double> tallyUnsplit(generations, 0.0), tallySplit(generations, 0.0);
    double rawW2Unsplit = 0.0, rawW2Split = 0.0, ballisticW2 = 0.0, heavyW2 = 0.0;
    for (size_t j = 0; j < generations; ++j) {
        std::mt19937_64 rng(1234 + j);
        std::uniform_real_distribution<double> u(0.0, 1.0);
        for (int k = 0; k < 50; ++k) {
            double const weight = u(rng) < 0.05 ? 100.0 : 1.0;
            tallyUnsplit[j] += weight;
            rawW2Unsplit += weight * weight;
            Packet p = MakePacket(weight, 0.5, 0);
            std::vector<Packet> copies;
            ScalarCounters c;
            GroupCounters gc;
            ps::apply<kG>(p, copies, in, c, gc, LabGroup, InitRng);
            copies.push_back(p);
            // Ballistic escape: every copy crosses the same bin, nothing stochastic.
            for (Packet const& e : copies) {
                tallySplit[j] += e.weight;
                rawW2Split += e.weight * e.weight;
                if (weight > 10.0) {
                    heavyW2 += e.weight * e.weight;
                    if (pv::classify(e.provenance0, e.provenance1, 0) == pv::TailClass::Ballistic)
                        ballisticW2 += e.weight * e.weight;
                }
            }
        }
    }
    auto variance = [](std::vector<double> const& x) {
        double mean = 0.0, v = 0.0;
        for (double const a : x) mean += a / static_cast<double>(x.size());
        for (double const a : x) v += (a - mean) * (a - mean);
        return v / static_cast<double>(x.size() - 1);
    };
    double const varianceRatio = variance(tallySplit) / variance(tallyUnsplit);
    double const rawRatio = rawW2Unsplit / rawW2Split;
    gMetrics << "ballistic variance_ratio " << varianceRatio << " raw_w2_gain " << rawRatio << "\n";
    check(std::abs(varianceRatio - 1.0) < 1e-9 && rawRatio > 2.0,
          "ballistic copies: raw sum w^2 drops " + std::to_string(rawRatio) +
          "x while the generation variance is unchanged (ratio " + std::to_string(varianceRatio) + ")");
    check(heavyW2 > 0.0 && ballisticW2 == heavyW2, "every heavy ballistic copy is classified ballistic, none useful");
}

void TestCommandLine()
{
    auto parse = [](std::string const& value, PostProcessIMC::PostProcessConfig& config) {
        std::string const option = "--transport.seed-offset";
        std::vector<std::string> args = {"test", option, value};
        std::vector<char*> argv;
        for (auto& a : args)
            argv.push_back(a.data());
        return PostProcessIMC::ParseCommandLine(static_cast<int>(argv.size()), argv.data(), config, 1);
    };
    PostProcessIMC::PostProcessConfig config;
    bool const maxAccepted = parse("18446744073709551615", config) == PostProcessIMC::CommandLineAction::Run &&
        config.transport.seedOffset == std::numeric_limits<size_t>::max();
    check(maxAccepted, "seed offset: the largest 64-bit value is accepted");
    bool rejected = true;
    for (std::string const bad : {"18446744073709551616", "-1", " -1", " 1", "1e3", "abc", "", "+1"}) {
        PostProcessIMC::PostProcessConfig c;
        rejected = rejected && parse(bad, c) == PostProcessIMC::CommandLineAction::Error;
    }
    check(rejected, "seed offset: overflow, signs, spaces, exponents and junk are rejected");
}

// replaceWithEditedCopy (face re-emission, Compton): a copy whose stream is
// behind the packet's on the same key must be refused (the face bug: the
// copy was taken, two draws came from the packet, the copy was assigned
// back); a copy at or past the packet's counter, or on another key, is taken.
void TestEditedCopyGuard()
{
    auto attempt = [](Packet packet, Packet const& edited) {
        try {
            STORM::radiation_imc_detail::replaceWithEditedCopy(packet, edited);
        } catch (STORM::StormError const&) {
            return std::make_pair(true, packet);
        }
        return std::make_pair(false, packet);
    };
    Packet packet = MakePacket(1.0, 0.5, 0);
    packet.rngCounter = 24;
    Packet edited = packet;
    edited.weight = 2.0;

    edited.rngCounter = 22;   // two draws taken from the packet after the copy
    auto const stale = attempt(packet, edited);
    check(stale.first && stale.second.rngCounter == 24 && stale.second.weight == 1.0,
          "edited-copy guard: a copy behind the packet's stream is refused and the packet kept");
    edited.rngCounter = 24;
    auto const equal = attempt(packet, edited);
    check(!equal.first && equal.second.weight == 2.0 && equal.second.rngCounter == 24,
          "edited-copy guard: a copy at the packet's counter is taken");
    edited.rngCounter = 26;   // the draws came from the copy (the fix)
    auto const advanced = attempt(packet, edited);
    check(!advanced.first && advanced.second.rngCounter == 26,
          "edited-copy guard: a copy past the packet's counter is taken");
    edited.rngKey = packet.rngKey + 1;
    edited.rngCounter = 3;    // a fresh stream
    auto const fresh = attempt(packet, edited);
    check(!fresh.first && fresh.second.rngKey == packet.rngKey + 1 && fresh.second.rngCounter == 3,
          "edited-copy guard: a copy on another key is taken");
}
} // namespace

int main()
{
    gMetrics.open("imc_postprocess_band_window_unit_metrics.txt");
    try {
        TestGroupWindows();
        TestSpectrumCapture();
        TestImportanceFloorGate();
        TestProvenanceStateMachine();
        TestProvenanceTally();
        TestSeedOffsets();
        TestSplitter();
        TestBallisticFixture();
        TestCommandLine();
        TestEditedCopyGuard();
    } catch (UniversalError const& eo) {
        reportError(eo, std::cerr);
        ++gFailures;
    } catch (std::exception const& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        ++gFailures;
    }
    gMetrics << "failures " << gFailures << "\n";
    gMetrics << "pass " << (gFailures == 0 ? 1 : 0) << "\n";
    std::cout << (gFailures == 0 ? "imc_postprocess_band_window_unit PASSED" : "imc_postprocess_band_window_unit FAILED")
              << std::endl;
    return gFailures == 0 ? 0 : 1;
}
