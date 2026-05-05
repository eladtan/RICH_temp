#ifndef MONTE_CARLO_CONFIG_HPP
#define MONTE_CARLO_CONFIG_HPP

#include <cstddef>
#include <algorithm>
#include <cmath>
#include <iostream>

constexpr double MONTECARLO_EPSILON = 1e-8;

enum class MonteCarloTransferDiagnosticsLevel
{
    Off,
    StepSummary,
    StepSummaryAndHistograms
};

struct MonteCarloConfig
{
private:
    static constexpr size_t defaultSendBufferMinSize = 200;
    static constexpr size_t sendBufferMinSizeMin = 200;
    static constexpr size_t sendBufferMinSizeMax = 5000;
    static constexpr size_t sendBufferTargetParticlesPerFlush = 1024;
    static constexpr double sendBufferHighTransferFraction = 0.20;
    static constexpr double sendBufferLowTransferFraction = 0.08;

    static constexpr size_t smallIdleFlushHoldoffCyclesMin = 32;
    static constexpr size_t smallIdleFlushHoldoffCyclesMax = 512;
    static constexpr double smallIdleFlushHighCallFraction = 0.80;
    static constexpr double smallIdleFlushLowCallFraction = 0.50;
    static constexpr size_t smallIdleFlushPendingSoftLimitFactor = 512;

    size_t smallIdleFlushHoldoffCycles = smallIdleFlushHoldoffCyclesMin;

public:
    size_t initialBufferSize          = 500;
    size_t shrinkBuffersCycle         = 50;
    size_t sendBufferMinSize          = defaultSendBufferMinSize;
    size_t amountProgressMinCycles    = 16;
    size_t transferDiagnosticsEveryNSteps = 1;
    double bufferReallocationFactor   = 1.5;
    size_t minimalBuffSize            = 50;
    double bufferShrinkFactor         = 0.1;
    double bufferShrinkNeighborFactor = 0.5;
    double shrinkPercent              = 0.25;
    bool holdSmallIdleFlushes = false;
    MonteCarloTransferDiagnosticsLevel transferDiagnosticsLevel = MonteCarloTransferDiagnosticsLevel::StepSummary;

    size_t GetSmallIdleFlushHoldoffCycles(void) const
    {
        return this->smallIdleFlushHoldoffCycles;
    }

    void SyncSmallIdleFlushHoldoffCycles(size_t value)
    {
        this->smallIdleFlushHoldoffCycles = std::min<size_t>(
            smallIdleFlushHoldoffCyclesMax,
            std::max<size_t>(smallIdleFlushHoldoffCyclesMin, value));
    }

    static MonteCarloConfig Auto(size_t particlesPerRank, size_t numNeighbors)
    {
        MonteCarloConfig cfg;
        numNeighbors = std::max<size_t>(numNeighbors, 1);

        cfg.initialBufferSize = std::max<size_t>(100, particlesPerRank / numNeighbors);
        cfg.minimalBuffSize   = std::max<size_t>(10, cfg.initialBufferSize / 10);
        cfg.sendBufferMinSize = std::max<size_t>(sendBufferMinSizeMin, particlesPerRank / (numNeighbors * 4));

        return cfg;
    }

    #ifdef TIMING
    struct StepStats
    {
        double totalReallocationTime  = 0;
        size_t totalReallocations     = 0;
        size_t numHandlers            = 0;
        size_t totalIterations        = 0;
        size_t totalTransfers         = 0;
        double mainLoopTime           = 0;
        size_t peakBufferUsage        = 0;
        size_t totalSendFlushCalls    = 0;
        size_t totalSendFlushedParticles = 0;
        size_t totalSendIdleDrainFlushCalls = 0;
        size_t totalSendIdleDrainFlushedParticles = 0;
        double avgFlushTransferFraction = 0;
        size_t maxPendingSendBufferParticles = 0;
    };

    void Adapt(const StepStats &stats, int rank)
    {
        double avgReallocsPerHandler = (stats.numHandlers > 0)
            ? static_cast<double>(stats.totalReallocations) / stats.numHandlers
            : 0;

        if(avgReallocsPerHandler > 5.0)
            bufferReallocationFactor = std::min(2.5, bufferReallocationFactor * 1.1);
        else if(avgReallocsPerHandler < 1.0)
            bufferReallocationFactor = std::max(1.3, bufferReallocationFactor * 0.95);

        double reallocationFraction = (stats.mainLoopTime > 0)
            ? stats.totalReallocationTime / stats.mainLoopTime
            : 0;

        if(reallocationFraction > 0.05)
            shrinkBuffersCycle = std::min<size_t>(500, shrinkBuffersCycle * 2);
        else if(reallocationFraction < 0.01 && shrinkBuffersCycle > 10)
            shrinkBuffersCycle = std::max<size_t>(10, shrinkBuffersCycle / 2);

        if(avgReallocsPerHandler > 3.0)
        {
            bufferShrinkFactor = std::min(0.8, bufferShrinkFactor * 1.2);
            bufferShrinkNeighborFactor = std::min(0.9, bufferShrinkNeighborFactor * 1.1);
        }
        else if(avgReallocsPerHandler < 0.5)
        {
            bufferShrinkFactor = std::max(0.05, bufferShrinkFactor * 0.9);
            bufferShrinkNeighborFactor = std::max(0.2, bufferShrinkNeighborFactor * 0.95);
        }

        double particlesPerFlush = (stats.totalSendFlushCalls > 0)
            ? static_cast<double>(stats.totalSendFlushedParticles) / static_cast<double>(stats.totalSendFlushCalls)
            : 0;
        double targetParticlesPerFlush = static_cast<double>(std::max<size_t>(1, sendBufferTargetParticlesPerFlush));
        double idleDrainCallFraction = (stats.totalSendFlushCalls > 0)
            ? static_cast<double>(stats.totalSendIdleDrainFlushCalls) / static_cast<double>(stats.totalSendFlushCalls)
            : 0;
        double idleDrainParticleFraction = (stats.totalSendFlushedParticles > 0)
            ? static_cast<double>(stats.totalSendIdleDrainFlushedParticles) / static_cast<double>(stats.totalSendFlushedParticles)
            : 0;

        if(stats.totalSendFlushCalls > 0 and
           stats.avgFlushTransferFraction > sendBufferHighTransferFraction and
           particlesPerFlush < 0.75 * targetParticlesPerFlush)
        {
            sendBufferMinSize = std::min<size_t>(
                sendBufferMinSizeMax,
                std::max<size_t>(sendBufferMinSize + 128, sendBufferMinSize * 3 / 2));
        }
        else if(stats.totalSendFlushCalls > 0 and
                stats.avgFlushTransferFraction < sendBufferLowTransferFraction and
                particlesPerFlush > 1.5 * targetParticlesPerFlush)
        {
            sendBufferMinSize = std::max<size_t>(sendBufferMinSizeMin, sendBufferMinSize * 3 / 4);
        }
        sendBufferMinSize = std::min<size_t>(sendBufferMinSizeMax, std::max<size_t>(sendBufferMinSizeMin, sendBufferMinSize));

        if(holdSmallIdleFlushes and stats.totalSendFlushCalls > 0)
        {
            smallIdleFlushHoldoffCycles = std::min<size_t>(
                smallIdleFlushHoldoffCyclesMax,
                std::max<size_t>(smallIdleFlushHoldoffCyclesMin, smallIdleFlushHoldoffCycles));

            size_t pendingSoftLimit = std::max<size_t>(
                sendBufferTargetParticlesPerFlush * smallIdleFlushPendingSoftLimitFactor,
                sendBufferMinSizeMax * 64);
            size_t pendingHardLimit = pendingSoftLimit * 2;

            bool idleCallsDominate = idleDrainCallFraction > smallIdleFlushHighCallFraction;
            bool batchesStillSmall = particlesPerFlush < 0.5 * targetParticlesPerFlush;
            bool pendingStillModest = stats.maxPendingSendBufferParticles < pendingSoftLimit;

            if(stats.maxPendingSendBufferParticles > pendingHardLimit)
            {
                smallIdleFlushHoldoffCycles = std::max<size_t>(
                    smallIdleFlushHoldoffCyclesMin,
                    smallIdleFlushHoldoffCycles / 2);
            }
            else if(idleCallsDominate and batchesStillSmall and pendingStillModest)
            {
                smallIdleFlushHoldoffCycles = std::min<size_t>(
                    smallIdleFlushHoldoffCyclesMax,
                    std::max<size_t>(smallIdleFlushHoldoffCycles + 16,
                                     smallIdleFlushHoldoffCycles * 2));
            }
            else if(idleDrainCallFraction < smallIdleFlushLowCallFraction and
                    particlesPerFlush > 0.75 * targetParticlesPerFlush)
            {
                smallIdleFlushHoldoffCycles = std::max<size_t>(
                    smallIdleFlushHoldoffCyclesMin,
                    smallIdleFlushHoldoffCycles * 3 / 4);
            }
        }

        if(stats.peakBufferUsage > 0)
        {
            size_t suggested = static_cast<size_t>(std::ceil(stats.peakBufferUsage * 1.2));
            initialBufferSize = std::max(minimalBuffSize, suggested);
        }

        if(rank == 0)
        {
            std::cout << "MonteCarloConfig adapted: initialBuf=" << initialBufferSize
                      << ", growFactor=" << bufferReallocationFactor
                      << ", shrinkCycle=" << shrinkBuffersCycle
                      << ", shrinkFactor=" << bufferShrinkFactor
                      << "/" << bufferShrinkNeighborFactor
                      << ", shrinkPercent=" << shrinkPercent
                      << ", sendMinSize=" << sendBufferMinSize
                      << ", particlesPerFlush=" << particlesPerFlush
                      << ", idleDrainCallFraction=" << idleDrainCallFraction
                      << ", idleDrainParticleFraction=" << idleDrainParticleFraction
                      << ", pendingParticlesPeak=" << stats.maxPendingSendBufferParticles
                      << ", flushTransferFraction=" << stats.avgFlushTransferFraction
                      << ", holdSmallIdleFlushes=" << holdSmallIdleFlushes
                      << ", smallIdleFlushHoldoffCycles=" << smallIdleFlushHoldoffCycles
                      << ", amountProgressCycles=" << amountProgressMinCycles
                      << std::endl;
        }
    }
    #endif // TIMING
};

#endif // MONTE_CARLO_CONFIG_HPP
