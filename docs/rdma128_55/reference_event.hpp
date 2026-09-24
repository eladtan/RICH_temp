namespace STORM { namespace transport {
template<typename ParticleT, typename ViewsT, typename OpacityPolicyT>
STORM_TRANSPORT_INLINE TransportResult ReferenceAdvanceIMC(ParticleT &particle, const ViewsT &views, const OpacityPolicyT &opacity)
{
    TransportResult result;
    const std::size_t cellIndex = static_cast<std::size_t>(particle.cellIndex);
    if(cellIndex >= views.grid.cellCount)
    {
        result.error = TransportError::InvalidCell;
        return result;
    }

    const double speed = Sqrt(particle.velocity.x * particle.velocity.x + particle.velocity.y * particle.velocity.y + particle.velocity.z * particle.velocity.z);
    const gpu::Intersection intersection = gpu::FindIntersection(particle, views.grid, speed);
    if(not intersection.valid)
    {
        result.error = TransportError::NoIntersection;
        return result;
    }

    double dopplerShift = 1.0;
    if(views.comovingTransport)
    {
        if(views.cellVelocities == nullptr)
        {
            result.error = TransportError::InvalidDoppler;
            return result;
        }
        const auto &cellVelocity = views.cellVelocities[cellIndex];
        const double velocitySquared = Dot(cellVelocity, cellVelocity);
        if(velocitySquared >= 1.0e-30)
        {
            const double inverseC2 = 1.0 / (views.speedOfLight * views.speedOfLight);
            const double gammaArgument = 1.0 - velocitySquared * inverseC2;
            if(not (gammaArgument > 0.0) or not IsFinite(gammaArgument))
            {
                result.error = TransportError::InvalidDoppler;
                return result;
            }
            const double gamma = 1.0 / Sqrt(gammaArgument);
            dopplerShift = gamma * (1.0 - Dot(cellVelocity, particle.velocity) * inverseC2);
                if(not (dopplerShift > 0.0) or not IsFinite(dopplerShift))
                {
                    result.error = TransportError::InvalidDoppler;
                    return result;
                }
        }
    }

    const double transportFrequency = particle.frequency * dopplerShift;
    const IMCOpacityState opacityState = opacity.Evaluate(particle, views, cellIndex, transportFrequency);
    const double absorptionOpacity = opacityState.absorption;
    const double scatteringOpacity = opacityState.scattering;
    const double fleck = opacityState.fleck;
    if(not IsFinite(absorptionOpacity) or not IsFinite(scatteringOpacity) or absorptionOpacity < 0.0 or scatteringOpacity < 0.0 or fleck < 0.0 or fleck > 1.0)
    {
        result.error = TransportError::InvalidOpacity;
        return result;
    }

    const double effectiveAbsorptionOpacity = (1.0 - fleck) * absorptionOpacity;
    const double eventOpacity = scatteringOpacity + effectiveAbsorptionOpacity;
    const double distanceRandom = CounterRNG::unitOpen(particle.rngKey, particle.rngCounter++);
    const double randomDistance = -Log1p(distanceRandom - 1.0);
    const double scatteringDistance = (eventOpacity > 0.0)? randomDistance / (eventOpacity * dopplerShift) : DBL_MAX;
    const double scatteringTime = (speed > 0.0)? scatteringDistance / speed : DBL_MAX;

    enum Event : std::uint8_t
    {
        IntersectionEvent,
        ScatteringEvent,
        CensusEvent
    };

    Event event = IntersectionEvent;
    double dt = intersection.time;
    if(scatteringTime < dt)
    {
        event = ScatteringEvent;
        dt = scatteringTime;
    }
    if(particle.timeLeft < dt)
    {
        event = CensusEvent;
        dt = particle.timeLeft;
    }

    particle.timeLeft -= dt;
    const double decayRate = absorptionOpacity * fleck * views.speedOfLight;
    const double materialExpFactor = Expm1(-dt * decayRate);
    // In a stationary material frame the two attenuation factors are
    // identical. Reuse the full-precision result instead of evaluating
    // the same transcendental function twice for every transport event.
    const double weightExpFactor = dopplerShift == 1.0 ? materialExpFactor :
                                   Expm1(-dt * decayRate * dopplerShift);
    double integratedEnergy = particle.weight * dt;
    if(Abs(decayRate * dt) >= 1.0e-12)
    {
        integratedEnergy = particle.weight * materialExpFactor * (-1.0 / decayRate);
    }

    particle.location.x += particle.velocity.x * dt;
    particle.location.y += particle.velocity.y * dt;
    particle.location.z += particle.velocity.z * dt;
    if(views.grid.slabTransport)
    {
        gpu::FoldSlabCoordinate(particle.location.y, particle.velocity.y,
                               views.grid.slabLowerY, views.grid.slabUpperY);
        gpu::FoldSlabCoordinate(particle.location.z, particle.velocity.z,
                               views.grid.slabLowerZ, views.grid.slabUpperZ);
    }

    if(views.depositMaterialEnergy)
    {
        STORM_TRANSPORT_ACCUMULATE(views.pendingMaterialEnergy[cellIndex], -materialExpFactor * particle.weight);
    }
    const double inverseC2 = 1.0 / (views.speedOfLight * views.speedOfLight);
    AddMomentum(views, cellIndex,
            -weightExpFactor * particle.weight * particle.velocity.x * inverseC2,
            -weightExpFactor * particle.weight * particle.velocity.y * inverseC2,
            -weightExpFactor * particle.weight * particle.velocity.z * inverseC2);
    STORM_TRANSPORT_ACCUMULATE(views.pendingRadiationEnergy[cellIndex], integratedEnergy);
    opacity.TallyGroupRadiation(particle, views, cellIndex, opacityState, integratedEnergy);

    particle.weight *= 1.0 + weightExpFactor;
    if(Abs(particle.weight) < particle.initialWeight * views.weightCutoffFraction)
    {
        if(views.depositMaterialEnergy)
        {
            STORM_TRANSPORT_ACCUMULATE(views.pendingMaterialEnergy[cellIndex], particle.weight);
        }
        result.step.change = ParticleStatus::REMOVE;
        return result;
    }

    if(event == IntersectionEvent)
    {
        result.directedFace = intersection.directedFace;
        const bool deviceReflect = intersection.boundaryCrossing and views.grid.deviceBoundaryBehaviors != nullptr and
                                    views.grid.deviceBoundaryBehaviors[intersection.directedFace] == static_cast<std::uint8_t>(DeviceBoundaryFaceBehavior::ReflectingRigid);
        if(deviceReflect)
        {
            const auto &normal = views.grid.normals[intersection.directedFace];
            const double normalVelocity = particle.velocity.x * normal.x + particle.velocity.y * normal.y + particle.velocity.z * normal.z;
            particle.velocity.x -= 2.0 * normalVelocity * normal.x;
            particle.velocity.y -= 2.0 * normalVelocity * normal.y;
            particle.velocity.z -= 2.0 * normalVelocity * normal.z;

            // Preserve the established host CrookedPipe semantics: its
            // boundary applies one centre nudge and the manager applies a
            // second after REFLECT.
            constexpr double epsilon = 1.0e-8;
            const auto &center = views.grid.cellCenters[cellIndex];
            for(std::uint8_t nudge = 0; nudge < 2; ++nudge)
            {
                particle.location.x = (1.0 - epsilon) * particle.location.x + epsilon * center.x;
                particle.location.y = (1.0 - epsilon) * particle.location.y + epsilon * center.y;
                particle.location.z = (1.0 - epsilon) * particle.location.z + epsilon * center.z;
            }
            result.step.change = ParticleStatus::NO_CELL_MOVE;
            return result;
        }
        result.step.change = ParticleStatus::CELL_MOVE;
        result.step.nextCellIndex = intersection.nextCellIndex;
        result.step.boundaryCrossing = intersection.boundaryCrossing;
    }
    else if(event == ScatteringEvent)
    {
        const double oldVelocityX = particle.velocity.x;
        const double oldVelocityY = particle.velocity.y;
        const double oldVelocityZ = particle.velocity.z;
        const double eventRandom = CounterRNG::unitOpen(particle.rngKey, particle.rngCounter++) * eventOpacity;
        const bool effectiveScatter = eventRandom >= scatteringOpacity;
        opacity.Scatter(particle, views, cellIndex, opacityState, effectiveScatter, dopplerShift);
        if(views.comovingTransport)
        {
            const double weightBeforeTransform = particle.weight;
            particle.weight *= dopplerShift;
            if(not TransformToLab(particle, views.cellVelocities[cellIndex], views.speedOfLight))
            {
                result.error = TransportError::InvalidDoppler;
                return result;
            }
            AddMomentum(views, cellIndex,
                    (weightBeforeTransform * oldVelocityX - particle.weight * particle.velocity.x) * inverseC2,
                        (weightBeforeTransform * oldVelocityY - particle.weight * particle.velocity.y) * inverseC2,
                        (weightBeforeTransform * oldVelocityZ - particle.weight * particle.velocity.z) * inverseC2);
        }
        else if(views.staticScatterers)
        {
            AddMomentum(views, cellIndex,
                particle.weight * (oldVelocityX - particle.velocity.x) * inverseC2,
                particle.weight * (oldVelocityY - particle.velocity.y) * inverseC2,
                particle.weight * (oldVelocityZ - particle.velocity.z) * inverseC2);
        }
        result.step.change = ParticleStatus::NO_CELL_MOVE;
    }
    else
    {
        result.step.change = ParticleStatus::DONE;
    }

    return result;
}


}}
