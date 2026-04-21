# Self-Gravity Reference

Detailed code excerpts and annotated algorithm logic for the RICH self-gravity subsystem.

---

## 1. Entry Point: `GravityAcceleration3D::operator()`

**File**: `source/newtonian/three_dimensional/GravityAcc3D.hpp`

**Note:** The serial path builds `GravityTree` directly rather than using
`SerialGravityCalculator` (which exists but is unused).

```cpp
void operator()(const Tessellation3D& tess, const vector<ComputationalCell3D>& cells,
                const vector<Conserved3D>& fluxes, const double time, vector<Vector3D> &acc) const
{
    // 1. Collect cell CMs and masses
    std::vector<Vector3D> points = tess.GetAllCM();
    points.resize(tess.GetPointNo());
    std::vector<gravity_result_t> masses;
    masses.reserve(points.size());
    for(size_t cellIdx = 0; cellIdx < points.size(); cellIdx++)
        masses.push_back(cells[cellIdx].density * tess.GetVolume(cellIdx));

    std::pair<Vector3D, Vector3D> boundaries = tess.GetBoxCoordinates();
    size_t N = tess.GetPointNo();

    #ifdef RICH_MPI
        // MPI: DistributedGravityCalculator handles tree build + exchange + walk
        DistributedGravityCalculator agent(tess, masses, this->theta, this->quadrupole);
        acc = agent.getAcceleration(points);
    #else
        // Serial: build local tree directly (not via SerialGravityCalculator)
        GravityTree<Vector3D> gravTree(boundaries.first, boundaries.second, this->theta, this->quadrupole);
        std::vector<MassedPoint<Vector3D>> massedPoints;
        for(size_t pointIdx = 0; pointIdx < N; pointIdx++)
            massedPoints.emplace_back(MassedPoint<Vector3D>(points[pointIdx], masses[pointIdx]));
        gravTree.build(massedPoints);
        acc.clear();
        for(size_t pointIdx = 0; pointIdx < N; pointIdx++)
            acc.push_back(gravTree.gravity(points[pointIdx]));
    #endif

    // Scale by gravitational constant
    for(size_t i = 0; i < N; ++i)
        acc[i] *= this->G;
}
```

---

## 2. Tree Build: `GravityTree::build` + `calculateMassHelper`

**File**: `source/3D/gravity/GravityTree.hpp:120-134, 218-277`

### Insert phase

```cpp
bool GravityTree<T>::build(const std::vector<MassedPoint<T>> &points)
{
    for(const MassedPoint<T> &_point : points)
    {
        MassedValue<T> value(_point.point, _point.mass);
        if(!this->octTree->insert(value))
            throw UniversalError("Could not add a point to the gravity tree");
    }
    this->calculateMasses();  // post-order aggregation
    return true;
}
```

`OctTree::insert` splits leaf nodes when two points collide in the same octant, creating intermediate internal nodes until the points separate. Node IDs are assigned incrementally.

### Mass/CM/quadrupole aggregation

Post-order traversal. For each internal node:

```cpp
void GravityTree<T>::calculateMassHelper(Node *node)
{
    if(node == nullptr) return;
    MassedValue<T> &value = node->value;
    std::array<typename MassedValue<T>::coord_type, 6> &Q = value.Q;

    if(!node->isLeaf)
    {
        value.mass = 0;
        value.CM = T();
        for(int i = 0; i < CHILDREN; i++)
        {
            Node *child = node->children[i];
            if(child != nullptr)
            {
                this->calculateMassHelper(child);  // recurse first
                value.CM += childValue.CM * childValue.mass;
                value.mass += childValue.mass;
            }
        }
        value.CM = value.CM / value.mass;  // mass-weighted CM

        // Quadrupole via parallel-axis theorem
        if(this->quadrupole)
        {
            for(int i = 0; i < CHILDREN; i++)
            {
                // For each child, shift its Q to the parent CM frame:
                // Q_parent += Q_child + m_child * (3 * d_i * d_j - delta_ij * r^2)
                // where d = child.CM - parent.CM
                double qx = childValue.CM[0] - value.CM[0];
                double qy = childValue.CM[1] - value.CM[1];
                double qz = childValue.CM[2] - value.CM[2];
                double qr2 = qx*qx + qy*qy + qz*qz;
                Q[0] += childValue.Q[0] + childMass * (3*qx*qx - qr2);  // Qxx
                Q[1] += childValue.Q[1] + 3*childMass * qx*qy;          // Qxy
                Q[2] += childValue.Q[2] + 3*childMass * qx*qz;          // Qxz
                Q[3] += childValue.Q[3] + childMass * (3*qy*qy - qr2);  // Qyy
                Q[4] += childValue.Q[4] + 3*childMass * qz*qy;          // Qyz
            }
            Q[5] = -Q[0] - Q[3];  // Qzz from traceless constraint
        }
    }
}
```

---

## 3. Tree Walk: `GravityTree::gravity`

**File**: `source/3D/gravity/GravityTree.hpp:280-326`

```cpp
T GravityTree<T>::gravity(const T &point, const direction_t *directions) const
{
    T gravity;
    const Node *startingNode = this->octTree->getNodeByDirections(directions);
    this->stack.reserve(this->octTree->getDepth() * CHILDREN);
    stack.push_back({startingNode, startingNode->boundingBox.contains(point)});

    while(!stack.empty())
    {
        const Node *node = stack.back().first;
        bool containsPoint = stack.back().second;
        stack.pop_back();

        if(node == nullptr) continue;

        if(!node->isLeaf and (containsPoint or ShouldOpenBox(point, node->boundingBox,
                              node->value.CM, this->thetaSquared)))
        {
            // OPEN: push children
            int childContains = -1;
            if(containsPoint)
            {
                childContains = node->getChildNumberContaining(point);
                stack.push_back({node->children[childContains], true});
            }
            for(int i = 0; i < CHILDREN; i++)
            {
                if(i == childContains) continue;
                stack.push_back({node->children[i], false});
            }
        }
        else
        {
            // ACCUMULATE: node is far enough or is a leaf
            gravity += CalculateLeafGravityContribution(node->value, point, this->quadrupole);
        }
    }
    return gravity;
}
```

Key optimization: the `containsPoint` flag is tracked per stack entry. When the evaluation point is inside a node, that node is always opened (its child containing the point gets `containsPoint=true`). This avoids the `ShouldOpenBox` distance calculation for ancestors of the evaluation point.

---

## 4. Opening Criterion: `ShouldOpenBox`

**File**: `source/3D/gravity/GravityTree.hpp:16-30`

```cpp
template<typename T, typename BB_T>
bool ShouldOpenBox(const T &point, const BoundingBox<BB_T> &boundingBox,
                   const T &centerOfMass, double thetaSquared)
{
    coord_type width2 = boundingBox.getWidthSquared();
    #ifdef USE_VCL_VECTORIZATION
        Vec4d diff(...);  // VCL SIMD distance
        coord_type distanceToCM2 = ...;
    #else
        coord_type distanceToCM2 = (point[0]-centerOfMass[0])^2 + ... ;
    #endif
    return width2 >= (distanceToCM2 * thetaSquared);
}
```

Opens the box when the node's angular size (width/distance) exceeds theta. Equivalently: `width^2 >= theta^2 * distance^2`.

---

## 5. Force Kernel: `CalculateLeafGravityContribution`

**File**: `source/3D/gravity/GravityTree.hpp:32-67`

```cpp
template<typename Mass_T, typename T>
T CalculateLeafGravityContribution(const Mass_T &nodeValue, const T &point, bool quadrupole)
{
    T gravity;
    const T &CM = nodeValue.CM;
    const T &temp = point - CM;
    gravity_result_t length = abs(temp);
    gravity_result_t length2 = length * length;

    if(length < EPSILON) return gravity;  // self-interaction guard

    // Monopole: F = -m * r_hat / r^2 = -m * (point - CM) / |point - CM|^3
    gravity_result_t sizeOfForce = 1 / (length2 * length);
    gravity -= temp * (sizeOfForce * nodeValue.mass);

    // Quadrupole correction
    if(quadrupole)
    {
        coord_type Qfactor = sizeOfForce / length2;  // 1/r^5
        const coord_type *Q = nodeValue.Q.data();
        coord_type dx = temp[0], dy = temp[1], dz = temp[2];

        // Gradient of quadrupole potential: Q_ij * d_j / r^5
        quadrupoleAddition[0] += Qfactor * (dx*Q[0] + dy*Q[1] + dz*Q[2]);
        quadrupoleAddition[1] += Qfactor * (dx*Q[1] + dy*Q[3] + dz*Q[4]);
        quadrupoleAddition[2] += Qfactor * (dx*Q[2] + dy*Q[4] + dz*Q[5]);

        // -5/2 * (d^T Q d) / r^7 * d  (radial correction term)
        coord_type mrr = dx*dx*Q[0] + dy*dy*Q[3] + dz*dz*Q[5]
                       + 2*dx*dy*Q[1] + 2*dx*dz*Q[2] + 2*dy*dz*Q[4];
        Qfactor *= -5 * mrr / (2 * length2);
        quadrupoleAddition[0] += Qfactor * dx;
        quadrupoleAddition[1] += Qfactor * dy;
        quadrupoleAddition[2] += Qfactor * dz;
    }
    gravity += quadrupoleAddition;
    return gravity;
}
```

The quadrupole force is the gradient of the quadrupole potential:
`F_quad_i = (Q_ij * d_j) / r^5 - (5/2) * (d^T Q d) * d_i / r^7`

---

## 6. MPI: `DistributedGravityCalculator` Constructor

**File**: `source/3D/gravity/DistributedGravityCalculator.hpp:50-108`

```cpp
DistributedGravityCalculator::DistributedGravityCalculator(
    const Tessellation3D &tess_, const std::vector<gravity_result_t> &masses_,
    double theta_, bool quadrupole_ = false,
    const MPI_Comm &comm_ = MPI_COMM_WORLD)
{
    MPI_Comm_size / MPI_Comm_rank;

    // Build local tree from owned cells
    auto [ll, ur] = tess.GetBoxCoordinates();
    GravityTree<Vector3D> *gravTree = new GravityTree<Vector3D>(ll, ur, theta, quadrupole);
    // ... insert MassedPoints from tess.GetCellCM(i), masses[i] ...
    gravTree->build(massedPoints);

    // relevantRanksByDepths[0] = all ranks except self
    // type: vector<boost::container::flat_set<int>>
    relevantRanksByDepths.resize(gravityTree->getOctTree()->getDepth() + 1);
    for(int _rank = 0; _rank < size; _rank++)
        if(_rank != rank) relevantRanksByDepths[0].insert(_rank);

    // Find "real root": skip down single-child octants
    // (since the global bounding box is much larger than this rank's data,
    //  most top-level octants are empty)
    realRootOfGravityTree = gravityTree->getOctTree()->getRoot();
    while(true)
    {
        // count non-null children; if exactly one, descend
        // if zero or multiple, stop
    }

    // Exchange coarse metadata with all ranks
    // returns vector<vector<GravityNodeData>>
    boundingBoxesOfRanks = calculateBoundingBoxesOfRanks(tess);
    // uses MPI_All_cast_by_ranks: each rank broadcasts one GravityNodeData
}
```

---

## 7. MPI: `getSendListHelper` (what to send to each rank)

**File**: `source/3D/gravity/DistributedGravityCalculator.hpp:172-251`

This is the core MPI decision function. It walks the local tree top-down, deciding for each remote rank whether to send a coarse summary or recurse deeper.

```
getSendListHelper(localNode, result, depth):
    if localNode is leaf:
        send localNode's MassedValue to all relevantRanks at this depth
        return

    for each relevantRank at this depth:
        contained = any of rank's bounding boxes fully contains localNode's box?

        if contained:
            radius = localNode.width / theta
            intersecting = tess.getIntersectingRanks(localNode.CM, radius)
            shouldOpen = (rank in intersecting)
        else:
            shouldOpen = ShouldOpenBox(localNode.CM, localNode.box,
                                       remote.closestPoint(localNode.CM), theta^2)

        if shouldOpen:
            add rank to relevantRanks[depth+1]  // will recurse
        else:
            send localNode's MassedValue to rank  // coarse enough

    if any rank wants to open:
        for each child of localNode:
            getSendListHelper(child, result, depth+1)
```

The key insight: we're deciding from the **sender's** perspective what resolution each remote rank needs. If a remote rank's domain is far from a local tree node, a coarse summary suffices. If it's close, we send finer-grained child data.

---

## 8. MPI: `getAcceleration`

**File**: `source/3D/gravity/DistributedGravityCalculator.hpp:136-169`

```cpp
std::vector<Vector3D> DistributedGravityCalculator::getAcceleration(
    const std::vector<Vector3D> &points) const
{
    // Phase 1: Exchange tree data
    std::vector<std::vector<MassedValue<Vector3D>>> sendList = getSendList();
    auto insertToTreeByRanks = MPI_Exchange_all_to_all(sendList, comm);

    // Phase 2: Insert remote data into local tree
    for(int _rank = 0; _rank < size; _rank++)
    {
        if(_rank != rank)
            gravityTree->addExternalValues(insertToTreeByRanks[_rank]);
        insertToTreeByRanks[_rank].clear();
        insertToTreeByRanks[_rank].shrink_to_fit();
    }
    gravityTree->calculateMasses();  // recompute after inserts

    // Phase 3: Local tree walk (now with remote contributions embedded)
    std::vector<Vector3D> results;
    for(const Vector3D &point : points)
        results.emplace_back(gravityTree->gravity(point));
    return results;
}
```

`addExternalValues` inserts each received `MassedValue` as a new leaf at `CM` position with the given mass and Q. The tree walk then naturally includes these remote summaries.

---

## 9. `MassedValue` Serialization

**File**: `source/3D/gravity/MassedValue.hpp:55-77`

```cpp
size_t dump(Serializer *serializer) const override
{
    size_t bytes = 0;
    bytes += this->value.dump(serializer);        // Vector3D (3 doubles = 24 bytes)
    bytes += this->CM.dump(serializer);           // Vector3D (24 bytes)
    bytes += serializer->insert(this->mass);      // double (8 bytes)
    bytes += serializer->insert_array(this->Q);   // 6 doubles (48 bytes)
    return bytes;                                 // total: 104 bytes per node
}

size_t load(const Serializer *serializer, size_t byteOffset) override
{
    // symmetric extraction
}
```

`GravityNodeData` serialization (`DistributedGravityTree.hpp:26-44`) is similar but includes `BoundingBox` (2 x Vector3D = 48 bytes) instead of `value`.

---

## 10. `ConservativeForce3D` -- Momentum/Energy Update

**File**: `source/newtonian/three_dimensional/ConservativeForce3D.cpp:32-83`

```cpp
void ConservativeForce3D::operator()(...) const
{
    size_t N = tess.GetPointNo();
    vector<Vector3D> acc;
    acc_(tess, cells, fluxes, t, acc);  // calls GravityAcceleration3D::operator()

    for(size_t i = 0; i < N; ++i)
    {
        // Timestep suggestion: dt ~ sqrt(width / |acc|)
        double res_temp = fastsqrt(fastabs(acc[i]) / tess.GetWidth(i));
        dt_ = max(dt_, res_temp);

        double volume = tess.GetVolume(i);

        // Momentum update: dp = rho * V * acc * dt
        double Ek = 0.5 * ScalarProd(extensives[i].momentum, extensives[i].momentum)
                    / extensives[i].mass;
        extensives[i].momentum += volume * cells[i].density * acc[i] * dt;

        // Energy update: kinetic energy change (avoids explicit work integral)
        if(mass_flux_ && condition)
            // AREPO eq. 94: includes mass-flux work term
            extensives[i].energy += (part0 + part1) * dt;
        else
            // Default: dE = Ek_new - Ek_old (exact kinetic energy bookkeeping)
            double Eknew = 0.5 * ScalarProd(extensives[i].momentum, extensives[i].momentum)
                           / extensives[i].mass;
            extensives[i].energy += Eknew - Ek;
    }

    // Global max inverse timestep via MPI_Allreduce(MPI_MAXLOC)
    MPI_Allreduce(MPI_IN_PLACE, &max_data, 1, MPI_DOUBLE_INT, MPI_MAXLOC, MPI_COMM_WORLD);
}
```

The energy update uses `Ek_new - Ek_old` rather than `rho * V * v . acc * dt` to avoid energy drift from the explicit Euler momentum update. The `mass_flux_` branch implements eq. 94 from the AREPO paper for cases where mass flux contributes to the gravitational work.

When using the `Simulation` class (newer path), `HydroStep` wraps `HDSim3D`,
so `ConservativeForce3D` runs in the same position within the hydro substep.
Cell ID management (`initializeCellIDs`, `recomputeMaxID`) has moved to
`Simulation` (`source/newtonian/three_dimensional/simulation/Simulation.cpp`).

---

## 11. Monopole Gravity (`MonopoleSelfGravity3D`)

**File**: `source/newtonian/three_dimensional/MonopoleSelfGravity3D.cpp`

Algorithm:
1. Compute global CM and total mass M (local sum + `MPI_Allreduce`).
2. Create uniform radial grid `r_list[i] = i * dr` where `dr = Rmax / (resolution - 1)`.
3. Bin cell masses into radial shells with linear interpolation between adjacent bins.
4. `MPI_Allreduce` the binned mass array.
5. Cumulative sum to get `M(r)` = enclosed mass profile.
6. For each cell: `acc = -M(r) * r_hat / max(r, smoothlength)^2`.

Constructor: `MonopoleSelfGravity3D(resolution, smoothlength)`.

---

## 12. Quadrupole Gravity (`QuadrupoleGravity3D`)

**File**: `source/newtonian/three_dimensional/QuadrupoleGravity3D.cpp`

Extends monopole with l=2 spherical harmonic moments:
- Q20 (axisymmetric), Q21 (real/imag), Q22 (real/imag)
- Inner moments: `Q_in(r) = sum_{r' < r} m(r') * Y_lm(r')`
- Outer moments: `Q_out(r) = sum_{r' > r} m(r') * Y_lm(r') / r'^5`

Algorithm:
1. Same global CM computation as monopole.
2. **Adaptive radial binning** (3 iterations): redistribute bin edges so each bin contains roughly equal mass (via cumulative mass inversion).
3. Bin mass and all 10 moment arrays (5 inner + 5 outer) into radial shells.
4. `MPI_Allreduce` each moment array separately (11 reductions total + mass).
5. Cumulative sum for inner moments (prefix sum), reverse cumulative for outer moments.
6. For each cell: monopole force + quadrupole gradient via finite differences (`CalcQuadGrad`).

The gradient is computed numerically: perturb point by `dx` in each direction, evaluate potential, compute `acc = -dPhi/dx` via central differences. Step size: `dx = 1e-5 * (r[index] - r[index-1])`.

Constructor: `QuadrupoleGravity3D(resolution, smoothlength, output)`. If `output=true`, also computes and stores the gravitational potential per cell.

---

## 13. `OctTree` Node Splitting

**File**: `source/ds/OctTree/OctTree.hpp:455-482`

When `insert` reaches a leaf that already has a value, `splitNode()` is called:

1. The existing leaf becomes a child of a new internal node.
2. The new internal node takes the leaf's place in the parent's children array.
3. The leaf is re-parented and placed in the correct octant of the new internal node.
4. Heights are fixed recursively up to the root.

This process repeats until the new point and the existing point land in different octants. In the worst case (coincident points), the tree reaches `MAX_DEPTH` (64).

Child index calculation (`getChildNumberContaining`): a 3-bit number where bit `(DIM-1-i)` is 1 if `point[i] > center[i]`. For 3D: bit 2 = x comparison, bit 1 = y, bit 0 = z. Example: point in (+x, -y, +z) octant = binary 101 = child 5.

---

## 14. `DistributedGravityTree` (Not Used in Production)

**File**: `source/3D/gravity/DistributedGravityTree.hpp:48-97, 140-192`

This class builds a **rank-partitioned** tree (not an octree) by recursively
splitting the rank range based on an `ownerSplit` parameter. Each internal
`Node` has a `vector<Node*> children` (variable fan-out), and each leaf owns
one MPI rank's data (broadcast via `MPI_Bcast_serializable`). Its `gravity()`
method returns both a partial force (`Vector3D`) and a
`boost::container::flat_set<int>` of ranks whose data needs to be requested
at finer resolution.

It is **not** connected to `GravityAcceleration3D` -- the header is included
by `DistributedGravityCalculator` only for the `GravityNodeData` struct
definition. The actual production code uses `DistributedGravityCalculator`.
