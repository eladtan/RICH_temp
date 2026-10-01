# Positivity-preserving velocity term in the gray diffusion matrix — design for approval (2026-09-30)

Designed with gpt-6-astra (high); inputs: /home/elads/mpi_tmp/claude-5014/-home-elads-RICH-ablation-integration/8b3f6501-5576-40d5-ae43-ef71993c29ce/scratchpad/momentum_positivity_design_v1.md, review /home/elads/mpi_tmp/claude-5014/-home-elads-RICH-ablation-integration/8b3f6501-5576-40d5-ae43-ef71993c29ce/scratchpad/astra_mompos.md, final /home/elads/mpi_tmp/claude-5014/-home-elads-RICH-ablation-integration/8b3f6501-5576-40d5-ae43-ef71993c29ce/scratchpad/astra_mompos2.md. Evidence: D2/D3/D4 (jobs 10233490, 10233500, 10233558). Status: awaiting the author's approval; nothing implemented.

**1. Scope and motivation**

Add `RICH_RADIATION_MOMENTUM_POSITIVITY=1`, **default off**. Enable the new discretization only for gray `Diffusion` with `hydro_on_ == true`, on interior faces.

The supplied job 10233558 measurement identifies pressure work as the dominant positive coupling: \(\alpha\approx0.013\) even at \(f=1\), so the original bracket remains near \(-1\). Small Fleck factors primarily weaken the absorption diagonal. Long individual radiation intervals aggravate both effects. This design changes neither the Fleck formula nor the scheduler.

Flag-on also fixes interior PostCG face timesteps on **every eligible row**, including rows requiring no lumping, and removes the uncompensated kinetic-energy undo. Boundary face formulas remain unchanged; hydro-off, multigroup, and flag-off retain their current paths.

**2. Discretization and minimal lumping**

Use physical units below. Let \(E_i\) be radiation energy density, \(S_{ij}\) face area, \(\phi_i\) `cell_flux_limiter`, \(\chi_i\) `v_ratio`, and
\[
\mathbf r_{ij}=\frac{\mathbf x_i-\mathbf x_j}{|\mathbf x_i-\mathbf x_j|},
\qquad
\alpha_i=\frac{6f_i\chi_i\sigma_{P,i}D_i}{c},
\qquad
k_{ij}=\frac{\Delta t_{ij}\phi_iS_{ij}}3
                  \mathbf v_i\cdot\mathbf r_{ij}.
\]
Here \(\Delta t_{ij}\) is exactly the assembled `individualFaceTimeStep`, including candidate fraction and existing endpoint/fallback rules.

The velocity contribution is
\[
T_{ij}=\tfrac12(1-\alpha_i)k_{ij},
\qquad
Q_{ij}=T_{ij}(E_i+E_j).
\]
Because \(\mathbf r_{ij}\) points inward, its common-timestep continuum interpretation is
\[
-\Delta t\,V_i\frac{\phi_i}{3}(1-\alpha_i)
       \mathbf v_i\cdot\nabla E.
\]
For unequal timesteps, retain the face sum.

After **all contributions to each interior neighbor column are assembled**, define
\[
a_{ij}=d_{ij}+T_{ij},\qquad e_{ij}=\max(a_{ij},0),
\]
and apply
\[
a'_{ij}=a_{ij}-e_{ij},\qquad
A'_{ii}=A_{ii}+\sum_j e_{ij}.
\]
Use the actual assembled diffusion contribution \(d_{ij}\), including geometric projection and special branches. Do not replace it with a nominal \(D/\Delta x\) estimate or conceal invalid positive diffusion coefficients.

For a unique neighbor face,
\[
w_{ij}=
\begin{cases}
e_{ij}/T_{ij},&e_{ij}>0,\\
0,&e_{ij}=0,
\end{cases}
\qquad
E'_{f,ij}=\tfrac12[(1+w_{ij})E_i+(1-w_{ij})E_j].
\]
Require finite coefficients, \(d_{ij}\le0\), and \(0\le w_{ij}\le1\), allowing only quantified roundoff. Invalid geometry or coefficients reject the candidate.

Then
\[
Q'_{ij}-Q_{ij}=e_{ij}(E_i-E_j).
\]
This is the minimum correction in this family; constant-field row action is preserved.

**Repeated neighbor columns:** decide from their aggregate coefficient, never a transient positive slot. For geometric slots \(s\) sharing column \(j\), set
\[
P_{ij}=\sum_{s\to j}\max(T_{is},0),\qquad
w_{is}=
\begin{cases}
e_{ij}/P_{ij},&T_{is}>0,\ e_{ij}>0,\\
0,&\text{otherwise}.
\end{cases}
\]
Thus \(\sum_sT_{is}w_{is}=e_{ij}\). Require \(0<e_{ij}\le P_{ij}\). Preserve diffusion transport accounting independently of this column aggregation.

**3. Stored data and exact gas–radiation exchange**

Store candidate-local weights by **owned cell and `GetCellFaces(i)` geometric slot**, never sparse-matrix position. Retain the assembled face timestep and matrix velocity needed by PostCG; reuse existing coefficient arrays for \(f,\chi,D,\phi,\sigma_P\). Check face/neighbor correspondence. Rebuild on every matrix build and retry; invalidate across mesh reconstruction or repartitioning.

For interior faces compute, from the source-evaluation solution vector,
\[
\Delta\mathbf p_i^{I}
 =\sum_{j\in I}\frac{\Delta t_{ij}\phi_iS_{ij}}3
                         E'_{f,ij}\mathbf r_{ij},
\qquad
W'_i=\sum_{j\in I}k_{ij}E'_{f,ij}.
\]

The two matrix contributions have different partners:

| Matrix contribution | Radiation change | Gas partner |
|---|---:|---:|
| Pressure work, original bracket’s “\(-1\)” | \(-W'_i\) | Momentum-induced kinetic change |
| Relativity | \(+\alpha_iW'_i\) | Internal energy \(-\alpha_iW'_i\) |

For an interior-only row, apply
\[
\Delta U_{\mathrm{gas,rel}}=-\alpha_iW'_i,\qquad
\Delta K_i=\frac{|\mathbf p_i+\Delta\mathbf p_i|^2-|\mathbf p_i|^2}{2m_i},
\qquad
\boxed{\Delta U_{\mathrm{rad,kin}}=W'_i-\Delta K_i.}
\]
Consequently,
\[
(-W'_i+\alpha_iW'_i)+(W'_i-\Delta K_i)
-\alpha_iW'_i+\Delta K_i=0.
\]
Convert physical energies by \(M_sL_s^2/t_s^2\) and momentum by \(M_sL_s/t_s\) before updating `extensives`.

Use the matrix velocity in \(W'_i\). Substituting \((\mathbf p_i/m_i)\cdot\Delta\mathbf p_i\) is valid only when primitive and conserved velocities agree. Measure that mismatch; the explicit \(W'_i\) formula remains valid when they differ.

PostCG therefore:

- Uses \(E'_f\) and the **assembled face timestep** for both interior relativity exchange and force impulse, including \(w=0\) faces. Cell-local absorption/emission retains its cell timestep.
- Leaves the old-state gradient used for flux limiters and `R2` unchanged.
- Keeps the commented pressure-work addition to internal energy disabled.
- Recomputes gas total energy from updated internal plus kinetic energy.
- Removes the kinetic-debit undo. Any inadmissible final candidate is collectively rejected, restored, and retried.

For rows touching boundaries, preserve existing boundary impulse \(\Delta\mathbf p^B\) and thermal contributions. Use the combined impulse when calculating \(\Delta K\), including cross terms:
\[
\Delta U_{\mathrm{rad,kin}}
 =W'_i+\mathbf v_{\mathrm{conserved}}\cdot\Delta\mathbf p_i^B
       -\Delta K_i.
\]
This retains the existing boundary work term while correcting interior work. The approved **cell-level undo removal still applies**. Boundary exchange defects remain separately reported; no exact boundary-conservation claim is made.

**4. Code placement and solver contracts**

- [Diffusion.cpp](/home/elads/RICH-ablation-integration/source/Radiation/Diffusion.cpp:1402): finish interior assembly, aggregate columns, lump, populate face data, and check matrix admissibility before returning `BuildMatrix`.
- [Diffusion.hpp](/home/elads/RICH-ablation-integration/source/Radiation/Diffusion.hpp:287): add only the flag and required candidate-local face/state storage.
- [PostCG](/home/elads/RICH-ablation-integration/source/Radiation/Diffusion.cpp:1514): implement the exchanges above. The current undo is at lines 1709–1710 in this tree.
- [RadiationDriver.cpp](/home/elads/RICH-ablation-integration/source/Radiation/RadiationDriver.cpp:7149): preserve both serial and distributed reduction contracts and collective rollback.

Lumping precedes frozen-column elimination:
\[
b_{\mathrm{verify}}=b-A'_{ap}E_p,\qquad
A'_{aa}\delta=b_{\mathrm{verify}}-A'_{aa}E_{0,a}.
\]
Require nonnegative **physical verification RHS**, not correction RHS. Active initial guesses affect convergence; frozen `full_initial` values define prescribed physical endpoints.

Keep `recordIndividualFaceCoefficient` restricted to physical diffusion transport. Lumping is paired with local gas work, not a new passive-cell radiation transfer. Preserve existing frozen-passive defect accounting.

Weights are directed owned-row data; opposite ranks need not use equal weights. Synchronize endpoint primitives and solutions through existing canonical mappings. Never pass compact partial-mesh arrays to canonical-index exchanges. Ranks with no active rows still participate in collective acceptance.

Preserve PostCG’s two-vector contract: global mode evaluates sources/faces using the raw vector and initializes radiation from the applied/corrected vector; individual mode currently passes the same final vector twice. Account separately for residual correction, limiting, and floors.

**5. Candidate gates and diagnostics**

Require finite positive diagonals, nonpositive assembled interior off-diagonals, and strictly positive row sums for the sufficient M-matrix certificate. Zero margins require a separate nonsingularity argument; v1 rejects uncertified eligible candidates. Lumping preserves row sums, so it cannot repair a bad margin.

Also check physical RHS, true residual, solved radiation, and final radiation/internal/total energies using existing admissibility requirements. Solver convergence alone is insufficient.

Emit one rank-0 aggregate `RICH_RADIATION_MOMENTUM_POSITIVITY` record per candidate, accepted or rejected:

- Candidate fraction, face-timestep range, retry depth/reason; changed rows/faces; maximum \(w\) and excess/diagonal ratio.
- Remaining positive couplings, minimum diagonal/row margin/verification RHS; boundary-face and hydro-off exclusions.
- Signed and absolute \(\delta Q=\sum e_{ij}(E_i-E_j)\), pressure work, relativity exchange, kinetic correction, and maximum local exchange-closure error.
- Separate residual-correction, floor, boundary, and frozen-passive energy ledgers; primitive/conserved velocity mismatch; acceptance status and elapsed time.

Detailed failure records retain integer stable cell/neighbor IDs, coefficients, and timesteps. Excluded paths are labeled as such, never reported as positivity-certified.

**6. Tests and runtime acceptance — all pending**

Required focused tests:

- Flag-off equivalence; unchanged hydro-off/MG/boundary face formulas; all boundary types with nonzero normal velocity.
- Analytic small matrices: minimal excess, duplicate columns, constant-field action, nonnegative exact solution, and \(w=0\) cases.
- Independent exchange accounting with nonunit scales, unequal timesteps, velocity mismatch, quadratic kinetic energy, and mixed interior/boundary impulses.
- Both reduced RHS identities; frozen endpoints; raw versus corrected vectors; limited residual corrections and floors.
- Serial and 2-/4-rank MPI; partial meshes, reordered ownership, zero-active ranks, retries and cache refresh.
- Negative margins/RHS, invalid geometry, and kinetic-debit rejection with complete transactional restoration.

**Proposed numerical acceptance limits:** local algebraic exchange closure within \(256\epsilon_{\mathrm{mach}}S_i\), where \(S_i\) includes absolute exchange terms and old/new kinetic energies. Independently assembled manufactured cases must meet that bound without a production solver allowance.

For production, report
\[
R_E=\Delta(E_{\mathrm{gas}}+E_{\mathrm{rad}})
 -E_{\mathrm{external}}
 -E_{\mathrm{boundary\ defect}}
 -E_{\mathrm{frozen\ defect}}
 -E_{\mathrm{solver/floor}},
\]
with every ledger computed independently. Proposed unexplained-closure limits: \(10^{-10}E_{\rm ref}\) per candidate and \(10^{-8}E_{\rm ref}\) cumulatively, plus explicitly quantified summation roundoff; \(E_{\rm ref}\) is event-start gas-plus-radiation energy. Local gates prevent global normalization from hiding dim-cell failures.

Preserve current frozen-passive limits and their existing normalization: local \(10^{-2}\) with \(10^{-9}\) absolute allowance, event target \(10^{-6}\), cumulative signed \(10^{-4}\), cumulative absolute \(10^{-3}\). Do not relax positivity-floor or solver gates.

After independent implementation review, user-run builds and separately authorized runtime tests:

1. Matched bounded late-TDE comparisons, including individual and global flag-on/off controls.
2. Require zero accepted eligible positive couplings or unexplained energy violations; target at least **90% fewer pressure-work negativity retries**, with no increase in total retries per simulated interval.
3. Require lower total wall time for the affected window; proposed overhead ceiling **5%** in a matched no-lumping control. Include retries and diagnostics in timing.
4. Only after these pass, validate the proposed `snap70 → t=50` endpoint.

**Remaining risks:** directional lumping changes force accuracy; constant-field preservation does not establish convergence order. Boundary defects, invalid source diagonals, and kinetic depletion can still cause rejection. Small Fleck factors remain physically unchanged. No unresolved author decision blocks implementation; runtime benefit and numerical accuracy remain acceptance gates. Implementation completion requires an independent reviewer and resolution of all findings.
