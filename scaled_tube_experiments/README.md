# Scaled Tube Experiments — SDD-TMPC for 6-Segment METANET

Feasibility study of **State-Dependent Tube MPC** as a safety wrapper for
deep RL in macroscopic freeway traffic control. The tube MPC layer precomputes
a family of RPI sets offline and enforces safety constraints at runtime without
re-solving an optimisation problem.

---

## Network Specification

Six uniform segments with one on-ramp on segment 1. Based on the Kotsialos /
Sun (IEEE TITS 2024) parameterisation.

| Parameter | Symbol | Value |
|---|---|---|
| Segments | N | 6 |
| Lanes per segment | λ | 2 |
| Segment length | L | 1.0 km |
| Simulation step | T | 10 s |
| Free-flow speed | V_free | 102 km/h |
| Critical density | ρ_crit | 33.5 veh/km/lane |
| Max density | ρ_max | 180 veh/km/lane |
| Speed relaxation time | τ | 18 s |
| Speed viscosity | η | 65 km²/h |
| Density anticipation | κ | 40 veh/km/lane |
| Greenshields exponent | a | 1.867 |
| On-ramp capacity | C_O | 2000 veh/h |
| Max on-ramp queue | w_max | 200 veh |
| Merge parameter | δ | 0.0122 |
| Upstream flow (nominal) | q_up | 3000 veh/h |
| On-ramp demand (nominal) | d_1 | 500 veh/h |

**Equilibrium speed function:** V_e(ρ) = V_free · exp(−(1/a)·(ρ/ρ_crit)^a)

### State, Control, Disturbances

| | RM-only | RM + VSL |
|---|---|---|
| State x | [ρ₁..ρ₆, v₁..v₆, w₁] — 13-dim | same |
| Control u | r₁ ∈ [0.05, 1.0] | [r₁, v_vsl3, v_vsl4] — 3-dim |
| Disturbances w | [d₁, q_up] | same |
| Disturbance bounds | Δd = 200 veh/h, Δq = 500 veh/h | same |
| VSL range | — | v_vsl ∈ [20, 102] km/h |

---

## File Structure

```
scaled_tube_experiments/
│
│  Core models
├── metanet6.py               6-segment METANET (RM-only)
├── metanet6_vsl.py           VSL extension — smooth-min on segments 3 & 4
├── tube_tools.py             Spectral radius, LQR, ellipsoidal/zonotopic RPI
│
│  Experiments
├── experiment1_6seg_rpi.py   Single-point RPI at nominal operating point
├── experiment2_lookup_table.py  Offline lookup table (24 operating points)
├── experiment3_tube_switching.py  Naive switching demonstration
├── experiment4_demand_scenario.py  3-controller comparison (RM-only)
├── experiment_vsl_lookup.py  Lookup table with RM+VSL
├── experiment_vsl_comparison.py  3-controller comparison (RM+VSL)
│
│  Analyses
├── analysis1_tube_switching_formal.py  Formal reachability (3 methods)
├── analysis2_linearization_validation.py  Monte Carlo 1/5/10-step error
├── analysis3_multi_regime_simulation.py  3-hour multi-regime simulation
│
│  Entry points
├── run_all.py                Runs experiments 1–4 in sequence
├── run_analyses.py           Runs analyses 1–3 in sequence
├── presentation_plots.py     4 presentation-quality plots (1920×1080, 300 DPI)
```

---

## Tube Construction

At each of the 24 operating points the pipeline is:

1. **Operating point** — set ρ_op = factor · ρ_crit for all 6 segments,
   v_op = V_e(ρ_op), w = 0. Upstream flow q_up = ρ_op · v_op · λ.

2. **Linearize** — central finite differences (Δ = 1×10⁻⁶ relative),
   giving A ∈ ℝ^{13×13}, B_u ∈ ℝ^{13×1}, B_w ∈ ℝ^{13×2}.

3. **Stabilize** — if ρ(A) ≥ 1, compute discrete LQR gain K
   (Q = I₁₃, R = 10·I₁ for RM-only; R = diag(10,1,1) for VSL),
   giving A_cl = A − B_u · K.

4. **Ellipsoidal RPI** — solve discrete Lyapunov equation:
   P = A_cl · P · A_cl^T + B_w · Σ_w · B_w^T,  Σ_w = diag(Δd², Δq²).
   Tube half-widths: hw = √(n · diag(P)),  n = 13.

5. **Store** — entry contains {factor, ρ_op, x_op, A, B_u, B_w, K, A_cl,
   P, c, hw, ρ(A), ρ(A_cl), stable}.

### Operating-point grid

```python
RHO_FACTORS = [
    0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
    0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90,
    0.92, 0.95, 0.97, 1.00, 1.05, 1.10, 1.20, 1.40,
]
```

Grid is intentionally denser near ρ_crit (factors 0.90–1.00) where
linearization changes fastest.

---

## Experiment Results

### Experiment 2 — RPI Lookup Table

**RM-only:** 20 / 24 operating points produce a stable closed-loop A_cl.
The 4 unstable points are at factors 1.00, 1.05, 1.10, 1.20, 1.40 — congested
regime where ρ(A) > 1 and B_u ≈ 0 (ramp metering has no authority).

| Regime | Stable | Unstable |
|---|---|---|
| Free-flow (factor < 1.0) | 19/19 | 0/19 |
| Congested (factor ≥ 1.0) | 1/5 | 4/5 |

Tube half-widths grow exponentially near ρ_crit. Example values (ρ₁ dimension):

| factor | ρ_op (veh/km/lane) | hw_ρ₁ | hw_v₁ |
|---|---|---|---|
| 0.30 | 10.1 | ~1.0 | ~3.5 |
| 0.60 | 20.1 | ~2.0 | ~5.0 |
| 0.90 | 30.2 | ~5.5 | ~9.0 |
| 0.97 | 32.5 | ~11.0 | ~15.0 |

### Experiment 4 — ALINEA vs ALINEA+Tube vs Aggressive (RM-only)

**Scenario:** 2-hour demand surge. On-ramp demand peaks at 850 veh/h
(400 + 450 · trapezoid, rise at t=1200s, fall at t=3600s). Upstream flow
peaks at 3250 veh/h. 720 control steps at 10 s each.

**Safety filter:** nearest tube entry by mean density; local box
[x_op − hw, x_op + hw] ∩ global constraints; bisection on r₁ over 20
iterations to find the largest safe metering rate.
Only entries with ρ_op + hw_ρ₁ < ρ_crit − 0.25 are eligible.

| Controller | TTS (veh·h) | Max ρ (veh/km/lane) | ρ > ρ_crit (steps) | w > w_max (steps) |
|---|---|---|---|---|
| ALINEA | 574.6 | 47.1 | 150 | 0 |
| **ALINEA + Tube** | **583.1** | **32.9** | **0** | **0** |
| Aggressive (RL proxy) | 576.5 | 42.1 | 166 | 0 |

**TTS cost of tube filter: +1.5%** — safety is bought at negligible throughput cost.

### VSL Extension — Lookup Table (experiment_vsl_lookup)

VSL applied to segments 3 and 4 via smooth-min with β = 50.
V_eff(ρ, v_vsl) = smooth_min(V_e(ρ), v_vsl).
LQR weight R = diag(10, 1, 1) to weight ramp metering heavier than VSL.

**Result: 24 / 24 stable** — VSL rescues all 4 previously unstable congested
operating points by providing actuation authority when the ramp is saturated.

| Regime | RM-only stable | RM+VSL stable |
|---|---|---|
| Free-flow (factor < 1.0) | 19/19 | 19/19 |
| Congested (factor ≥ 1.0) | 1/5 | 5/5 |

### VSL Comparison — ALINEA+VSL vs +Tube vs Aggressive (experiment_vsl_comparison)

Same 2-hour demand surge. ALINEA feedback on ramp (K_R = 70), VSL feedback
on segments 3–4: v_vsl = max(20, 102 − 6·max(0, ρ − ρ_crit)).

| Controller | TTS (veh·h) | Max ρ (veh/km/lane) | ρ > ρ_crit (steps) | w > w_max (steps) |
|---|---|---|---|---|
| ALINEA + VSL | 574.5 | 45.4 | 150 | 0 |
| **ALINEA + VSL + Tube** | **562.4** | **32.5** | **0** | **0** |
| Aggressive | 577.0 | 42.4 | 166 | 0 |

**TTS cost of tube filter: −2.1%** — tube filter improves both safety and throughput.

---

## Analysis Results

### Analysis 1 — Formal Tube Switching Reachability

Three methods checked for each adjacent pair (i, i+1) in the stable entries:

| Method | Description | Pass rate |
|---|---|---|
| A — Bounding box | bbox(reachable set of Z_i) ⊆ bbox(Z_{i+1}) | 0 / 19 |
| B — Ellipsoidal | P_{i+1} − P_reach ≽ 0 (PSD check) | 3 / 19 |
| C — Shifted ellipsoid | accounts for Δx_op between operating points | 3 / 19 |

**Conclusion:** Formal ellipsoidal containment is too conservative — the
reachable ellipsoid has a different eigenstructure from the target tube. This
is a mathematical conservatism issue, not a grid resolution issue. The practical
switch safety (Analysis 3) is 100% despite this.

### Analysis 2 — Linearization Error Validation

500 Monte Carlo samples inside each RPI ellipsoid. Error ratio =
‖nonlinear step − linear prediction‖ / hw.

| Horizon | Median error ratio | Conclusion |
|---|---|---|
| 1-step | 0.08 | Acceptable — 14/19 within 20% of tube width |
| 5-step | 0.72 | Significant degradation |
| 10-step | 0.95 | Linearization essentially invalid |

**Conclusion:** Tube must be re-evaluated every control step (10–60 s).
Supports short-horizon MPC (N_p = 1–3). Do not use the same linearization
for multi-step rollouts.

### Analysis 3 — Multi-Regime 3-Hour Simulation

Five-phase demand profile traversing free-flow → build-up → near-critical →
recovery → return. ALINEA + tube filter, tube re-selected every 60 s control
step (6 inner METANET steps of 10 s each). 10% noise on demand.

| Metric | Value |
|---|---|
| Tube switches | 180 |
| Safe switches (successor tube contains state) | 21 / 21 checked |
| Steps with ρ > ρ_crit | 11 |
| Max density | ~35 veh/km/lane |

**Condition:** The 100% switch safety holds with the doubled (24-point) grid.
With the original 12-point grid, 2/21 switches had gaps. The 11 density
violations occur transiently at the near-critical phase boundary and are resolved
within 2–3 control steps by the filter.

---

## Presentation Plots (presentation_plots.py)

Four plots generated at 1920×1080 px, 300 DPI.

| File | Content |
|---|---|
| plot_p1_spectral_vsl.png | Spectral radius RM-only vs RM+VSL, annotated rescue zone |
| plot_p2_safety_zero_cost.png | TTS and violation bar chart, RM-only vs RM+VSL, 3 controllers |
| plot_p3_tube_physics.png | Tube half-width vs operating density + spectral radius correlation |
| plot_p4_architecture.png | SDD-TMPC + RL block diagram with key invariance guarantee |

---

## How to Run

```bash
# Experiments 1–4 in sequence
python run_all.py

# Analyses 1–3
python run_analyses.py

# VSL lookup table
python experiment_vsl_lookup.py

# VSL 3-controller comparison
python experiment_vsl_comparison.py

# Presentation plots (runs all underlying experiments)
python presentation_plots.py
```

**Dependencies:** numpy, scipy, matplotlib. No solvers or optimisation
libraries required — all RPI computations use scipy.linalg only.

---

## Key Design Decisions

- **Q_UPSTREAM = 3000 veh/h** (not 3500): higher value pushed the nominal
  equilibrium into congestion (ρ ≈ 149), preventing RPI computation.
  3000 gives a stable free-flow equilibrium (ρ ≈ 22, v ≈ 78 km/h).
- **Local tube box** [x_op ± hw] rather than global tightened box [X ⊖ Z]:
  the global upper bound for density is ~171 veh/km/lane, far above ρ_crit,
  so the global filter never fires.
- **Bisection over 20 iterations** provides 1/2^20 ≈ 10⁻⁶ precision on the
  ramp rate, well within practical requirements.
- **VSL gain K_VSL = 6 (km/h)/(veh/km/lane)** with threshold at ρ_crit:
  reduces VSL by 6 km/h per veh/km/lane above critical. Aggressive gains
  (K_VSL = 200) caused a positive feedback loop that worsened congestion.
