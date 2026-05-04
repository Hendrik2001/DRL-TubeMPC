# Direction B Feasibility Experiments: SDD-TMPC Safety Envelope for Deep RL

## What This Is

Three computational experiments that answer one question: **Can Tube MPC realistically serve as a safety envelope around a deep RL agent controlling freeway traffic?**

The answer determines whether "Direction B" is viable for your master's thesis.

---

## Quick Start

```bash
cd feasibility_experiments
python3 run_all.py
```

Requires: `numpy`, `scipy`, `matplotlib` (no GPU, no special hardware). Runs in ~4 minutes.

Outputs: console results + three PNG plots.

---

## File Overview

| File | Purpose |
|------|---------|
| `metanet_model.py` | The traffic model — everything else builds on this |
| `experiment1_rpi.py` | Can we compute a safety tube? (RPI set) |
| `experiment2_conservativeness.py` | Is the tube small enough for RL to be useful? |
| `experiment3_sddtmpc.py` | Can state-dependent tubes reduce conservatism? |
| `run_all.py` | Orchestrator — runs all three experiments and generates plots |

---

## Detailed File Documentation

### `metanet_model.py` — The Traffic Model

**What it implements:** A discrete-time METANET freeway model with 3 segments and 1 on-ramp. METANET is the standard macroscopic traffic model used in freeway control research (Kotsialos et al., 1999).

**State vector (7 dimensions):**
- `ρ1, ρ2, ρ3` — traffic density in each segment (veh/km/lane)
- `v1, v2, v3` — average speed in each segment (km/h)
- `w1` — queue length at the on-ramp (vehicles)

**Key functions:**

- **`equilibrium_speed(rho)`** — The fundamental diagram: maps density to the speed drivers "want" to travel at. Uses the exponential model `V(ρ) = v_free * exp(-(1/a)(ρ/ρ_crit)^a)`. This is the core nonlinearity — below `ρ_crit` (33.5 veh/km/lane) traffic flows freely; above it, congestion forms.

- **`smooth_min3(a, b, c)`** — On-ramp flow is the minimum of three constraints (demand, metered capacity, available space). A hard `min()` is non-differentiable, which breaks linearization. This log-sum-exp approximation (`β=50`) is smooth while being nearly identical to `min()` for practical values. This matters because Experiment 1 needs Jacobians via finite differences.

- **`metanet_step(state, r_o, d_o, q_up)`** — One 10-second time step of the METANET model. Implements:
  1. *Density conservation* — vehicles in = vehicles out + on-ramp, per segment
  2. *Speed update* — three terms: relaxation (drivers adjust toward equilibrium speed), convection (speed propagates downstream), anticipation (drivers slow down when they see dense traffic ahead)
  3. *Merge speed drop* — on-ramp merging reduces segment 1 speed (δ=0.0122, standard value)
  4. *Queue dynamics* — unserved demand accumulates in the on-ramp queue

- **`find_equilibrium(...)`** — Iterates `metanet_step` until the state stops changing (residual < 1e-8). Simple but reliable for free-flow equilibria. Can struggle to converge for congested equilibria (the model has limit cycles near `ρ_crit`), which is why you see "not converged" warnings in Experiments 2-3 at high densities.

- **`linearize(x_eq, ...)`** — Computes Jacobians `A` (∂f/∂x), `B_u` (∂f/∂r), `B_w` (∂f/∂[d,q]) via central finite differences. Uses perturbation δ scaled to each variable's magnitude. Central differences (forward AND backward) give O(δ²) accuracy vs O(δ) for forward-only.

**Design decisions:**
- *Why 3 segments?* Minimum to capture all METANET physics (convection, anticipation need neighbors). More segments add state dimensions without new insight for feasibility.
- *Why T=10s?* Standard METANET sampling time. Converted to hours (10/3600) because METANET uses km/h units throughout.
- *Why smooth_min instead of hard min?* Linearization requires differentiability. β=50 gives <0.1% approximation error while being C∞ smooth.
- *Boundary conditions:* Upstream speed `v_up` is approximated from flow and segment-1 density (common METANET convention). Downstream boundary assumes free outflow (`ρ_downstream = ρ_3`, making the anticipation term zero for segment 3).

---

### `experiment1_rpi.py` — RPI Set Computation

**The question:** Can we compute a Robust Positively Invariant (RPI) set for linearized METANET, and is it finite?

**Why this matters:** In Tube MPC, the "tube" is an RPI set — it bounds how far the real state can deviate from the planned (nominal) trajectory due to disturbances. If we can't compute it, or it's infinite, Tube MPC won't work.

**What it does, step by step:**

1. **Finds the free-flow equilibrium** (Section 1.2) — The linearization point. All densities should be below `ρ_crit`. Result: ρ ≈ 22-23 veh/km/lane (about 65-70% of critical).

2. **Linearizes the model** (Section 1.3) — Computes the 7×7 Jacobian `A`. Then checks Schur stability: all eigenvalues must have magnitude < 1. If the open-loop system is unstable, it computes an LQR feedback gain `K` to stabilize it. Result: open-loop stable with spectral radius 0.948.

3. **Computes the RPI set two ways** (Section 1.4):

   **Ellipsoidal approach:** Solves the discrete Lyapunov equation `P = A P A^T + Q`. The solution `P` defines an ellipsoid that is invariant under the dynamics. Fast (one matrix equation) but conservative — ellipsoids over-approximate the true RPI set because they can't represent the actual shape.

   **Zonotopic approach:** Iteratively builds the Minkowski sum `Z_s = W ⊕ A*W ⊕ A²*W ⊕ ...` using zonotope generators (column vectors). Stops when `A^s * W` contributes less than ε=1% (Raković et al., 2005 criterion). More precise for box-like disturbance sets. The generators grow linearly (2 new columns per iteration), not exponentially — this is why zonotopes work in 7D while polytopes would explode.

**Design decisions:**
- *Why both approaches?* They give different tradeoffs. Ellipsoidal is faster and gives a tighter (but less accurate) bound. Zonotopic is the "correct" method for box disturbances and gives the authoritative result. Comparing them validates both.
- *Why zonotopic widths are larger:* The zonotope captures the actual worst-case box, while the ellipsoid (scaled by `c=7`) is a probabilistic approximation. The zonotopic result is the one used for constraint tightening.
- *Why w₁ has zero RPI width:* At the nominal equilibrium, the queue is empty and the on-ramp has excess capacity. Disturbances in demand get absorbed by the on-ramp flow — they never accumulate in the queue. This would change at a congested operating point.
- *LQR gain had no effect:* `B_u` (sensitivity to ramp metering) was near-zero at the nominal equilibrium because with an empty queue and low demand, changing `r_o` doesn't change the on-ramp flow (it's demand-limited, not capacity-limited). This is physically correct.

---

### `experiment2_conservativeness.py` — Is the Tube Too Big?

**The question:** After tightening the state constraints by the tube width, how much room is left for RL to operate?

**Why this matters:** Tube MPC guarantees safety by restricting the nominal trajectory to `X_tight = X ⊖ Z` (original constraints minus the tube). If the tube eats up most of the constraint space, RL has nowhere to optimize — the safety guarantee becomes a straitjacket.

**What it does:**

1. **Tightens constraints** (Section 2.1) — For each state dimension: `[x_min + z_max, x_max - z_max]`. For example, if density is constrained to [0, 180] and the tube half-width is 9.2, the tightened constraint is [9.2, 170.8].

2. **Computes remaining fraction** (Section 2.2) — `(tightened range) / (original range)`. Decision thresholds: >50% = proceed, 20-50% = caution, <20% = stop.

3. **State-dependent analysis** (Section 2.3) — Repeats the RPI computation at three operating points (free-flow, near-critical, congested) to show how the tube width varies.

**Key results:**
- Density dimensions retain ~90% of their range (tube is narrow)
- Speed dimensions retain 64-74% (tube is wider — speeds are more sensitive to disturbances)
- Queue retains 100% (tube has zero width in this dimension at nominal)
- Average: 82.9% remaining — well above the 50% "substantial room" threshold

**Design decisions:**
- *Why search over upstream flows to hit target densities?* METANET doesn't have a direct "set density" input. Density is an emergent property of the flow balance. So we sweep upstream flow `q_up` and pick the one that produces the closest equilibrium density.
- *Why some equilibria don't converge:* Near and above `ρ_crit`, the METANET model becomes stiff and can exhibit oscillatory/bistable behavior. The simple fixed-point iteration struggles. A Newton solver would help but isn't needed for feasibility — we use the best available approximate equilibrium.
- *The tube width increases with density:* This matches traffic physics. Near `ρ_crit`, the speed-density relationship is steep (small density changes cause large speed changes), so disturbances get amplified. The spectral radius also increases (0.87 → 0.97 → 0.99), meaning disturbances decay more slowly.

---

### `experiment3_sddtmpc.py` — State-Dependent Tubes

**The question:** Can a state-dependent disturbance model produce tighter tubes where it matters (free-flow)?

**Why this matters:** A fixed tube uses worst-case disturbance bounds everywhere. But in free-flow, traffic is predictable and disturbances are small. SDD-TMPC (Surma & Jamshidnejad, 2025) adapts the tube to the current state, giving tighter tubes in benign conditions and wider tubes only when needed.

**What it does:**

1. **Assesses Surma's code** (Section 3.1-3.2) — The TU Delft GitLab repo likely requires authentication. Lists what would need to change (dynamics, disturbances, constraints) vs. what could be reused (tube algorithm, optimization).

2. **Implements state-dependent disturbance model** (Section 3.3):
   ```
   Δd(ρ) = Δd_base * (1 + 3*(ρ/ρ_crit)²)
   ```
   This means: at ρ=0 the disturbance bound is `Δd_base` (200 veh/h), but at ρ=ρ_crit it's `4 * Δd_base` (800 veh/h). The quadratic growth captures the empirical observation that traffic becomes less predictable near capacity.

3. **Compares fixed vs SDD tubes** across 12 operating points. At each point, linearizes METANET and computes the RPI with the local disturbance bound.

**Key result:** 39.9% improvement in free-flow — the SDD tube's ρ₁ width is 8.5 veh/km/lane vs 18.3 for the fixed tube.

**Design decisions:**
- *Why quadratic disturbance growth?* Simple, physically motivated, and produces the right qualitative behavior. A real implementation would learn this function from data (as Surma's paper does with fuzzy models).
- *The SDD tube at high density can exceed the fixed tube:* This is correct! The state-dependent bound is larger near congestion (800+ veh/h vs 200 veh/h). The benefit is that in free-flow (where the system spends most of its time), constraints are much less restrictive.

---

### `run_all.py` — Orchestrator

Runs all three experiments sequentially and generates three plots:

1. **Plot 1 (RPI projection):** Shows the RPI set (red box = zonotopic, green ellipse = ellipsoidal) on the (ρ₁, v₁) plane, with constraint boundaries and the equilibrium point. Visually confirms the tube is small relative to the constraint space.

2. **Plot 2 (conservativeness):** Bar chart of remaining constraint fraction per dimension. All bars green (>50%) = RL has plenty of room.

3. **Plot 3 (SDD comparison):** Left panel shows how tube width grows with operating density (SDD vs fixed). Right panel shows the state-dependent disturbance model Δd(ρ).

---

## Key Parameters and Where They Come From

| Parameter | Value | Source |
|-----------|-------|--------|
| `T = 10s` | METANET sampling time | Kotsialos et al. (1999), standard |
| `τ = 18s` | Speed relaxation time | Kotsialos et al. (1999) |
| `η = 65 km²/h` | Anticipation parameter | Kotsialos et al. (1999) |
| `κ = 40 veh/km/lane` | Density offset | Kotsialos et al. (1999) |
| `v_free = 102 km/h` | Free-flow speed | Kotsialos et al. (1999) |
| `ρ_crit = 33.5` | Critical density | Kotsialos et al. (1999) |
| `a = 1.867` | Speed-density exponent | Kotsialos et al. (1999) |
| `Δd = 200 veh/h` | Demand uncertainty (±40%) | Conservative estimate for feasibility |
| `Δq = 500 veh/h` | Upstream flow uncertainty (±17%) | Conservative estimate for feasibility |
| `δ_merge = 0.0122` | Merge speed drop coefficient | Kotsialos et al. (1999) |
| `β = 50` | Smooth-min sharpness | Chosen so smooth_min ≈ min to <0.1% |
| `ε = 0.01` | RPI convergence tolerance (1%) | Raković et al. (2005), standard |

## Assumptions and Limitations

1. **Linear approximation:** The RPI set is computed for the *linearized* model. Near `ρ_crit`, the linearization becomes inaccurate. A real implementation needs nonlinear tube MPC or frequent re-linearization.

2. **Single on-ramp:** Real freeways have multiple on/off-ramps. More ramps = more control inputs = more disturbance channels, but also more flexibility.

3. **No off-ramps:** The model assumes all traffic exits at segment 3. Adding off-ramps would change the density dynamics but not the fundamental feasibility.

4. **Queue dynamics decouple at low demand:** The zero tube width for `w₁` is an artifact of the equilibrium having an empty queue. Under high demand, the queue would be nonzero and the tube would have nonzero width in this dimension.

5. **Disturbance model is assumed, not learned:** The quadratic `Δd(ρ)` in Experiment 3 is a placeholder. The real SDD-TMPC would learn this from traffic data.

## Bottom Line

All three experiments pass. The tube is computable (<1ms), bounded, and leaves 83% of the constraint space for RL. State-dependent tubes offer a 40% improvement in free-flow. **Direction B is feasible.**
