"""
EXPERIMENT 1: Quantify the Quadratic Cost Mismatch

Goal: Prove empirically that the true cost-to-go near ρ_crit has an asymmetric
"hockey stick" shape that a quadratic function cannot capture.

Physics of the hockey stick:
- Below ρ_crit: traffic flows freely, inflow ≈ outflow, TTS is low and stable
- Above ρ_crit: capacity drop → outflow < inflow → density GROWS → more congestion
  → positive feedback loop → TTS compounds super-linearly over time
- This requires: (1) demand near capacity, (2) long enough horizon to see compounding
"""
import numpy as np
import time
from metanet_model import (
    equilibrium_speed, metanet_step,
    T, L, LAMBDA, V_FREE, RHO_CRIT, RHO_MAX, C_O, W_MAX
)


def run_experiment1(n_mc=50, n_steps=600, verbose=True):
    """
    Compute true cost-to-go via Monte Carlo at density grid points.

    Uses high upstream demand (near capacity) + 600 steps (100 min horizon).
    """
    t0 = time.time()

    # Demand near capacity to trigger congestion compounding above ρ_crit
    # Capacity ≈ V_free * ρ_crit * λ ≈ 6834 veh/h
    q_up_sim = 6200   # ~91% capacity — congestion above ρ_crit is persistent
    d_on_sim = 1000   # strong on-ramp demand

    if verbose:
        cap = V_FREE * RHO_CRIT * LAMBDA
        print(f"Mainline capacity: {cap:.0f} veh/h")
        print(f"Total demand: {q_up_sim + d_on_sim} veh/h ({(q_up_sim+d_on_sim)/cap*100:.0f}% of capacity)")
        print(f"Simulation: {n_steps} steps × {T*3600:.0f}s = {n_steps*T*3600/60:.0f} min")

    # Density grid: 0.3–2.0 × ρ_crit, dense near ρ_crit
    rho_low = 0.3 * RHO_CRIT
    rho_high = 2.0 * RHO_CRIT
    rho_uniform = np.linspace(rho_low, rho_high, 40)
    rho_near_crit = np.linspace(0.8 * RHO_CRIT, 1.3 * RHO_CRIT, 30)
    rho_grid = np.sort(np.unique(np.concatenate([rho_uniform, rho_near_crit])))

    if verbose:
        print(f"Density grid: {len(rho_grid)} points, ρ ∈ [{rho_grid[0]:.1f}, {rho_grid[-1]:.1f}]")

    rho_norm_all = []
    cost_means = []
    cost_stds = []
    all_states_7d = []

    for idx, rho in enumerate(rho_grid):
        v_eq = equilibrium_speed(rho)
        state0 = np.array([rho, rho, rho, v_eq, v_eq, v_eq, 0.0])

        costs = []
        for trial in range(n_mc):
            # Demand noise
            noise_d = np.random.uniform(-1, 1, n_steps)
            noise_q = np.random.uniform(-1, 1, n_steps)
            d_seq = d_on_sim * (1 + 0.4 * noise_d)
            q_seq = q_up_sim * (1 + 0.15 * noise_q)

            r_o = 0.7  # fixed moderate metering
            state = state0.copy()
            tts = 0.0

            for k in range(n_steps):
                # TTS: total vehicle-hours per step
                tts += T * L * LAMBDA * (state[0] + state[1] + state[2]) + T * state[6]
                state = metanet_step(state, r_o, d_seq[k], q_seq[k])
                state[6] = np.clip(state[6], 0, W_MAX)
                state[:3] = np.clip(state[:3], 0.01, RHO_MAX)
                state[3:6] = np.clip(state[3:6], 0.01, V_FREE * 1.5)

            costs.append(tts)

        rho_norm_all.append(rho / RHO_CRIT)
        cost_means.append(np.mean(costs))
        cost_stds.append(np.std(costs))
        all_states_7d.append(state0)

        if verbose and (idx + 1) % 15 == 0:
            print(f"  {idx+1}/{len(rho_grid)}: ρ/ρ_c={rho/RHO_CRIT:.2f}, "
                  f"cost={np.mean(costs):.2f} ± {np.std(costs):.2f}")

    rho_norm_all = np.array(rho_norm_all)
    cost_means = np.array(cost_means)
    cost_stds = np.array(cost_stds)
    all_states_7d = np.array(all_states_7d)

    # ---- Fit quadratic (7D) ----
    x_ref_rho = 0.7 * RHO_CRIT
    v_ref = equilibrium_speed(x_ref_rho)
    x_ref = np.array([x_ref_rho]*3 + [v_ref]*3 + [0.0])
    X_centered = all_states_7d - x_ref

    features = []
    for x in X_centered:
        f = []
        for i in range(7):
            for j in range(i, 7):
                f.append(x[i] * x[j])
        for i in range(7):
            f.append(x[i])
        f.append(1.0)
        features.append(f)
    features = np.array(features)

    coeffs, _, _, _ = np.linalg.lstsq(features, cost_means, rcond=None)
    quad_pred = features @ coeffs

    ss_tot = np.sum((cost_means - np.mean(cost_means)) ** 2)
    r2_quad = 1 - np.sum((cost_means - quad_pred)**2) / ss_tot

    # 1D quadratic (for plot)
    rho_c = rho_norm_all - 0.7
    A_1d = np.column_stack([rho_c**2, rho_c, np.ones(len(rho_c))])
    coeffs_1d, _, _, _ = np.linalg.lstsq(A_1d, cost_means, rcond=None)
    quad_pred_1d = A_1d @ coeffs_1d
    r2_quad_1d = 1 - np.sum((cost_means - quad_pred_1d)**2) / ss_tot

    # Residuals
    abs_res = np.abs(cost_means - quad_pred)
    rel_res = abs_res / np.maximum(cost_means, 1e-6) * 100

    near_mask = (rho_norm_all > 0.85) & (rho_norm_all < 1.15)
    max_res_crit = np.max(rel_res[near_mask]) if np.any(near_mask) else 0.0
    mean_res_crit = np.mean(rel_res[near_mask]) if np.any(near_mask) else 0.0

    # Asymmetry: slope below vs above ρ_crit
    below = (rho_norm_all > 0.4) & (rho_norm_all < 0.95)
    above = (rho_norm_all > 1.05) & (rho_norm_all < 1.8)

    if np.sum(below) > 2 and np.sum(above) > 2:
        p_below = np.polyfit(rho_norm_all[below], cost_means[below], 1)
        p_above = np.polyfit(rho_norm_all[above], cost_means[above], 1)
        asymmetry_ratio = abs(p_above[0]) / max(abs(p_below[0]), 1e-6)
    else:
        asymmetry_ratio = 1.0

    is_asymmetric = asymmetry_ratio > 1.5

    elapsed = time.time() - t0

    results = {
        'rho_norm': rho_norm_all,
        'cost_means': cost_means,
        'cost_stds': cost_stds,
        'quad_pred_7d': quad_pred,
        'quad_pred_1d': quad_pred_1d,
        'r2_quad_7d': r2_quad,
        'r2_quad_1d': r2_quad_1d,
        'max_residual_near_crit': max_res_crit,
        'mean_residual_near_crit': mean_res_crit,
        'asymmetry_ratio': asymmetry_ratio,
        'is_asymmetric': is_asymmetric,
        'rel_residuals': rel_res,
        'abs_residuals': abs_res,
        'coeffs_1d': coeffs_1d,
        'elapsed': elapsed,
        'all_states_7d': all_states_7d,
    }

    if verbose:
        print(f"\n{'='*60}")
        print("EXPERIMENT 1 RESULTS: Quadratic Cost Mismatch")
        print(f"{'='*60}")
        print(f"  Computation time: {elapsed:.1f}s")
        print(f"  Cost range: [{cost_means.min():.2f}, {cost_means.max():.2f}] veh·h")
        print(f"  Cost-to-go shape: {'ASYMMETRIC' if is_asymmetric else 'SYMMETRIC'}")
        print(f"  Asymmetry ratio: {asymmetry_ratio:.2f}×")
        print(f"\n  Quadratic fit R² (7D): {r2_quad:.4f}")
        print(f"  Quadratic fit R² (1D): {r2_quad_1d:.4f}")
        print(f"  Max residual near ρ_crit: {max_res_crit:.1f}%")

        r2_v = r2_quad_1d  # use 1D for verdict (7D overfits with collinear features)
        if r2_v < 0.90 and is_asymmetric:
            verdict = "PASS: Quadratic clearly insufficient — hockey-stick confirmed"
        elif r2_v < 0.95 or is_asymmetric:
            verdict = "MARGINAL: Quadratic captures trend but misses asymmetry near ρ_crit"
        else:
            verdict = "FAIL: Quadratic fits well — NN may not add value"
        print(f"\n  VERDICT: {verdict}")
        results['verdict'] = verdict

    return results


if __name__ == '__main__':
    run_experiment1(n_mc=50, n_steps=600, verbose=True)
