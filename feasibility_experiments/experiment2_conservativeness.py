"""
Experiment 2: Tube Conservativeness Analysis
"""
import time
import numpy as np
from scipy.linalg import solve_discrete_lyapunov, solve_discrete_are
from metanet_model import (
    find_equilibrium, linearize, equilibrium_speed, metanet_step,
    R_NOMINAL, D_NOMINAL, Q_UPSTREAM, RHO_CRIT, RHO_MAX, V_FREE, W_MAX,
    T, L, LAMBDA, TAU, ETA, KAPPA, A_PARAM, C_O
)

STATE_NAMES = ['ρ1', 'ρ2', 'ρ3', 'v1', 'v2', 'v3', 'w1']
STATE_UNITS = ['veh/km/lane'] * 3 + ['km/h'] * 3 + ['veh']
DELTA_D = 200
DELTA_Q = 500


def compute_rpi_zonotopic(A_cl, B_w, delta_d, delta_q, epsilon=0.01, s_max=500):
    """Compute zonotopic RPI bounding box widths."""
    G_w = B_w @ np.diag([delta_d, delta_q])
    G_w_norm = np.max(np.abs(G_w).sum(axis=1))

    A_power = np.eye(A_cl.shape[0])
    generators = []

    for s in range(s_max):
        G_s = A_power @ G_w
        generators.append(G_s)
        G_s_norm = np.max(np.abs(G_s).sum(axis=1))
        alpha_s = G_s_norm / G_w_norm if G_w_norm > 0 else 0
        if alpha_s < epsilon / (epsilon + 1) and s > 0:
            break
        A_power = A_cl @ A_power

    alpha_final = alpha_s
    G_total = np.hstack(generators)
    scale = 1.0 / (1.0 - alpha_final) if alpha_final < 1 else 1.0

    widths = np.zeros(A_cl.shape[0])
    for j in range(A_cl.shape[0]):
        widths[j] = 2 * scale * np.sum(np.abs(G_total[j, :]))

    return widths, G_total, len(generators), alpha_final


def compute_lqr_cl(A, B_u):
    """Compute LQR closed-loop A_cl = A + B_u @ K."""
    Q_lqr = np.eye(A.shape[0])
    R_lqr = 10 * np.eye(B_u.shape[1])
    try:
        P_are = solve_discrete_are(A, B_u, Q_lqr, R_lqr)
        K = -np.linalg.solve(R_lqr + B_u.T @ P_are @ B_u, B_u.T @ P_are @ A)
        return A + B_u @ K, K
    except Exception:
        return A, None


def run_experiment2(exp1_results=None):
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Tube Conservativeness")
    print("=" * 70)

    results = {}

    # --- 2.1: Tightened Constraints ---
    print("\n--- 2.1: Tightened Constraints ---")

    if exp1_results is None:
        from experiment1_rpi import run_experiment1
        exp1_results = run_experiment1()

    rpi_widths = exp1_results['rpi_widths']
    x_eq = exp1_results['x_eq']

    # State constraints: [0, ρ_max] for densities, [0, v_free] for speeds, [0, w_max] for queue
    x_min = np.array([0, 0, 0, 0, 0, 0, 0], dtype=float)
    x_max = np.array([RHO_MAX, RHO_MAX, RHO_MAX, V_FREE, V_FREE, V_FREE, W_MAX], dtype=float)

    # Tightened: x_min + z_max ≤ x ≤ x_max - z_max, where z_max = rpi_width/2
    z_max = rpi_widths / 2

    x_min_tight = x_min + z_max
    x_max_tight = x_max - z_max

    print(f"\n{'Dim':<6} {'x_min':>8} {'x_max':>8} {'z_max':>10} {'x_min_t':>10} {'x_max_t':>10} {'Remaining%':>12}")
    print("-" * 70)

    remaining_fracs = np.zeros(7)
    for j in range(7):
        orig_range = x_max[j] - x_min[j]
        tight_range = max(x_max_tight[j] - x_min_tight[j], 0)
        frac = tight_range / orig_range if orig_range > 0 else 0
        remaining_fracs[j] = frac
        print(f"{STATE_NAMES[j]:<6} {x_min[j]:>8.1f} {x_max[j]:>8.1f} {z_max[j]:>10.4f} "
              f"{x_min_tight[j]:>10.4f} {x_max_tight[j]:>10.4f} {frac*100:>11.1f}%")

    results['remaining_fracs'] = remaining_fracs
    results['x_min_tight'] = x_min_tight
    results['x_max_tight'] = x_max_tight
    results['z_max'] = z_max

    # --- 2.2: Decision ---
    print("\n--- 2.2: Conservativeness Decision ---")
    avg_frac = np.mean(remaining_fracs)
    queue_frac = remaining_fracs[6]  # w1

    print(f"Average remaining fraction: {avg_frac*100:.1f}%")
    print(f"Queue (w1) remaining fraction: {queue_frac*100:.1f}%")

    if avg_frac > 0.5:
        verdict2 = "RL has substantial room to optimize — PROCEED"
    elif avg_frac > 0.2:
        verdict2 = "Tube is moderately conservative — PROCEED WITH CAUTION"
    else:
        verdict2 = "Tube is too conservative — RL has no room — STOP"

    print(f"\nVerdict: {verdict2}")
    results['verdict'] = verdict2
    results['avg_frac'] = avg_frac
    results['queue_frac'] = queue_frac

    # --- 2.3: State-Dependent Analysis ---
    print("\n--- 2.3: State-Dependent Tube Analysis ---")

    operating_points = [
        ("Free-flow", 0.3 * RHO_CRIT),
        ("Near-critical", 0.9 * RHO_CRIT),
        ("Congested", 1.5 * RHO_CRIT),
    ]

    tube_data = []

    for label, rho_target in operating_points:
        print(f"\n  Operating point: {label} (ρ ≈ {rho_target:.1f} veh/km/lane)")

        # Find a demand that produces roughly this density
        # From q = ρ * V(ρ) * λ, solve for upstream flow
        v_eq_approx = equilibrium_speed(rho_target)
        q_needed = rho_target * v_eq_approx * LAMBDA

        # Adjust upstream flow to achieve target density
        # Keep on-ramp demand fixed
        q_up_adj = max(q_needed - D_NOMINAL * R_NOMINAL * 0.5, 500)  # rough adjustment

        # Try to find equilibrium at this operating point
        # Use a range of upstream flows to hit the target density
        best_eq = None
        best_diff = float('inf')

        for q_try in np.linspace(max(500, q_needed * 0.5), min(q_needed * 2, 15000), 50):
            try:
                x_test, _ = find_equilibrium(r_o=R_NOMINAL, d_o=D_NOMINAL, q_up=q_try, max_iter=100000)
                diff = abs(x_test[0] - rho_target)
                if diff < best_diff:
                    best_diff = diff
                    best_eq = x_test
                    best_q_up = q_try
            except Exception:
                continue

        if best_eq is None or best_diff > rho_target * 0.3:
            print(f"    Could not find equilibrium near ρ = {rho_target:.1f}")
            print(f"    Best achieved: ρ1 = {best_eq[0]:.2f} (diff = {best_diff:.2f})")
            # Use approximate equilibrium anyway
            if best_eq is None:
                continue

        x_op = best_eq
        print(f"    Equilibrium: ρ1={x_op[0]:.2f}, v1={x_op[3]:.2f}, q_up={best_q_up:.0f}")

        # Linearize at this operating point
        A_op, B_u_op, B_w_op = linearize(x_op, R_NOMINAL, D_NOMINAL, best_q_up)

        # Check stability and use LQR if needed
        eigvals_op = np.linalg.eigvals(A_op)
        max_eig = max(abs(ev) for ev in eigvals_op)
        print(f"    Spectral radius: {max_eig:.6f} ({'stable' if max_eig < 1 else 'UNSTABLE'})")

        A_cl_op, K_op = compute_lqr_cl(A_op, B_u_op)
        if K_op is not None:
            eigvals_cl = np.linalg.eigvals(A_cl_op)
            max_eig_cl = max(abs(ev) for ev in eigvals_cl)
            print(f"    CL spectral radius: {max_eig_cl:.6f}")

        if max(abs(ev) for ev in np.linalg.eigvals(A_cl_op)) >= 1:
            print(f"    WARNING: System unstable at this operating point, skipping RPI.")
            tube_data.append((label, rho_target, x_op, None, None))
            continue

        # Compute RPI
        try:
            widths, _, s, alpha = compute_rpi_zonotopic(A_cl_op, B_w_op, DELTA_D, DELTA_Q)
            print(f"    RPI (s={s}, α={alpha:.2e}):")
            for j, (name, unit) in enumerate(zip(STATE_NAMES, STATE_UNITS)):
                print(f"      {name}: ±{widths[j]/2:.4f} {unit}")

            tube_data.append((label, rho_target, x_op, widths, remaining_fracs))
        except Exception as e:
            print(f"    RPI computation failed: {e}")
            tube_data.append((label, rho_target, x_op, None, None))

    results['tube_data'] = tube_data

    # Print comparison
    print("\n--- State-Dependent Tube Width Comparison ---")
    print(f"{'Regime':<15} {'ρ_op':>8} {'Tube ρ1':>12} {'Tube v1':>12} {'Tube w1':>12}")
    print("-" * 65)
    for label, rho, x_op, widths, _ in tube_data:
        if widths is not None:
            print(f"{label:<15} {rho:>8.1f} {widths[0]:>12.4f} {widths[3]:>12.4f} {widths[6]:>12.4f}")
        else:
            print(f"{label:<15} {rho:>8.1f} {'N/A':>12} {'N/A':>12} {'N/A':>12}")

    # Check if state-dependent tube shows expected pattern
    if len(tube_data) >= 2:
        valid = [(d[1], d[3]) for d in tube_data if d[3] is not None]
        if len(valid) >= 2:
            # Check if tube width increases with density (in density dimension)
            rhos = [v[0] for v in valid]
            density_widths = [v[1][0] for v in valid]
            if density_widths[-1] > density_widths[0]:
                print("\n  ✓ Tube width increases with density (as expected)")
            else:
                print("\n  Note: Tube width does not increase monotonically with density")

    success = avg_frac > 0.2 and queue_frac > 0.3
    print(f"\n{'='*50}")
    print(f"EXPERIMENT 2 {'SUCCEEDED' if success else 'FAILED'}")
    if success:
        print(f"  ✓ Average remaining fraction: {avg_frac*100:.1f}% > 20%")
        print(f"  ✓ Queue remaining fraction: {queue_frac*100:.1f}% > 30%")
    else:
        if avg_frac <= 0.2:
            print(f"  ✗ Average remaining fraction: {avg_frac*100:.1f}% ≤ 20%")
        if queue_frac <= 0.3:
            print(f"  ✗ Queue remaining fraction: {queue_frac*100:.1f}% ≤ 30%")
    print(f"{'='*50}")

    results['success'] = success
    return results


if __name__ == '__main__':
    results = run_experiment2()
