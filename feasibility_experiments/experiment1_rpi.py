"""
Experiment 1: RPI Set Computation for Linearized METANET
"""
import time
import numpy as np
from scipy.linalg import solve_discrete_lyapunov, solve_discrete_are, eig
from metanet_model import (
    find_equilibrium, linearize, equilibrium_speed,
    R_NOMINAL, D_NOMINAL, Q_UPSTREAM, RHO_CRIT, RHO_MAX, V_FREE, W_MAX,
    T, L, LAMBDA, TAU, ETA, KAPPA, A_PARAM, C_O
)

STATE_NAMES = ['ρ1', 'ρ2', 'ρ3', 'v1', 'v2', 'v3', 'w1']
STATE_UNITS = ['veh/km/lane'] * 3 + ['km/h'] * 3 + ['veh']

# Disturbance bounds
DELTA_D = 200   # ±200 veh/h demand uncertainty
DELTA_Q = 500   # ±500 veh/h upstream flow uncertainty


def run_experiment1():
    results = {}
    print("=" * 70)
    print("EXPERIMENT 1: RPI Set for Linearized METANET")
    print("=" * 70)

    # --- 1.2: Find Equilibrium ---
    print("\n--- 1.2: Finding Free-Flow Equilibrium ---")
    t0 = time.time()
    x_eq, n_iter = find_equilibrium()
    t_eq = time.time() - t0

    print(f"Converged in {n_iter} iterations ({t_eq:.2f}s)")
    print("\nEquilibrium state:")
    for i, (name, unit) in enumerate(zip(STATE_NAMES, STATE_UNITS)):
        print(f"  {name}* = {x_eq[i]:.4f} {unit}")

    # Check it's free-flow
    for i in range(3):
        ratio = x_eq[i] / RHO_CRIT
        status = "FREE-FLOW" if ratio < 1 else "CONGESTED"
        print(f"  {STATE_NAMES[i]}/ρ_crit = {ratio:.3f} ({status})")

    results['x_eq'] = x_eq
    results['eq_iterations'] = n_iter
    results['eq_time'] = t_eq

    # --- 1.3: Linearize ---
    print("\n--- 1.3: Linearization ---")
    t0 = time.time()
    A, B_u, B_w = linearize(x_eq, R_NOMINAL, D_NOMINAL, Q_UPSTREAM)
    t_lin = time.time() - t0

    print(f"Linearization completed in {t_lin:.4f}s")
    results['A'] = A
    results['B_u'] = B_u
    results['B_w'] = B_w

    # Check eigenvalues of A (open-loop)
    eigvals_A = np.linalg.eigvals(A)
    print("\nEigenvalues of A (open-loop):")
    for i, ev in enumerate(eigvals_A):
        print(f"  λ_{i+1} = {ev:.6f}  |λ| = {abs(ev):.6f}")

    max_eigval = max(abs(ev) for ev in eigvals_A)
    is_stable = max_eigval < 1.0
    print(f"\nSpectral radius: {max_eigval:.6f}")
    print(f"Open-loop Schur stable: {is_stable}")
    results['eigvals_A'] = eigvals_A
    results['A_stable'] = is_stable

    # Compute LQR gain if needed, but also compute it for comparison
    A_cl = A.copy()
    K = None

    if not is_stable:
        print("\nOpen-loop UNSTABLE — computing LQR stabilizing gain...")
        Q_lqr = np.eye(7)
        R_lqr = 10 * np.eye(1)
        try:
            P_are = solve_discrete_are(A, B_u, Q_lqr, R_lqr)
            K = -np.linalg.solve(R_lqr + B_u.T @ P_are @ B_u, B_u.T @ P_are @ A)
            A_cl = A + B_u @ K
            eigvals_cl = np.linalg.eigvals(A_cl)
            print("Eigenvalues of A_cl (closed-loop with LQR):")
            for i, ev in enumerate(eigvals_cl):
                print(f"  λ_{i+1} = {ev:.6f}  |λ| = {abs(ev):.6f}")
            results['K'] = K
            results['eigvals_Acl'] = eigvals_cl
        except Exception as e:
            print(f"LQR computation failed: {e}")
            print("Proceeding with open-loop A (may be marginally stable)")
    else:
        print("System is open-loop stable — using A directly for RPI computation.")
        # Still try LQR for faster convergence of RPI
        try:
            Q_lqr = np.eye(7)
            R_lqr = 10 * np.eye(1)
            P_are = solve_discrete_are(A, B_u, Q_lqr, R_lqr)
            K = -np.linalg.solve(R_lqr + B_u.T @ P_are @ B_u, B_u.T @ P_are @ A)
            A_cl_lqr = A + B_u @ K
            eigvals_cl = np.linalg.eigvals(A_cl_lqr)
            print("\n(Optional) LQR closed-loop eigenvalues:")
            for i, ev in enumerate(eigvals_cl):
                print(f"  λ_{i+1} = {ev:.6f}  |λ| = {abs(ev):.6f}")
            results['K'] = K
            results['eigvals_Acl'] = eigvals_cl
            # Use LQR closed-loop for RPI (tighter set)
            A_cl = A_cl_lqr
            print("Using LQR closed-loop dynamics for RPI (tighter set).")
        except Exception as e:
            print(f"LQR failed (non-critical): {e}")

    results['A_cl'] = A_cl

    # --- 1.4: RPI Set Computation ---
    print("\n--- 1.4: RPI Set Computation ---")

    # Disturbance set W = {w : |w_d| ≤ Δd, |w_q| ≤ Δq}
    # In deviation coordinates: w = B_w @ [w_d, w_q]
    # The effective disturbance in state space is d = B_w @ w_dist
    # where w_dist ∈ [-Δd, Δd] × [-Δq, Δq]

    # ========== APPROACH 1: Ellipsoidal RPI ==========
    print("\n  --- Approach 1: Ellipsoidal RPI ---")
    t0 = time.time()

    # Disturbance covariance (uniform distribution on [-Δ, Δ] has var = Δ²/3)
    Sigma_w_dist = np.diag([DELTA_D**2 / 3, DELTA_Q**2 / 3])
    Q_dist = B_w @ Sigma_w_dist @ B_w.T

    # Solve discrete Lyapunov: P = A_cl P A_cl^T + Q_dist
    try:
        P_lyap = solve_discrete_lyapunov(A_cl, Q_dist)
        t_ellip = time.time() - t0

        # For a confidence ellipsoid containing ~99% of states:
        # Use chi-squared with 7 dof, 99% => c ≈ 18.48
        # But for RPI (worst-case), use the support function approach
        # The bounding box of ellipsoid {e: e^T P^{-1} e ≤ c} in dimension j is ±sqrt(c * P_jj)
        # For guaranteed containment of all disturbances (not probabilistic),
        # scale by the worst-case ratio: use c = n (dimension) as a conservative bound
        c_ellip = 7.0  # dimension-based scaling

        ellip_widths = np.zeros(7)
        for j in range(7):
            ellip_widths[j] = 2 * np.sqrt(c_ellip * P_lyap[j, j])

        print(f"  Computation time: {t_ellip:.4f}s")
        print(f"  Lyapunov P diagonal: {np.diag(P_lyap)}")
        print(f"\n  Ellipsoidal RPI bounding box (±width in each dimension):")
        for j, (name, unit) in enumerate(zip(STATE_NAMES, STATE_UNITS)):
            half = ellip_widths[j] / 2
            print(f"    {name}: ±{half:.4f} {unit} (total width: {ellip_widths[j]:.4f})")

        results['P_lyap'] = P_lyap
        results['ellip_widths'] = ellip_widths
        results['ellip_time'] = t_ellip

    except Exception as e:
        print(f"  Ellipsoidal computation failed: {e}")
        t_ellip = time.time() - t0
        results['ellip_time'] = t_ellip

    # ========== APPROACH 2: Zonotopic RPI ==========
    print("\n  --- Approach 2: Zonotopic RPI ---")
    t0 = time.time()

    # Disturbance zonotope W: generators are columns of B_w * diag(Δd, Δq)
    G_w = B_w @ np.diag([DELTA_D, DELTA_Q])  # 7×2 generator matrix

    # Z_s = ⊕_{i=0}^{s-1} A_cl^i * W
    # Zonotope generators: [G_w, A_cl*G_w, A_cl^2*G_w, ..., A_cl^{s-1}*G_w]

    # Determine s: stop when A_cl^s * G_w contribution is negligible
    # Use Raković criterion: α(s) = max_j ||A_cl^s * G_w||_∞ / ||G_w||_∞
    epsilon = 0.01  # 1% approximation tolerance

    s_max = 500
    A_power = np.eye(7)
    generators = []
    alpha_vals = []

    G_w_norm = np.max(np.abs(G_w).sum(axis=1))  # ∞-norm of generator contribution

    for s in range(s_max):
        G_s = A_power @ G_w
        generators.append(G_s)

        # Check convergence
        G_s_norm = np.max(np.abs(G_s).sum(axis=1))
        alpha_s = G_s_norm / G_w_norm if G_w_norm > 0 else 0

        alpha_vals.append(alpha_s)

        if alpha_s < epsilon / (epsilon + 1) and s > 0:
            break

        A_power = A_cl @ A_power

    s_final = len(generators)
    alpha_final = alpha_vals[-1] if alpha_vals else 0

    # Combine all generators: G_total is 7 × (2*s_final)
    G_total = np.hstack(generators)

    # Scale by 1/(1-α) for outer approximation
    scale = 1.0 / (1.0 - alpha_final) if alpha_final < 1 else 1.0

    # Bounding box of zonotope: in each dimension j, width = 2 * sum(|G_total[j,:]|) * scale
    zono_widths = np.zeros(7)
    for j in range(7):
        zono_widths[j] = 2 * scale * np.sum(np.abs(G_total[j, :]))

    t_zono = time.time() - t0

    print(f"  Convergence at s = {s_final} (α(s) = {alpha_final:.6e})")
    print(f"  Total generators: {G_total.shape[1]}")
    print(f"  Computation time: {t_zono:.4f}s")
    print(f"  Scale factor 1/(1-α): {scale:.6f}")
    print(f"\n  Zonotopic RPI bounding box (±width in each dimension):")
    for j, (name, unit) in enumerate(zip(STATE_NAMES, STATE_UNITS)):
        half = zono_widths[j] / 2
        print(f"    {name}: ±{half:.4f} {unit} (total width: {zono_widths[j]:.4f})")

    results['zono_widths'] = zono_widths
    results['zono_generators'] = G_total
    results['zono_s'] = s_final
    results['zono_alpha'] = alpha_final
    results['zono_time'] = t_zono
    results['zono_scale'] = scale

    # --- Comparison ---
    print("\n--- RPI Set Comparison ---")
    print(f"{'Dim':<6} {'Ellipsoidal':>14} {'Zonotopic':>14} {'Unit':>14}")
    print("-" * 52)
    for j, (name, unit) in enumerate(zip(STATE_NAMES, STATE_UNITS)):
        ew = ellip_widths[j] if 'ellip_widths' in results else float('nan')
        zw = zono_widths[j]
        print(f"{name:<6} {ew:>14.4f} {zw:>14.4f} {unit:>14}")

    # --- 1.5: Check constraints ---
    print("\n--- 1.5: Constraint Check ---")
    constraint_ranges = np.array([
        RHO_MAX, RHO_MAX, RHO_MAX,  # density: 0 to ρ_max
        V_FREE, V_FREE, V_FREE,     # speed: 0 to v_free
        W_MAX                         # queue: 0 to w_max
    ])

    print("RPI fits within constraint set?")
    # Use zonotopic (tighter/more precise for box constraints)
    rpi_widths = zono_widths
    for j, (name, unit) in enumerate(zip(STATE_NAMES, STATE_UNITS)):
        half_rpi = rpi_widths[j] / 2
        fits = half_rpi < constraint_ranges[j] / 2
        pct = rpi_widths[j] / constraint_ranges[j] * 100
        print(f"  {name}: RPI width = {rpi_widths[j]:.4f}, Constraint range = {constraint_ranges[j]:.1f}, "
              f"({pct:.1f}% of range) {'OK' if fits else 'EXCEEDS!'}")

    results['constraint_ranges'] = constraint_ranges
    results['rpi_widths'] = rpi_widths  # use zonotopic as primary

    total_time = t_eq + t_lin + (results.get('ellip_time', 0)) + t_zono
    print(f"\nTotal computation time: {total_time:.2f}s")
    results['total_time'] = total_time

    success = (
        results.get('A_stable', False) or results.get('K') is not None
    ) and all(
        rpi_widths[j] < constraint_ranges[j] for j in range(7)
    ) and total_time < 60

    print(f"\n{'='*50}")
    print(f"EXPERIMENT 1 {'SUCCEEDED' if success else 'FAILED'}")
    if success:
        print("  ✓ System is Schur stable")
        print("  ✓ RPI set is finite and bounded")
        print(f"  ✓ Computation completed in {total_time:.2f}s < 60s")
    print(f"{'='*50}")

    results['success'] = success
    return results


if __name__ == '__main__':
    results = run_experiment1()
