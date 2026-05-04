"""
Experiment 1: Scale to 6-segment METANET, compute RPI, report tightened constraints.
"""
import numpy as np
import time
import metanet6 as mn
import tube_tools as tt


def run(verbose=True):
    t_start = time.time()

    # 1. Equilibrium
    t0 = time.time()
    x_eq, iters = mn.find_equilibrium()
    t_eq = time.time() - t0
    if verbose:
        rho_eq, v_eq, w_eq = mn.unpack(x_eq)
        print("  Equilibrium:")
        print(f"    ρ* = {rho_eq}")
        print(f"    v* = {v_eq}")
        print(f"    w* = {w_eq:.3f}")
        print(f"    iterations = {iters}, time = {t_eq:.2f}s")

    # 2. Linearize
    t0 = time.time()
    A, Bu, Bw = mn.linearize(x_eq, mn.R_NOMINAL, mn.D_NOMINAL, mn.Q_UPSTREAM)
    t_lin = time.time() - t0
    eig = np.linalg.eigvals(A)
    rho_A = np.max(np.abs(eig))
    if verbose:
        print(f"  Linearization time: {t_lin*1000:.1f} ms")
        print(f"  Spectral radius(A) = {rho_A:.6f}  "
              f"[{'STABLE' if rho_A < 1 else 'UNSTABLE'}]")

    # 3. Stabilize if needed
    A_cl, K = tt.stabilize(A, Bu)
    rho_cl = tt.spectral_radius(A_cl)
    if verbose:
        print(f"  LQR used: {K is not None}, spectral radius(A_cl) = {rho_cl:.6f}")

    # 4. RPI
    Delta = np.array([mn.DELTA_D, mn.DELTA_Q])
    t0 = time.time()
    P, c, hw_ell = tt.ellipsoidal_rpi(A_cl, Bw, Delta)
    t_ell = time.time() - t0

    t0 = time.time()
    G, hw_zono, s = tt.zonotopic_rpi(A_cl, Bw, Delta, eps=0.01)
    t_zono = time.time() - t0

    if verbose:
        print(f"  Ellipsoidal RPI: {t_ell*1000:.1f} ms")
        print(f"  Zonotopic RPI : {t_zono*1000:.1f} ms (s={s} iters, {G.shape[1]} gens)")
        labels = [f"ρ{i+1}" for i in range(mn.N_SEG)] + \
                 [f"v{i+1}" for i in range(mn.N_SEG)] + ["w1"]
        print(f"\n  RPI half-widths (zonotopic → used for tightening):")
        for lbl, we, wz in zip(labels, hw_ell, hw_zono):
            print(f"    {lbl:>4}: ell={we:>9.4f}   zono={wz:>9.4f}")

    # 5. Tightened constraints
    lo, hi = mn.constraint_box()
    hw = hw_zono
    tight_lo = lo + hw
    tight_hi = hi - hw
    ranges = hi - lo
    tight_ranges = np.maximum(tight_hi - tight_lo, 0.0)
    frac = tight_ranges / ranges

    if verbose:
        print(f"\n  Tightened constraints & remaining fraction:")
        labels = [f"ρ{i+1}" for i in range(mn.N_SEG)] + \
                 [f"v{i+1}" for i in range(mn.N_SEG)] + ["w1"]
        for lbl, l, h, tl, th, f in zip(labels, lo, hi, tight_lo, tight_hi, frac):
            print(f"    {lbl:>4}: [{l:6.1f},{h:6.1f}] → "
                  f"[{tl:7.2f},{th:7.2f}]   {f*100:5.1f}%")
        print(f"\n  Average remaining fraction: {frac.mean()*100:.1f}%")
        print(f"  Minimum remaining fraction: {frac.min()*100:.1f}%  "
              f"({labels[int(np.argmin(frac))]})")

    t_total = time.time() - t_start
    if verbose:
        print(f"\n  Total Experiment 1 time: {t_total:.2f}s")

    return {
        'x_eq': x_eq, 'A': A, 'Bu': Bu, 'Bw': Bw, 'A_cl': A_cl, 'K': K,
        'spectral_A': rho_A, 'spectral_Acl': rho_cl,
        'P_ell': P, 'c_ell': c, 'hw_ell': hw_ell,
        'G_zono': G, 'hw_zono': hw_zono, 's_zono': s,
        'frac': frac, 'tight_lo': tight_lo, 'tight_hi': tight_hi,
        't_lin': t_lin, 't_ell': t_ell, 't_zono': t_zono, 't_total': t_total,
    }


if __name__ == '__main__':
    print("=" * 70)
    print("  EXPERIMENT 1: 6-segment METANET RPI")
    print("=" * 70)
    run()
