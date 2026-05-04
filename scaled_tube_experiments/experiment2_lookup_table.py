"""
Experiment 2: Lookup table of RPI sets across 12 operating points.

At each ρ_op = factor * ρ_crit, we synthesise a homogeneous operating point
(all densities = ρ_op, all speeds = V_e(ρ_op), w1 = 0), linearize there, and
compute the ellipsoidal RPI. We store everything in a dictionary list so later
experiments can do lookup + switching analysis.
"""
import numpy as np
import time
import metanet6 as mn
import tube_tools as tt


RHO_FACTORS = [
    0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
    0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90,
    0.92, 0.95, 0.97, 1.00, 1.05, 1.10, 1.20, 1.40,
]


def make_operating_point(factor):
    """
    Build a physically consistent homogeneous operating state at density
    rho = factor * rho_crit. Upstream flow and demand are adjusted so the
    metanet step is (approximately) at steady state.
    """
    rho_op = factor * mn.RHO_CRIT
    v_op = mn.Ve(rho_op)
    q_op = rho_op * v_op * mn.LAM  # mainline flow at this density
    x = mn.pack(np.full(mn.N_SEG, rho_op), np.full(mn.N_SEG, v_op), 0.0)
    # Use this flow as the upstream boundary and keep the on-ramp small so
    # the linearization point is consistent. Demand = nominal.
    q_up = q_op
    d1 = mn.D_NOMINAL
    return x, q_up, d1, rho_op, v_op


def run(verbose=True):
    t_all = time.time()
    Delta = np.array([mn.DELTA_D, mn.DELTA_Q])
    table = []

    if verbose:
        print(f"  {'idx':>3}  {'factor':>6}  {'ρ_op':>6}  {'v_op':>6}  "
              f"{'q_up':>6}  {'ρ(A)':>7}  {'LQR':>4}  {'ρ1_w':>7}  "
              f"{'v1_w':>7}  {'frac_avg':>8}  {'frac_min':>8}  {'t_ms':>6}")
        print("  " + "-" * 96)

    for i, f in enumerate(RHO_FACTORS):
        t0 = time.time()
        x_op, q_up, d1, rho_op, v_op = make_operating_point(f)
        A, Bu, Bw = mn.linearize(x_op, mn.R_NOMINAL, d1, q_up)
        rho_A = tt.spectral_radius(A)
        A_cl, K = tt.stabilize(A, Bu)
        rho_cl = tt.spectral_radius(A_cl)

        if rho_cl >= 1.0 - 1e-9:
            # give up on this point
            entry = {
                'idx': i, 'factor': f, 'rho_op': rho_op, 'v_op': v_op,
                'q_up': q_up, 'x_op': x_op,
                'A': A, 'A_cl': A_cl, 'Bu': Bu, 'Bw': Bw, 'K': K,
                'spectral_A': rho_A, 'spectral_Acl': rho_cl,
                'stable': False, 'P': None, 'c': None, 'hw': None,
                'frac': None, 'time_ms': (time.time() - t0) * 1000,
            }
            table.append(entry)
            if verbose:
                print(f"  {i:>3}  {f:>6.2f}  {rho_op:>6.2f}  {v_op:>6.2f}  "
                      f"{q_up:>6.0f}  {rho_A:>7.4f}  {'yes' if K is not None else 'no':>4}  "
                      f"  UNSTABLE — skipped")
            continue

        P, c, hw = tt.ellipsoidal_rpi(A_cl, Bw, Delta)
        if hw is None:
            continue

        lo, hi = mn.constraint_box()
        tight_lo = lo + hw
        tight_hi = hi - hw
        ranges = hi - lo
        tight_ranges = np.maximum(tight_hi - tight_lo, 0.0)
        frac = tight_ranges / ranges

        t_ms = (time.time() - t0) * 1000
        entry = {
            'idx': i, 'factor': f, 'rho_op': rho_op, 'v_op': v_op,
            'q_up': q_up, 'x_op': x_op,
            'A': A, 'A_cl': A_cl, 'Bu': Bu, 'Bw': Bw, 'K': K,
            'spectral_A': rho_A, 'spectral_Acl': rho_cl,
            'stable': True, 'P': P, 'c': c, 'hw': hw,
            'frac': frac, 'tight_lo': tight_lo, 'tight_hi': tight_hi,
            'time_ms': t_ms,
        }
        table.append(entry)

        if verbose:
            rho1_w = hw[0]
            v1_w = hw[mn.N_SEG]
            print(f"  {i:>3}  {f:>6.2f}  {rho_op:>6.2f}  {v_op:>6.2f}  "
                  f"{q_up:>6.0f}  {rho_cl:>7.4f}  {'yes' if K is not None else 'no':>4}  "
                  f"{rho1_w:>7.3f}  {v1_w:>7.3f}  "
                  f"{frac.mean()*100:>7.1f}%  {frac.min()*100:>7.1f}%  {t_ms:>6.1f}")

    t_total = time.time() - t_all
    n_stable = sum(1 for e in table if e['stable'])
    if verbose:
        print(f"\n  {n_stable}/{len(RHO_FACTORS)} operating points produced a stable RPI.")
        print(f"  Total time: {t_total*1000:.1f} ms")

    return table, t_total


if __name__ == '__main__':
    print("=" * 110)
    print("  EXPERIMENT 2: RPI Lookup Table (12 operating points)")
    print("=" * 110)
    run()
