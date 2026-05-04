"""
Analysis 2: Linearization error validation.

For each stable operating point, sample states inside the RPI and compare
the true nonlinear METANET step against the linearized prediction.

Reports error ratios relative to the RPI width at 1-step, 5-step, 10-step.
"""
import numpy as np
import metanet6 as mn
import tube_tools as tt
import experiment2_lookup_table as ex2

N_SAMPLES = 500
HORIZONS = [1, 5, 10]
RNG = np.random.RandomState(42)
Delta = np.array([mn.DELTA_D, mn.DELTA_Q])


def sample_in_ellipsoid(P, c, n_samples, rng):
    """Rejection-sample vectors e such that e^T P^{-1} e ≤ c.
    Regularize P so Cholesky works even when queue dim has zero variance."""
    n = P.shape[0]
    P_reg = P + 1e-12 * np.eye(n)
    L = np.linalg.cholesky(P_reg)
    P_inv = np.linalg.pinv(P_reg, rcond=1e-10)
    samples = []
    attempts = 0
    while len(samples) < n_samples and attempts < n_samples * 50:
        z = rng.randn(n)
        e = L @ z
        if e @ P_inv @ e <= c:
            samples.append(e)
        attempts += 1
    if len(samples) < n_samples:
        # Fall back: scale samples to sit on the boundary
        while len(samples) < n_samples:
            z = rng.randn(n)
            e = L @ z
            m = e @ P_inv @ e
            if m > 0:
                e *= np.sqrt(c / m) * rng.uniform(0.1, 1.0)
            samples.append(e)
    return np.array(samples[:n_samples])


def sample_disturbance(n_samples, rng):
    """Uniform in the disturbance box."""
    w_d = rng.uniform(-Delta[0], Delta[0], n_samples)
    w_q = rng.uniform(-Delta[1], Delta[1], n_samples)
    return np.column_stack([w_d, w_q])


def multi_step_error(entry, horizon, n_samples, rng):
    """
    Simulate `horizon` steps from states inside the RPI.
    Returns max and mean error per dimension, plus the ratio to hw.
    """
    x_op = entry['x_op']
    A = entry['A_cl']
    Bw = entry['Bw']
    P, c, hw = entry['P'], entry['c'], entry['hw']
    q_up = entry['q_up']
    d1 = mn.D_NOMINAL

    # Replace zero hw with small positive for ratio computation
    hw_safe = np.where(hw > 1e-8, hw, 1e-8)

    es = sample_in_ellipsoid(P, c, n_samples, rng)
    errors = np.zeros((n_samples, mn.NX))

    for i in range(n_samples):
        e = es[i]
        x_nl = x_op + e          # true nonlinear state
        e_lin = e.copy()          # linearized deviation

        for t in range(horizon):
            w_d = rng.uniform(-Delta[0], Delta[0])
            w_q = rng.uniform(-Delta[1], Delta[1])
            w = np.array([w_d, w_q])

            # Nonlinear
            x_nl = mn.metanet_step(x_nl, mn.R_NOMINAL, d1 + w_d, q_up + w_q)
            # Linearized
            e_lin = A @ e_lin + Bw @ w

        errors[i] = x_nl - (x_op + e_lin)

    abs_err = np.abs(errors)
    max_err = abs_err.max(axis=0)
    mean_err = abs_err.mean(axis=0)
    max_ratio = max_err / hw_safe
    mean_ratio = mean_err / hw_safe

    return {
        'max_err': max_err,
        'mean_err': mean_err,
        'max_ratio': max_ratio,
        'mean_ratio': mean_ratio,
        'max_ratio_overall': float(max_ratio.max()),
        'mean_ratio_overall': float(mean_ratio.mean()),
    }


def run(verbose=True):
    table, _ = ex2.run(verbose=False)
    stable = [e for e in table if e['stable']
              and e['hw'] is not None
              and e['hw'][0] < mn.RHO_MAX / 2]

    all_results = {}

    for h in HORIZONS:
        if verbose:
            print(f"\n  ── {h}-step error ──")
            print(f"  {'factor':>6}  {'maxRatio':>9}  {'meanRatio':>9}  "
                  f"{'maxErr_ρ1':>10}  {'maxErr_v1':>10}  {'quality':>12}")
            print("  " + "-" * 64)

        for e in stable:
            rng = np.random.RandomState(42)
            res = multi_step_error(e, h, N_SAMPLES, rng)
            key = (e['factor'], h)
            all_results[key] = res

            mr = res['max_ratio_overall']
            quality = ("EXCELLENT" if mr < 0.10 else
                       "ACCEPTABLE" if mr < 0.30 else
                       "POOR")

            if verbose:
                print(f"  {e['factor']:>6.2f}  "
                      f"{mr:>9.4f}  "
                      f"{res['mean_ratio_overall']:>9.4f}  "
                      f"{res['max_err'][0]:>10.4f}  "
                      f"{res['max_err'][mn.N_SEG]:>10.4f}  "
                      f"{quality:>12}")

    # Summary
    if verbose:
        print("\n  ── Summary by horizon ──")
        for h in HORIZONS:
            ratios = [all_results[(e['factor'], h)]['max_ratio_overall']
                      for e in stable if (e['factor'], h) in all_results]
            ok = sum(1 for r in ratios if r < 0.30)
            print(f"  {h:>2}-step: "
                  f"max ratio = {max(ratios):.4f}, "
                  f"mean ratio = {np.mean(ratios):.4f}, "
                  f"{ok}/{len(ratios)} ACCEPTABLE or better")

    return all_results, stable


if __name__ == '__main__':
    print("=" * 78)
    print("  ANALYSIS 2: Linearization Error Validation")
    print("=" * 78)
    run()
