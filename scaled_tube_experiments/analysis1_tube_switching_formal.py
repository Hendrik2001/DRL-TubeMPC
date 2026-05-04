"""
Analysis 1: Formal tube switching reachability.

Three progressively tighter checks for each adjacent pair (i, i+1):
  A  Bounding-box inclusion   (quick but conservative)
  B  Ellipsoidal inclusion    (P_j − P_reach ≥ 0)
  C  Shifted-ellipsoid check  (accounts for Δx_op between operating points)
"""
import numpy as np
import metanet6 as mn
import tube_tools as tt
import experiment2_lookup_table as ex2


Delta = np.array([mn.DELTA_D, mn.DELTA_Q])


# ── helpers ───────────────────────────────────────────────────────────
def reach_cov(A, P, Bw, Sigma_w):
    """Covariance of one-step reachable ellipsoid: A P A^T + Bw Σ Bw^T."""
    return A @ P @ A.T + Bw @ Sigma_w @ Bw.T


def bbox_halfwidths_from_cov(P, c):
    return np.sqrt(c * np.diag(P))


def dist_box_halfwidths(Bw, Delta):
    return np.abs(Bw) @ Delta


def method_A_bbox(ei, ej):
    """Bounding-box inclusion: does bbox(R_i) ⊆ bbox(Z_j)?"""
    Sigma_w = np.diag(Delta ** 2)
    P_r = reach_cov(ei['A_cl'], ei['P'], ei['Bw'], Sigma_w)
    hw_r = bbox_halfwidths_from_cov(P_r, ei['c']) + dist_box_halfwidths(ei['Bw'], Delta)
    hw_j = ej['hw']
    margins = hw_j - hw_r
    return margins, bool(np.all(margins >= -1e-10))


def method_B_ellipsoidal(ei, ej):
    """
    Ellipsoidal containment: P_reach ⊆ Z_j iff P_j − α P_reach ≥ 0
    where both are scaled to the SAME c. We check P_j − P_reach directly
    (both using c_j for the threshold).
    """
    Sigma_w = np.diag(Delta ** 2)
    P_reach = reach_cov(ei['A_cl'], ei['P'], ei['Bw'], Sigma_w)
    # Scale both to the same c (c_j)
    # Z_j = {e : e^T P_j^{-1} e ≤ c_j}
    # R_i ≈ {e : e^T P_reach^{-1} e ≤ c_i}
    # For containment with same c: need c_j P_reach ≤ c_i P_j
    # i.e.  P_j − (c_j/c_i) P_reach ≥ 0
    ratio = ej['c'] / max(ei['c'], 1e-12)
    diff = ej['P'] - ratio * P_reach
    eigs = np.linalg.eigvalsh(diff)
    min_eig = float(np.min(eigs))
    contained = min_eig >= -1e-10
    return min_eig, contained


def method_C_shifted(ei, ej):
    """
    Account for the nominal-point shift  Δx = x_op_j − x_op_i.

    In the j-frame the reachable set from Z_i is centred at −Δx (not 0).
    The worst-case deviation in the j-frame is therefore
       max_{e ∈ Z_i, w ∈ W}  ||A_i e + B_w w − Δx||_{P_j^{-1}}
    We approximate: check if the CENTRE (−Δx) already lies inside Z_j,
    THEN check if centre + RPI halfwidths still fits.
    """
    dx = ej['x_op'] - ei['x_op']                   # shift in state space
    # "Centre" of the reachable set in j-frame = A_i · 0 + 0 − Δx = −Δx
    centre = -dx
    # Is the centre inside Z_j?  (Use pseudo-inverse; P can be singular
    # in the queue dimension when the RPI has zero width there.)
    P_j_inv = np.linalg.pinv(ej['P'], rcond=1e-12)
    maha_centre = float(centre @ P_j_inv @ centre)
    centre_inside = maha_centre <= ej['c']

    # Bounding-box check: is centre ± reach_hw inside bbox(Z_j)?
    Sigma_w = np.diag(Delta ** 2)
    P_r = reach_cov(ei['A_cl'], ei['P'], ei['Bw'], Sigma_w)
    hw_r = bbox_halfwidths_from_cov(P_r, ei['c']) + dist_box_halfwidths(ei['Bw'], Delta)
    hw_j = ej['hw']
    # Effective margin: must contain the shifted + spread ellipsoid
    margin = hw_j - (np.abs(centre) + hw_r)
    all_fit = bool(np.all(margin >= -1e-10))

    return centre_inside, maha_centre, all_fit, margin


# ── main ──────────────────────────────────────────────────────────────
def run(verbose=True):
    table, _ = ex2.run(verbose=False)
    stable = [e for e in table if e['stable']]

    results = []
    if verbose:
        hdr = (f"  {'i→j':>5}  {'f_i':>5}{'f_j':>5}  "
               f"{'BBox':>6}  {'Ell':>6}  {'minEig':>10}  "
               f"{'CtrIn':>6}  {'Shift':>6}  {'Verdict':>10}")
        print(hdr)
        print("  " + "-" * 72)

    for k in range(len(stable) - 1):
        ei, ej = stable[k], stable[k + 1]
        marg_A, ok_A = method_A_bbox(ei, ej)
        min_eig, ok_B = method_B_ellipsoidal(ei, ej)
        ctr_in, maha, ok_C, marg_C = method_C_shifted(ei, ej)

        if ok_A and ok_B and ok_C:
            verdict = "SAFE"
        elif ok_A or ok_B:
            verdict = "MARGINAL"
        else:
            verdict = "UNSAFE"

        entry = dict(
            ei=ei, ej=ej,
            ok_A=ok_A, margins_A=marg_A,
            ok_B=ok_B, min_eig=min_eig,
            ctr_inside=ctr_in, maha=maha, ok_C=ok_C, margins_C=marg_C,
            verdict=verdict,
        )
        results.append(entry)

        if verbose:
            print(f"  {ei['factor']:.2f}→{ej['factor']:.2f}  "
                  f"{ei['factor']:>5.2f}{ej['factor']:>5.2f}  "
                  f"{'YES' if ok_A else 'NO':>6}  "
                  f"{'YES' if ok_B else 'NO':>6}  "
                  f"{min_eig:>10.4f}  "
                  f"{'YES' if ctr_in else 'NO':>6}  "
                  f"{'YES' if ok_C else 'NO':>6}  "
                  f"{verdict:>10}")

    n_safe = sum(1 for r in results if r['verdict'] == 'SAFE')
    n_marginal = sum(1 for r in results if r['verdict'] == 'MARGINAL')
    n_total = len(results)

    if verbose:
        print(f"\n  Summary: {n_safe} SAFE + {n_marginal} MARGINAL "
              f"out of {n_total} transitions "
              f"({(n_safe + n_marginal)/max(n_total,1)*100:.0f}% feasible)")

    return results


if __name__ == '__main__':
    print("=" * 78)
    print("  ANALYSIS 1: Formal Tube Switching Reachability")
    print("=" * 78)
    run()
