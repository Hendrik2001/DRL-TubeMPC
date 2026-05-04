"""
Experiment 3: Tube switching reachability.

For each pair (i, i+1) of STABLE adjacent operating points in the lookup table,
check whether the one-step reachable set R_i = A_i · Z_i ⊕ W fits inside Z_{i+1}.

Using bounding boxes of ellipsoids (half-widths), the per-dimension overlap
ratio is   min(hw_R, hw_Zj) / hw_R   per dimension, combined as a mean.
"""
import numpy as np
import metanet6 as mn
import tube_tools as tt
import experiment2_lookup_table as ex2


def reach_halfwidths(entry, Delta):
    """One-step reachable halfwidths from the ellipsoidal RPI at this entry."""
    A_cl = entry['A_cl']
    P = entry['P']
    c = entry['c']
    Bw = entry['Bw']
    P_reach = A_cl @ P @ A_cl.T
    hw_state = np.sqrt(c * np.diag(P_reach))
    hw_dist = np.abs(Bw) @ Delta
    return hw_state + hw_dist


def overlap_ratio(hw_R, hw_Z):
    """
    Box overlap measure: what fraction of R's "span" is contained in Z's span
    per dimension, averaged across dimensions.
    """
    hw_R = np.maximum(hw_R, 1e-12)
    ratios = np.minimum(hw_R, hw_Z) / hw_R
    return float(ratios.mean()), ratios


def run(verbose=True):
    table, _ = ex2.run(verbose=False)
    Delta = np.array([mn.DELTA_D, mn.DELTA_Q])

    pairs = []
    for i in range(len(table) - 1):
        e_i = table[i]
        e_j = table[i + 1]
        if not e_i['stable'] or not e_j['stable']:
            pairs.append({
                'i': i, 'j': i + 1,
                'factor_i': e_i['factor'], 'factor_j': e_j['factor'],
                'feasible': False, 'reason': 'one endpoint unstable',
                'overlap_mean': 0.0, 'hw_R': None, 'hw_Zj': None,
            })
            continue

        hw_R = reach_halfwidths(e_i, Delta)
        hw_Zj = e_j['hw']
        ov, per_dim = overlap_ratio(hw_R, hw_Zj)

        if ov > 0.80:
            verdict, code = "YES", "ok"
        elif ov > 0.50:
            verdict, code = "MARGINAL", "marginal"
        else:
            verdict, code = "NO", "gap"

        pairs.append({
            'i': i, 'j': i + 1,
            'factor_i': e_i['factor'], 'factor_j': e_j['factor'],
            'feasible': ov > 0.50, 'verdict': verdict, 'code': code,
            'overlap_mean': ov, 'per_dim': per_dim,
            'hw_R': hw_R, 'hw_Zj': hw_Zj,
            'gap_max': float(np.max(np.maximum(hw_R - hw_Zj, 0.0))),
        })

    if verbose:
        print(f"  {'pair':>6}  {'f_i→f_j':>12}  {'overlap':>9}  {'gap_max':>9}  verdict")
        print("  " + "-" * 60)
        for p in pairs:
            tag = f"{p['factor_i']:.2f}→{p['factor_j']:.2f}"
            if 'verdict' in p:
                print(f"  {p['i']:>2}-{p['j']:<3}  {tag:>12}  "
                      f"{p['overlap_mean']*100:>7.1f}%  {p['gap_max']:>9.3f}  {p['verdict']}")
            else:
                print(f"  {p['i']:>2}-{p['j']:<3}  {tag:>12}  "
                      f"{'-':>9}  {'-':>9}  {p['reason']}")

        n_ok = sum(1 for p in pairs if p.get('feasible'))
        n_total = len(pairs)
        print(f"\n  Feasible transitions: {n_ok}/{n_total} "
              f"({n_ok/n_total*100:.0f}%)")

    return table, pairs


if __name__ == '__main__':
    print("=" * 70)
    print("  EXPERIMENT 3: Tube Switching Reachability")
    print("=" * 70)
    run()
