"""
Runner for the three deep-dive analyses + 6 plots.
Reuses the existing lookup table from experiment2.
"""
import os
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.collections import LineCollection

import metanet6 as mn
import analysis1_tube_switching_formal as a1
import analysis2_linearization_validation as a2
import analysis3_multi_regime_simulation as a3

HERE = os.path.dirname(os.path.abspath(__file__))


def banner(t):
    print("\n" + "=" * 78)
    print("  " + t)
    print("=" * 78)


def main():
    t0 = time.time()

    # ── Analysis 1 ─────────────────────────────────────────────────────
    banner("ANALYSIS 1: Formal Tube Switching Reachability")
    a1_res = a1.run()

    # ── Analysis 2 ─────────────────────────────────────────────────────
    banner("ANALYSIS 2: Linearization Error Validation")
    a2_res, a2_stable = a2.run()

    # ── Analysis 3 ─────────────────────────────────────────────────────
    banner("ANALYSIS 3: Multi-Regime 3-hour Simulation")
    a3_res = a3.run()

    # ────────────────────────────────────────────────────────────────────
    # PLOTS
    # ────────────────────────────────────────────────────────────────────

    # ── Plot A2: linearization error ratio vs density ──────────────────
    factors = [e['factor'] for e in a2_stable]
    fig, ax = plt.subplots(figsize=(8, 5))
    for h, ls, mkr in [(1, '-', 'o'), (5, '--', 's'), (10, ':', '^')]:
        ratios = [a2_res[(f, h)]['max_ratio_overall'] for f in factors]
        ax.plot(factors, ratios, ls, marker=mkr, lw=2,
                label=f'{h}-step max ratio')
    ax.axhline(0.10, color='green', ls='--', alpha=0.6, label='EXCELLENT (0.10)')
    ax.axhline(0.30, color='orange', ls='--', alpha=0.6, label='ACCEPTABLE (0.30)')
    ax.axvline(1.0, color='gray', ls=':', alpha=0.5)
    ax.set_xlabel('Operating density ρ_op / ρ_crit', fontsize=12)
    ax.set_ylabel('Max error ratio (|ε| / tube halfwidth)', fontsize=12)
    ax.set_title('Linearization Error vs Operating Density',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(HERE, 'analysis_plot2_linearization.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved analysis_plot2_linearization.png")

    # ── Plot A3: State evolution with tube-switching overlay ────────────
    t_min = a3_res['t'] / 60.0
    rho1 = a3_res['x'][:, 0]
    idx_arr = a3_res['idx']
    entries = a3_res['entries']

    fig, ax = plt.subplots(figsize=(12, 5))
    # Background colour by active tube
    unique_factors = sorted(set(e['factor'] for e in entries))
    cmap = plt.cm.viridis_r
    norm = plt.Normalize(min(unique_factors), max(unique_factors))
    for i in range(len(t_min) - 1):
        f = idx_arr[i]
        if f > 0:
            ax.axvspan(t_min[i], t_min[i + 1], color=cmap(norm(f)), alpha=0.15)

    ax.plot(t_min, rho1, lw=2, color='#1976D2', label='ρ₁')
    ax.axhline(mn.RHO_CRIT, color='red', ls='--', lw=1.5, label='ρ_crit')
    # Tube switch markers
    for (step, old, new, safe) in a3_res['switches']:
        t_sw = a3_res['t'][step] / 60.0
        ax.axvline(t_sw, color='green' if safe else 'red',
                   ls=':', lw=0.8, alpha=0.5)
    ax.set_xlabel('Time (min)', fontsize=12)
    ax.set_ylabel('ρ₁ (veh/km/lane)', fontsize=12)
    ax.set_title('3-hour Simulation: ρ₁ with Tube-Switching Overlay',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(HERE, 'analysis_plot3_state_evolution.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved analysis_plot3_state_evolution.png")

    # ── Plot A4: Tube index timeline + average density ─────────────────
    fig, ax1 = plt.subplots(figsize=(12, 4))
    rho_mean = np.mean(a3_res['x'][:, :mn.N_SEG], axis=1)
    ax1.plot(t_min, rho_mean / mn.RHO_CRIT, lw=2, color='#1976D2',
             label='avg ρ / ρ_crit')
    ax1.set_ylabel('avg ρ / ρ_crit', fontsize=12, color='#1976D2')
    ax1.set_xlabel('Time (min)', fontsize=12)
    ax1.tick_params(axis='y', colors='#1976D2')

    ax2 = ax1.twinx()
    ax2.step(t_min, idx_arr, where='post', lw=2, color='#2E7D32',
             label='active tube factor')
    ax2.set_ylabel('Active tube factor', fontsize=12, color='#2E7D32')
    ax2.tick_params(axis='y', colors='#2E7D32')
    for (step, old, new, safe) in a3_res['switches']:
        c = 'green' if safe else 'red'
        t_sw = a3_res['t'][step] / 60.0
        ax2.axvline(t_sw, color=c, ls=':', lw=0.7, alpha=0.6)

    ax1.set_title('Tube Switch Log — Active Tube vs Density',
                  fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(HERE, 'analysis_plot4_tube_log.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved analysis_plot4_tube_log.png")

    # ── Plot A5: Queue + control action ────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    w = a3_res['x'][:, 2 * mn.N_SEG]
    axes[0].plot(t_min, w, lw=2, color='#FF8F00', label='queue w₁')
    axes[0].axhline(mn.W_MAX, color='red', ls='--', label='w_max')
    axes[0].set_ylabel('Queue (veh)', fontsize=11)
    axes[0].set_title('On-ramp queue', fontsize=12, fontweight='bold')
    axes[0].legend(fontsize=9); axes[0].grid(True, alpha=0.3)

    axes[1].plot(t_min, a3_res['r'], lw=2, color='#2E7D32', label='r₁ (tube-filtered)')
    axes[1].set_ylabel('Metering rate', fontsize=11)
    axes[1].set_xlabel('Time (min)', fontsize=12)
    axes[1].set_title('Ramp metering action', fontsize=12, fontweight='bold')
    axes[1].set_ylim(0, 1.05)
    axes[1].legend(fontsize=9); axes[1].grid(True, alpha=0.3)

    plt.suptitle('3-hour Simulation: Queue & Control',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(HERE, 'analysis_plot5_queue_ctrl.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved analysis_plot5_queue_ctrl.png")

    # ── Plot A6: Phase portrait (ρ₁, v₁) with tube ellipses ──────────
    v1 = a3_res['x'][:, mn.N_SEG]
    fig, ax = plt.subplots(figsize=(9, 7))

    # Draw lookup-table ellipses
    table = a3_res['table']
    stable_table = [e for e in table if e['stable']
                    and e['hw'] is not None
                    and e['hw'][0] < mn.RHO_MAX / 2]
    palette = plt.cm.cool(np.linspace(0, 1, len(stable_table)))
    for e, col in zip(stable_table, palette):
        sel = [0, mn.N_SEG]
        P_sub = e['P'][np.ix_(sel, sel)]
        vals, vecs = np.linalg.eigh(P_sub)
        vals = np.maximum(vals, 1e-12)
        w, h = 2 * np.sqrt(e['c'] * vals)
        angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
        ell = Ellipse((e['rho_op'], e['v_op']), w, h, angle=angle,
                      facecolor=col, edgecolor=col, alpha=0.18, lw=1.5)
        ax.add_patch(ell)
        ax.plot(e['rho_op'], e['v_op'], 'o', color=col, ms=5)

    # Trajectory coloured by time
    points = np.column_stack([rho1, v1]).reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap='plasma', linewidths=1.8)
    lc.set_array(t_min[:-1])
    ax.add_collection(lc)
    cbar = plt.colorbar(lc, ax=ax, label='Time (min)')

    ax.axvline(mn.RHO_CRIT, color='red', ls='--', alpha=0.5, label='ρ_crit')
    ax.set_xlabel('ρ₁ (veh/km/lane)', fontsize=12)
    ax.set_ylabel('v₁ (km/h)', fontsize=12)
    ax.set_title('Phase Portrait — Trajectory Through Tube Ellipses',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, loc='lower left')
    ax.grid(True, alpha=0.3)
    # Set axis limits around the data
    pad_rho, pad_v = 5, 10
    ax.set_xlim(rho1.min() - pad_rho, max(rho1.max(), mn.RHO_CRIT) + pad_rho)
    ax.set_ylim(v1.min() - pad_v, v1.max() + pad_v)
    plt.tight_layout()
    plt.savefig(os.path.join(HERE, 'analysis_plot6_phase_portrait.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved analysis_plot6_phase_portrait.png")

    # ── Summary ────────────────────────────────────────────────────────
    banner("CONSOLIDATED SUMMARY")

    # A1
    n_safe_a1 = sum(1 for r in a1_res if r['verdict'] in ('SAFE', 'MARGINAL'))
    print(f"  Analysis 1 — Formal reachability:")
    print(f"    {n_safe_a1}/{len(a1_res)} transitions feasible "
          f"(SAFE or MARGINAL)")

    # A2
    print(f"\n  Analysis 2 — Linearization error:")
    for h in a2.HORIZONS:
        ratios = [a2_res[(e['factor'], h)]['max_ratio_overall']
                  for e in a2_stable if (e['factor'], h) in a2_res]
        ok = sum(1 for r in ratios if r < 0.30)
        print(f"    {h:>2}-step: max ratio={max(ratios):.4f}, "
              f"{ok}/{len(ratios)} ≤ 0.30")

    # A3
    st = a3_res['stats']
    print(f"\n  Analysis 3 — Multi-regime simulation:")
    print(f"    max density  = {st['max_rho']:.2f} / ρ_crit={mn.RHO_CRIT}")
    print(f"    max queue    = {st['max_w']:.1f} / w_max={mn.W_MAX}")
    print(f"    ρ violations = {st['steps_over']}")
    print(f"    tube switches= {st['n_switches']}  "
          f"({st['n_safe']} safe, {st['n_unsafe']} unsafe)")
    print(f"    TTS          = {st['tts']:.2f} veh·h")

    # Overall
    c1 = n_safe_a1 / max(len(a1_res), 1) >= 0.70
    max_1step = max(a2_res[(e['factor'], 1)]['max_ratio_overall']
                    for e in a2_stable if (e['factor'], 1) in a2_res)
    max_5step = max(a2_res[(e['factor'], 5)]['max_ratio_overall']
                    for e in a2_stable if (e['factor'], 5) in a2_res)
    c2 = max_1step < 0.30 and max_5step < 0.50
    c3 = st['steps_over'] == 0 and st['n_unsafe'] == 0

    print(f"\n  Pass criteria:")
    print(f"    A1 ≥70% feasible     : {n_safe_a1}/{len(a1_res)}"
          f"  [{'PASS' if c1 else 'FAIL'}]")
    print(f"    A2 1-step<0.30       : {max_1step:.4f}"
          f"  [{'PASS' if max_1step < 0.30 else 'FAIL'}]")
    print(f"    A2 5-step<0.50       : {max_5step:.4f}"
          f"  [{'PASS' if max_5step < 0.50 else 'FAIL'}]")
    print(f"    A3 0 violations      : {st['steps_over']}"
          f"  [{'PASS' if st['steps_over'] == 0 else 'FAIL'}]")
    print(f"    A3 all switches safe : {st['n_unsafe']} unsafe"
          f"  [{'PASS' if st['n_unsafe'] == 0 else 'FAIL'}]")

    print(f"\n  ==> OVERALL: {'PASS' if (c1 and c2 and c3) else 'CONDITIONAL'}")
    print(f"\n  Wall time: {time.time() - t0:.1f}s")


if __name__ == '__main__':
    main()
