"""
Run all four scaled-tube experiments and generate six publication plots.
"""
import os
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

import metanet6 as mn
import experiment1_6seg_rpi as ex1
import experiment2_lookup_table as ex2
import experiment3_tube_switching as ex3
import experiment4_demand_scenario as ex4


HERE = os.path.dirname(os.path.abspath(__file__))


def banner(title):
    print("\n" + "=" * 72)
    print("  " + title)
    print("=" * 72)


def main():
    t_global = time.time()

    banner("EXPERIMENT 1  —  6-segment RPI")
    res1 = ex1.run()

    banner("EXPERIMENT 2  —  Lookup table across 12 operating points")
    table, t_lookup = ex2.run()

    banner("EXPERIMENT 3  —  Tube switching reachability")
    _, pairs = ex3.run()

    banner("EXPERIMENT 4  —  2-hour demand scenario (ALINEA vs +Tube vs Aggressive)")
    sim = ex4.run()
    alinea = sim['alinea']
    alinea_tube = sim['alinea_tube']
    aggressive = sim['aggressive']
    summaries = sim['summaries']

    # ── Plot 1: RPI width vs operating density ──────────────────────────
    stable = [e for e in table if e['stable'] and e['hw'][0] < mn.RHO_MAX / 2]
    fx = np.array([e['factor'] for e in stable])
    rho_op = np.array([e['rho_op'] for e in stable])
    hw_rho1 = np.array([e['hw'][0] for e in stable])
    hw_rho_mean = np.array([np.mean(e['hw'][:mn.N_SEG]) for e in stable])
    hw_v_mean = np.array([np.mean(e['hw'][mn.N_SEG:2 * mn.N_SEG]) for e in stable])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(fx, hw_rho1, 'o-', label='ρ₁ tube half-width', color='#2176AE', lw=2)
    ax.plot(fx, hw_rho_mean, 's--', label='mean ρ tube half-width',
            color='#1B5E20', lw=2)
    ax.axvline(1.0, color='gray', ls=':', label='ρ_crit')
    ax.set_xlabel('Operating density  ρ_op / ρ_crit', fontsize=12)
    ax.set_ylabel('RPI half-width  (veh/km/lane)', fontsize=12)
    ax.set_title('Plot 1: State-dependent density tube width',
                 fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(HERE, 'plot1_rpi_width_vs_density.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved plot1_rpi_width_vs_density.png")

    # ── Plot 2: remaining fraction vs operating density ─────────────────
    frac_avg = np.array([e['frac'].mean() for e in stable]) * 100
    frac_min = np.array([e['frac'].min() for e in stable]) * 100

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(fx, frac_avg, 'o-', lw=2, color='#2E7D32', label='average dimension')
    ax.plot(fx, frac_min, 's-', lw=2, color='#C62828', label='worst dimension')
    ax.axhline(20, color='red', ls='--', alpha=0.7, label='20% threshold')
    ax.axvline(1.0, color='gray', ls=':', alpha=0.7)
    ax.set_xlabel('Operating density  ρ_op / ρ_crit', fontsize=12)
    ax.set_ylabel('Remaining constraint fraction (%)', fontsize=12)
    ax.set_title('Plot 2: Tightened-constraint headroom vs operating point',
                 fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 105)
    plt.tight_layout()
    plt.savefig(os.path.join(HERE, 'plot2_remaining_fraction.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved plot2_remaining_fraction.png")

    # ── Plot 3: spectral radius vs operating density ────────────────────
    all_stable_or_not = [e for e in table]
    fx_all = np.array([e['factor'] for e in all_stable_or_not])
    spA = np.array([e['spectral_A'] for e in all_stable_or_not])

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ['#2E7D32' if e['stable'] else '#C62828' for e in all_stable_or_not]
    ax.scatter(fx_all, spA, c=colors, s=80, edgecolor='black', zorder=3,
               label=None)
    ax.plot(fx_all, spA, '--', color='#666', alpha=0.5)
    ax.axhline(1.0, color='red', ls='--', label='stability boundary')
    ax.axvline(1.0, color='gray', ls=':', alpha=0.7)
    ax.set_xlabel('Operating density  ρ_op / ρ_crit', fontsize=12)
    ax.set_ylabel('Spectral radius ρ(A)', fontsize=12)
    ax.set_title('Plot 3: Open-loop stability vs operating point',
                 fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(HERE, 'plot3_spectral_radius.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved plot3_spectral_radius.png")

    # ── Plot 4: 2D tube switching visualization ─────────────────────────
    # Pick 4 adjacent stable entries and show their ellipses in (ρ1, v1)
    chosen = stable[:4] if len(stable) >= 4 else stable
    fig, ax = plt.subplots(figsize=(9, 7))
    palette = ['#2176AE', '#2E7D32', '#FF8F00', '#7B1FA2']

    for idx, (e, col) in enumerate(zip(chosen, palette)):
        P = e['P']
        c = e['c']
        # Extract 2x2 sub-matrix for (ρ1, v1): indices 0 and N_SEG
        sel = [0, mn.N_SEG]
        P_sub = P[np.ix_(sel, sel)]
        # Ellipse axes from eigendecomp
        vals, vecs = np.linalg.eigh(P_sub)
        vals = np.maximum(vals, 1e-12)
        width, height = 2 * np.sqrt(c * vals)
        angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
        centre = (e['rho_op'], e['v_op'])
        ell = Ellipse(centre, width, height, angle=angle,
                      facecolor=col, edgecolor=col, alpha=0.25,
                      lw=2, label=f'Z at ρ/ρc={e["factor"]:.2f}')
        ax.add_patch(ell)
        ax.plot(*centre, 'o', color=col, ms=8)

        # Reachable set halfwidths (bounding box)
        hw_R = ex3.reach_halfwidths(e, np.array([mn.DELTA_D, mn.DELTA_Q]))
        # Show reach box (rectangle)
        from matplotlib.patches import Rectangle
        rect = Rectangle((centre[0] - hw_R[0], centre[1] - hw_R[mn.N_SEG]),
                          2 * hw_R[0], 2 * hw_R[mn.N_SEG],
                          fill=False, edgecolor=col, ls='--', lw=1.5)
        ax.add_patch(rect)

    ax.set_xlabel('Segment-1 density ρ₁ (veh/km/lane)', fontsize=12)
    ax.set_ylabel('Segment-1 speed v₁ (km/h)', fontsize=12)
    ax.set_title('Plot 4: Tube switching — RPI ellipses + one-step reachable boxes',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(HERE, 'plot4_tube_switching.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved plot4_tube_switching.png")

    # ── Plot 5: mainline density — three-controller comparison ─────────
    t_min = alinea['t'] / 60.0
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True, sharey=True)
    series = [('ALINEA', alinea, '#1976D2'),
              ('ALINEA + Tube', alinea_tube, '#2E7D32'),
              ('Aggressive (RL proxy)', aggressive, '#C62828')]
    for ax, (name, sim_r, col) in zip(axes, series):
        for i in range(mn.N_SEG):
            ax.plot(t_min, sim_r['x'][:, i], lw=1.0, alpha=0.5)
        ax.plot(t_min, sim_r['x'][:, 0], lw=2.0, color=col, label='ρ₁')
        ax.axhline(mn.RHO_CRIT, color='red', ls='--', alpha=0.7,
                   label='ρ_crit')
        ax.fill_between(t_min, mn.RHO_CRIT,
                         np.maximum(sim_r['x'][:, 0], mn.RHO_CRIT),
                         where=sim_r['x'][:, 0] > mn.RHO_CRIT,
                         color='red', alpha=0.15, label='violation zone')
        ax.set_ylabel('ρ (veh/km/lane)', fontsize=11)
        ax.set_title(name, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc='upper right')
    axes[-1].set_xlabel('Time (minutes)', fontsize=11)
    plt.suptitle('Plot 5: Mainline density during the demand surge',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(HERE, 'plot5_density_evolution.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved plot5_density_evolution.png")

    # ── Plot 6: safety analysis (queue, ramp rate, cum violations) ──────
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    # Queue length
    for name, sim_r, col in series:
        w = sim_r['x'][:, 2 * mn.N_SEG]
        axes[0].plot(t_min, w, lw=2, color=col, label=name)
    axes[0].axhline(mn.W_MAX, color='red', ls='--', label='w_max')
    axes[0].set_ylabel('Queue w₁ (veh)', fontsize=11)
    axes[0].set_title('On-ramp queue length', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3); axes[0].legend(fontsize=9)

    # Ramp metering rate
    for name, sim_r, col in series:
        axes[1].plot(t_min, sim_r['r'], lw=2, color=col, label=name)
    axes[1].set_ylabel('Metering rate r₁', fontsize=11)
    axes[1].set_title('Ramp metering action', fontsize=12, fontweight='bold')
    axes[1].set_ylim(0, 1.05)
    axes[1].grid(True, alpha=0.3); axes[1].legend(fontsize=9)

    # Cumulative density violations (any segment > ρ_crit)
    for name, sim_r, col in series:
        rho = sim_r['x'][:, :mn.N_SEG]
        viol = np.any(rho > mn.RHO_CRIT, axis=1).astype(int)
        axes[2].plot(t_min, np.cumsum(viol), lw=2, color=col, label=name)
    axes[2].set_ylabel('Cum. steps  ρ > ρ_crit', fontsize=11)
    axes[2].set_xlabel('Time (minutes)', fontsize=11)
    axes[2].set_title('Cumulative constraint violations',
                      fontsize=12, fontweight='bold')
    axes[2].grid(True, alpha=0.3); axes[2].legend(fontsize=9)

    plt.suptitle('Plot 6: Safety analysis — queue, control effort, violations',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(HERE, 'plot6_queue_and_control.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved plot6_queue_and_control.png")

    # ── Plot 7: TTS bar chart + violation bar chart ─────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    names = [s['label'] for s in summaries]
    tts_vals = [s['tts_total'] for s in summaries]
    viol_vals = [s['steps_rho_crit'] for s in summaries]
    colors = ['#1976D2', '#2E7D32', '#C62828']

    bars = axes[0].bar(names, tts_vals, color=colors, edgecolor='black')
    axes[0].set_ylabel('Total TTS (veh·h)', fontsize=11)
    axes[0].set_title('Performance — total time spent', fontsize=12,
                      fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='y')
    for b, v in zip(bars, tts_vals):
        axes[0].text(b.get_x() + b.get_width() / 2, v, f'{v:.1f}',
                     ha='center', va='bottom', fontsize=10)

    bars2 = axes[1].bar(names, viol_vals, color=colors, edgecolor='black')
    axes[1].set_ylabel('Steps with ρ > ρ_crit', fontsize=11)
    axes[1].set_title('Safety — density violations', fontsize=12,
                      fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    for b, v in zip(bars2, viol_vals):
        axes[1].text(b.get_x() + b.get_width() / 2, v, f'{v}',
                     ha='center', va='bottom', fontsize=10)

    plt.suptitle('Plot 7: Performance vs safety trade-off',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(HERE, 'plot7_tts_vs_safety.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved plot7_tts_vs_safety.png")

    # ── Verdict ─────────────────────────────────────────────────────────
    banner("FEASIBILITY VERDICT")

    c1 = res1['t_total'] < 10 and res1['frac'].mean() > 0.20
    n_stable = sum(1 for e in table if e['stable'])
    c2 = n_stable >= 8
    pairs_with_verdict = [p for p in pairs if 'verdict' in p]
    n_good = sum(1 for p in pairs_with_verdict if p['verdict'] in ('YES', 'MARGINAL'))
    c3 = (n_good / max(len(pairs_with_verdict), 1)) >= 0.70

    s_al = summaries[0]
    s_alt = summaries[1]
    s_ag = summaries[2]
    # Tube must yield zero violations AND TTS within 10% of ALINEA
    tts_cost = (s_alt['tts_total'] - s_al['tts_total']) / s_al['tts_total'] * 100
    c4 = (s_alt['steps_rho_crit'] == 0
          and s_al['steps_rho_crit'] > 0
          and abs(tts_cost) < 10.0)

    print(f"  Exp 1  : 6-seg RPI in {res1['t_total']:.2f}s, "
          f"avg remaining = {res1['frac'].mean()*100:.1f}%   "
          f"[{'PASS' if c1 else 'FAIL'}]")
    print(f"  Exp 2  : {n_stable}/12 stable operating points   "
          f"[{'PASS' if c2 else 'FAIL'}]")
    print(f"  Exp 3  : {n_good}/{len(pairs_with_verdict)} feasible transitions "
          f"between stable pairs   [{'PASS' if c3 else 'FAIL'}]")
    print(f"  Exp 4  : violations — ALINEA={s_al['steps_rho_crit']}, "
          f"ALINEA+Tube={s_alt['steps_rho_crit']}, "
          f"Aggressive={s_ag['steps_rho_crit']};   "
          f"TTS cost of tube = {tts_cost:+.2f}%   "
          f"[{'PASS' if c4 else 'FAIL'}]")

    print(f"\n  {'Controller':<18}{'TTS':>10}{'MaxRho':>10}{'MaxQ':>10}"
          f"{'RhoViol':>10}{'QViol':>10}")
    print("  " + "-" * 68)
    for s in summaries:
        print(f"  {s['label']:<18}{s['tts_total']:>10.2f}"
              f"{s['max_rho_any']:>10.2f}{s['max_w']:>10.1f}"
              f"{s['steps_rho_crit']:>10d}{s['steps_w_max']:>10d}")

    all_pass = c1 and c2 and c3 and c4
    verdict = "YES" if all_pass else ("CONDITIONAL" if (c1 and c2 and (c3 or c4)) else "NO")
    print(f"\n  ==> OVERALL: {verdict}")

    print(f"\n  Total wall time: {time.time() - t_global:.2f}s")


if __name__ == '__main__':
    main()
