"""
4 presentation-quality plots for supervisor meeting (April 9, 2026).

  P1: Spectral radius — RM-only vs RM+VSL (VSL "rescue" zone)
  P2: Safety at Zero Cost — TTS vs violations bar chart
  P3: Tube Adapts to Traffic Physics — state-dependent width
  P4: Framework Architecture — conceptual block diagram
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

import metanet6 as mn
import experiment2_lookup_table as ex2
import experiment_vsl_lookup as vsl_lut
import experiment4_demand_scenario as ex4
import experiment_vsl_comparison as ex_vsl

# ── Color palette ────────────────────────────────────────────────────
BLUE   = '#2196F3'
GREEN  = '#4CAF50'
RED    = '#F44336'
ORANGE = '#FF9800'
GREY   = '#9E9E9E'
DARK   = '#263238'

FIGSIZE = (19.20, 10.80)
DPI = 300
FONT = {'family': 'sans-serif', 'size': 14}
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 14,
    'axes.titlesize': 18,
    'axes.labelsize': 15,
    'legend.fontsize': 13,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.grid': True,
    'grid.alpha': 0.3,
})


# =====================================================================
# P1: Spectral Radius — RM-only vs RM+VSL
# =====================================================================
def plot_p1(table_rm, table_vsl):
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)

    factors = [e['factor'] for e in table_rm]
    rho_rm = [e['spectral_Acl'] if e.get('spectral_Acl') else e['spectral_A']
              for e in table_rm]
    rho_vsl = [e['spectral_Acl'] for e in table_vsl]

    ax.plot(factors, rho_rm, 'o-', color=BLUE, ms=8, lw=2.5,
            label='RM-only (closed-loop)', zorder=3)
    ax.plot(factors, rho_vsl, 's-', color=GREEN, ms=8, lw=2.5,
            label='RM + VSL (closed-loop)', zorder=3)

    ax.axhline(1.0, color=RED, ls='--', lw=2, alpha=0.8, label='Stability boundary')
    ax.axvline(1.0, color=GREY, ls=':', lw=1.5, alpha=0.6)

    # Shade the "rescue zone" — only where RM-only is unstable
    unstable_factors = [e['factor'] for e in table_rm if not e['stable']]
    if unstable_factors:
        rescue_lo = min(unstable_factors)
        rescue_hi = max(unstable_factors)
        ax.axvspan(rescue_lo - 0.02, rescue_hi + 0.02, alpha=0.12, color=GREEN,
                   label='VSL rescue zone')

    # Annotate rescued points
    rescued = [(table_rm[i], table_vsl[i]) for i in range(len(factors))
               if not table_rm[i]['stable'] and table_vsl[i]['stable']]
    for rm_e, vsl_e in rescued:
        ax.annotate(f"rescued\n({vsl_e['spectral_Acl']:.3f})",
                    xy=(vsl_e['factor'], vsl_e['spectral_Acl']),
                    xytext=(vsl_e['factor'] + 0.06, vsl_e['spectral_Acl'] - 0.08),
                    fontsize=11, color=GREEN, fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color=GREEN, lw=1.5))

    ax.set_xlabel(r'Operating density factor  $\rho_{op} / \rho_{crit}$')
    ax.set_ylabel('Closed-loop spectral radius  $\\rho(A_{cl})$')
    ax.set_title('VSL Extends Tube MPC to Congested Regime')
    ax.legend(loc='upper left', framealpha=0.9)
    ax.set_xlim(0.1, max(factors) + 0.05)
    ax.set_ylim(0.4, 1.25)

    fig.tight_layout()
    fig.savefig('plot_p1_spectral_vsl.png', dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print("  Saved plot_p1_spectral_vsl.png")


# =====================================================================
# P2: Safety at Zero Cost — TTS + violations comparison
# =====================================================================
def plot_p2(sums_rm, sums_vsl):
    """Bar chart comparing controllers on TTS and safety violations."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGSIZE, dpi=DPI)

    # RM-only results (experiment4)
    labels_rm = [s['label'] for s in sums_rm]
    tts_rm = [s['tts_total'] for s in sums_rm]
    viol_rm = [s['steps_rho_crit'] for s in sums_rm]
    colors_rm = [BLUE, GREEN, RED]

    # Left: TTS comparison
    x = np.arange(len(labels_rm))
    w = 0.35
    bars1 = ax1.bar(x - w/2, tts_rm, w, color=colors_rm, alpha=0.85,
                    edgecolor='white', linewidth=1.5, label='RM-only')

    # VSL results
    labels_vsl = ['ALINEA+VSL', 'ALINEA+VSL\n+Tube', 'Aggressive']
    tts_vsl = [sums_vsl['ALINEA+VSL']['tts_total'],
               sums_vsl['ALINEA+VSL+Tube']['tts_total'],
               sums_vsl['Aggressive']['tts_total']]
    bars2 = ax1.bar(x + w/2, tts_vsl, w, color=colors_rm, alpha=0.45,
                    edgecolor=colors_rm, linewidth=1.5, hatch='///',
                    label='RM+VSL')

    ax1.set_xticks(x)
    ax1.set_xticklabels(['ALINEA', 'ALINEA\n+Tube', 'Aggressive'])
    ax1.set_ylabel('Total Time Spent (veh·h)')
    ax1.set_title('Throughput (lower is better)')
    ax1.legend()

    # Add value labels
    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., h + 5,
                 f'{h:.0f}', ha='center', va='bottom', fontsize=11)

    # Right: Violations comparison
    viol_vsl = [sums_vsl['ALINEA+VSL']['steps_rho_crit'],
                sums_vsl['ALINEA+VSL+Tube']['steps_rho_crit'],
                sums_vsl['Aggressive']['steps_rho_crit']]
    bars3 = ax2.bar(x - w/2, viol_rm, w, color=colors_rm, alpha=0.85,
                    edgecolor='white', linewidth=1.5, label='RM-only')
    bars4 = ax2.bar(x + w/2, viol_vsl, w, color=colors_rm, alpha=0.45,
                    edgecolor=colors_rm, linewidth=1.5, hatch='///',
                    label='RM+VSL')

    ax2.set_xticks(x)
    ax2.set_xticklabels(['ALINEA', 'ALINEA\n+Tube', 'Aggressive'])
    ax2.set_ylabel(r'Steps with $\rho > \rho_{crit}$')
    ax2.set_title('Safety Violations (lower is better)')
    ax2.legend()

    for bar in list(bars3) + list(bars4):
        h = bar.get_height()
        if h > 0:
            ax2.text(bar.get_x() + bar.get_width()/2., h + 3,
                     f'{h}', ha='center', va='bottom', fontsize=11)
        else:
            ax2.text(bar.get_x() + bar.get_width()/2., 3,
                     '0', ha='center', va='bottom', fontsize=12,
                     fontweight='bold', color=GREEN)

    # Highlight the zero-violation results (both RM-only and VSL tube variants)
    ax2.annotate('Zero violations!\nSafety guaranteed',
                 xy=(1 - w/2, 0), xytext=(0.2, max(viol_rm)*0.55),
                 fontsize=14, fontweight='bold', color=GREEN,
                 arrowprops=dict(arrowstyle='->', color=GREEN, lw=2))

    fig.suptitle('Safety at Zero Cost: Tube Filter Guarantees Safety Without Hurting Throughput',
                 fontsize=20, fontweight='bold', y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig('plot_p2_safety_zero_cost.png', dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print("  Saved plot_p2_safety_zero_cost.png")


# =====================================================================
# P3: Tube Adapts to Traffic Physics — state-dependent tube width
# =====================================================================
def plot_p3(table):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGSIZE, dpi=DPI)

    stable = [e for e in table if e['stable'] and e['hw'] is not None
              and e['hw'][0] < mn.RHO_MAX / 2]

    factors = [e['factor'] for e in stable]
    rho_ops = [e['rho_op'] for e in stable]
    hw_rho1 = [e['hw'][0] for e in stable]
    hw_v1 = [e['hw'][mn.N_SEG] for e in stable]
    spectral = [e.get('spectral_Acl', e.get('spectral_A', 0)) for e in stable]

    # Left: Tube half-width vs operating density
    color_rho = BLUE
    color_v = ORANGE
    ax1.plot(rho_ops, hw_rho1, 'o-', color=color_rho, ms=7, lw=2,
             label=r'$\rho_1$ half-width (veh/km/lane)')
    ax1_twin = ax1.twinx()
    ax1_twin.plot(rho_ops, hw_v1, 's-', color=color_v, ms=7, lw=2,
                  label=r'$v_1$ half-width (km/h)')

    ax1.axvline(mn.RHO_CRIT, color=RED, ls='--', lw=1.5, alpha=0.7,
                label=r'$\rho_{crit}$')

    # Shade the free-flow vs congested regions
    ax1.axvspan(0, mn.RHO_CRIT, alpha=0.05, color=GREEN)
    ax1.axvspan(mn.RHO_CRIT, max(rho_ops) + 2, alpha=0.05, color=RED)
    ax1.text(mn.RHO_CRIT * 0.5, max(hw_rho1) * 0.9, 'Free-flow',
             ha='center', fontsize=13, color=GREEN, fontweight='bold', alpha=0.7)
    ax1.text(mn.RHO_CRIT * 1.15, max(hw_rho1) * 0.9, 'Congested',
             ha='center', fontsize=13, color=RED, fontweight='bold', alpha=0.7)

    ax1.set_xlabel(r'Operating density $\rho_{op}$ (veh/km/lane)')
    ax1.set_ylabel(r'$\rho_1$ tube half-width (veh/km/lane)', color=color_rho)
    ax1_twin.set_ylabel(r'$v_1$ tube half-width (km/h)', color=color_v)
    ax1.set_title('Tube Width Grows Near Critical Density')
    ax1.tick_params(axis='y', labelcolor=color_rho)
    ax1_twin.tick_params(axis='y', labelcolor=color_v)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    # Right: Spectral radius drives tube width
    ax2.plot(spectral, hw_rho1, 'o', color=BLUE, ms=9, zorder=3)
    ax2.set_xlabel(r'Closed-loop spectral radius $\rho(A_{cl})$')
    ax2.set_ylabel(r'$\rho_1$ tube half-width (veh/km/lane)')
    ax2.set_title('Spectral Radius Drives Tube Size')

    # Annotate a few key points
    for e, hw, sp in zip(stable, hw_rho1, spectral):
        if e['factor'] in [0.30, 0.60, 0.90]:
            ax2.annotate(f"f={e['factor']:.2f}",
                         xy=(sp, hw), xytext=(sp + 0.02, hw + 0.3),
                         fontsize=11, arrowprops=dict(arrowstyle='->', lw=1))

    ax2.axvline(1.0, color=RED, ls='--', lw=1.5, alpha=0.7,
                label='Stability limit')
    ax2.legend()

    fig.suptitle('Tube Adapts to Traffic Physics: Tighter in Free-Flow, Wider Near Congestion',
                 fontsize=20, fontweight='bold', y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig('plot_p3_tube_physics.png', dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print("  Saved plot_p3_tube_physics.png")


# =====================================================================
# P4: Framework Architecture — block diagram
# =====================================================================
def plot_p4():
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.set_aspect('equal')
    ax.axis('off')

    def box(x, y, w, h, text, color, fontsize=14, subtext=None):
        rect = mpatches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.15",
            facecolor=color, edgecolor=DARK, linewidth=2, alpha=0.85)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2 + (0.15 if subtext else 0),
                text, ha='center', va='center',
                fontsize=fontsize, fontweight='bold', color='white')
        if subtext:
            ax.text(x + w/2, y + h/2 - 0.25, subtext,
                    ha='center', va='center', fontsize=11, color='white',
                    alpha=0.85)

    def arrow(x1, y1, x2, y2, text='', color=DARK):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color=color, lw=2.5))
        if text:
            mx, my = 0.5*(x1+x2), 0.5*(y1+y2)
            ax.text(mx, my + 0.2, text, ha='center', va='bottom',
                    fontsize=11, color=color, fontweight='bold')

    # Title
    ax.text(8, 8.5, 'SDD-TMPC + RL  Framework Architecture',
            ha='center', va='center', fontsize=24, fontweight='bold',
            color=DARK)

    # Blocks
    box(0.5, 5.5, 3.0, 1.5, 'RL Agent', RED,
        subtext='learns cost-to-go')
    box(4.5, 5.5, 3.5, 1.5, 'Tube Safety\nFilter', GREEN,
        subtext='projects onto RPI')
    box(9.0, 5.5, 3.5, 1.5, 'METANET\n+ VSL', BLUE,
        subtext='freeway plant')
    box(13.5, 5.5, 2.0, 1.5, 'Traffic\nState', ORANGE)

    # Lookup table
    box(4.5, 2.5, 3.5, 1.5, 'RPI Lookup\nTable', '#7B1FA2',
        subtext='24 operating points')

    # Feedback
    arrow(3.5, 6.25, 4.5, 6.25, text=r'$u_{RL}$')
    arrow(8.0, 6.25, 9.0, 6.25, text=r'$u_{safe}$')
    arrow(12.5, 6.25, 13.5, 6.25, text=r'$x_{k+1}$')

    # State feedback loop
    ax.annotate('', xy=(0.5, 6.25), xytext=(13.5, 7.5),
                arrowprops=dict(arrowstyle='->', color=DARK, lw=2,
                                connectionstyle='arc3,rad=0.3'))
    ax.text(7.5, 8.0, r'state feedback $x_k$',
            ha='center', fontsize=12, color=DARK, fontstyle='italic')

    # Lookup arrow
    arrow(6.25, 4.0, 6.25, 5.5, text='nearest tube')

    # Disturbance
    ax.annotate('', xy=(10.75, 7.0), xytext=(10.75, 8.0),
                arrowprops=dict(arrowstyle='->', color=RED, lw=2))
    ax.text(10.75, 8.2, r'disturbances $w_k$',
            ha='center', fontsize=12, color=RED, fontstyle='italic')

    # Key insight box
    insight_rect = mpatches.FancyBboxPatch(
        (9.0, 1.8, ), 6.5, 2.2, boxstyle="round,pad=0.2",
        facecolor='#E8F5E9', edgecolor=GREEN, linewidth=2)
    ax.add_patch(insight_rect)
    ax.text(12.25, 3.5, 'Key Guarantee', ha='center', fontsize=15,
            fontweight='bold', color=GREEN)
    ax.text(12.25, 2.85, r'If $x_k \in \mathcal{T}_j$, then $x_{k+1} \in \mathcal{T}_j$',
            ha='center', fontsize=14, color=DARK)
    ax.text(12.25, 2.3, 'Recursive safety regardless of RL policy',
            ha='center', fontsize=12, color=GREY)

    fig.savefig('plot_p4_architecture.png', dpi=DPI, bbox_inches='tight',
                facecolor='white')
    plt.close(fig)
    print("  Saved plot_p4_architecture.png")


# =====================================================================
# Main
# =====================================================================
def run():
    print("=" * 70)
    print("  Generating presentation plots ...")
    print("=" * 70)

    # Data for P1: spectral radius
    print("\n  [P1] Computing RM-only vs RM+VSL lookup tables ...")
    table_rm, table_vsl = vsl_lut.run(verbose=False)
    plot_p1(table_rm, table_vsl)

    # Data for P2: safety bar chart
    print("  [P2] Running experiment4 (RM-only) + VSL comparison ...")
    res4 = ex4.run(verbose=False)
    sums_rm = res4['summaries']
    _, sums_vsl, _ = ex_vsl.run(verbose=False)
    plot_p2(sums_rm, sums_vsl)

    # Data for P3: tube width
    print("  [P3] Using lookup table for tube width plot ...")
    table_ex2, _ = ex2.run(verbose=False)
    plot_p3(table_ex2)

    # P4: architecture (no data needed)
    print("  [P4] Drawing architecture diagram ...")
    plot_p4()

    print("\n  All 4 plots saved.")


if __name__ == '__main__':
    run()
