"""
Run all three feasibility experiments and generate plots + summary.
"""
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from experiment1_rpi import run_experiment1
from experiment2_conservativeness import run_experiment2
from experiment3_sddtmpc import run_experiment3, state_dependent_disturbance

from metanet_model import RHO_CRIT, RHO_MAX, V_FREE, W_MAX

STATE_NAMES = ['ρ1', 'ρ2', 'ρ3', 'v1', 'v2', 'v3', 'w1']

OUTPUT_DIR = '/Users/hendrik/Documents/ScriptieMaster/feasibility_experiments'


def main():
    t_total_start = time.time()

    # ========================
    # Run Experiments
    # ========================
    print("\n" + "#" * 70)
    print("# DIRECTION B FEASIBILITY: SDD-TMPC Safety Envelope for Deep RL")
    print("#" * 70)

    # Experiment 1
    exp1 = run_experiment1()

    # Experiment 2
    exp2 = run_experiment2(exp1)

    # Experiment 3
    exp3 = run_experiment3()

    t_total = time.time() - t_total_start

    # ========================
    # Generate Plots
    # ========================
    print("\n\nGenerating plots...")

    # --- Plot 1: RPI set projected onto (ρ1, v1) plane ---
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    x_eq = exp1['x_eq']
    rpi_widths = exp1['rpi_widths']

    # Draw constraint box
    ax.add_patch(plt.Rectangle((0, 0), RHO_MAX, V_FREE,
                                fill=False, edgecolor='gray', linewidth=2,
                                linestyle='--', label='Constraint set'))

    # Draw tightened constraint box
    z_rho = rpi_widths[0] / 2
    z_v = rpi_widths[3] / 2
    ax.add_patch(plt.Rectangle((z_rho, z_v),
                                RHO_MAX - 2*z_rho, V_FREE - 2*z_v,
                                fill=False, edgecolor='blue', linewidth=2,
                                linestyle='-', label='Tightened constraints'))

    # Draw RPI zonotope (bounding box projection)
    rpi_rect = plt.Rectangle((x_eq[0] - rpi_widths[0]/2, x_eq[3] - rpi_widths[3]/2),
                              rpi_widths[0], rpi_widths[3],
                              fill=True, facecolor='red', alpha=0.3,
                              edgecolor='red', linewidth=2,
                              label='RPI set (zonotopic)')

    ax.add_patch(rpi_rect)

    # Draw ellipsoidal RPI if available
    if 'P_lyap' in exp1:
        P = exp1['P_lyap']
        # Project onto (ρ1, v1) = dimensions (0, 3)
        P_proj = np.array([[P[0,0], P[0,3]], [P[3,0], P[3,3]]])
        eigvals_p, eigvecs_p = np.linalg.eigh(P_proj)
        c = 7.0  # same scaling as experiment 1
        angle = np.degrees(np.arctan2(eigvecs_p[1,0], eigvecs_p[0,0]))
        width = 2 * np.sqrt(c * eigvals_p[0])
        height = 2 * np.sqrt(c * eigvals_p[1])
        ellipse = Ellipse(xy=(x_eq[0], x_eq[3]), width=width, height=height,
                          angle=angle, fill=True, facecolor='green', alpha=0.2,
                          edgecolor='green', linewidth=2, linestyle='--',
                          label='RPI set (ellipsoidal)')
        ax.add_patch(ellipse)

    # Mark equilibrium
    ax.plot(x_eq[0], x_eq[3], 'k*', markersize=15, label=f'Equilibrium', zorder=5)

    # Mark critical density
    ax.axvline(x=RHO_CRIT, color='orange', linestyle=':', linewidth=1.5, label='ρ_crit')

    ax.set_xlabel('Density ρ₁ (veh/km/lane)', fontsize=12)
    ax.set_ylabel('Speed v₁ (km/h)', fontsize=12)
    ax.set_title('Experiment 1: RPI Set Projection onto (ρ₁, v₁) Plane', fontsize=13)
    ax.legend(loc='upper right', fontsize=9)
    ax.set_xlim(-5, RHO_MAX + 10)
    ax.set_ylim(-5, V_FREE + 10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/plot1_rpi_projection.png', dpi=150)
    plt.close()
    print("  Saved plot1_rpi_projection.png")

    # --- Plot 2: Conservativeness bar chart ---
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    fracs = exp2['remaining_fracs']
    colors = ['green' if f > 0.5 else 'orange' if f > 0.2 else 'red' for f in fracs]

    bars = ax.bar(STATE_NAMES, fracs * 100, color=colors, edgecolor='black', alpha=0.8)

    # Add value labels
    for bar, frac in zip(bars, fracs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{frac*100:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.axhline(y=50, color='green', linestyle='--', linewidth=1.5, alpha=0.7, label='50% (substantial room)')
    ax.axhline(y=20, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='20% (minimum viable)')
    ax.axhline(y=30, color='orange', linestyle='--', linewidth=1.5, alpha=0.5, label='30% (queue threshold)')

    ax.set_xlabel('State Dimension', fontsize=12)
    ax.set_ylabel('Remaining Constraint Fraction (%)', fontsize=12)
    ax.set_title('Experiment 2: Tube Conservativeness — Room for RL Optimization', fontsize=13)
    ax.legend(fontsize=9)
    ax.set_ylim(0, 105)
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/plot2_conservativeness.png', dpi=150)
    plt.close()
    print("  Saved plot2_conservativeness.png")

    # --- Plot 3: State-dependent tube width vs operating density ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    if exp3.get('actual_densities') and len(exp3['actual_densities']) > 0:
        rho_arr = np.array(exp3['actual_densities'])
        sdd_arr = np.array(exp3['sdd_tube_widths'])
        fixed_arr = np.array(exp3['fixed_tube_widths'])

        # Left: tube width in density dimension
        ax1.plot(rho_arr, sdd_arr[:, 0], 'bo-', linewidth=2, markersize=6,
                 label='SDD tube (ρ₁ width)')
        ax1.axhline(y=exp3['fixed_widths_nominal'][0], color='r', linestyle='--',
                     linewidth=2, label='Fixed tube (ρ₁ width)')
        ax1.axvline(x=RHO_CRIT, color='orange', linestyle=':', linewidth=1.5, label='ρ_crit')

        ax1.set_xlabel('Operating Density ρ (veh/km/lane)', fontsize=12)
        ax1.set_ylabel('Tube Width in ρ₁ (veh/km/lane)', fontsize=12)
        ax1.set_title('SDD vs Fixed Tube: Density Dimension', fontsize=13)
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)

        # Right: state-dependent disturbance bound
        rho_fine = np.linspace(0.1 * RHO_CRIT, 2.0 * RHO_CRIT, 100)
        delta_d_fine = [state_dependent_disturbance(r) for r in rho_fine]

        ax2.plot(rho_fine, delta_d_fine, 'b-', linewidth=2, label='Δd(ρ) state-dependent')
        ax2.axhline(y=200, color='r', linestyle='--', linewidth=2, label='Δd = 200 (fixed)')
        ax2.axvline(x=RHO_CRIT, color='orange', linestyle=':', linewidth=1.5, label='ρ_crit')

        ax2.set_xlabel('Operating Density ρ (veh/km/lane)', fontsize=12)
        ax2.set_ylabel('Disturbance Bound Δd (veh/h)', fontsize=12)
        ax2.set_title('State-Dependent Disturbance Model', fontsize=13)
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/plot3_sdd_tube_width.png', dpi=150)
    plt.close()
    print("  Saved plot3_sdd_tube_width.png")

    # ========================
    # Final Summary
    # ========================
    print("\n\n" + "=" * 70)
    print("FINAL RESULTS SUMMARY")
    print("=" * 70)

    print(f"\nTotal computation time: {t_total:.1f}s")

    print("\n--- Experiment 1: RPI Set ---")
    print(f"  Equilibrium: ρ = [{exp1['x_eq'][0]:.2f}, {exp1['x_eq'][1]:.2f}, {exp1['x_eq'][2]:.2f}] veh/km/lane")
    print(f"               v = [{exp1['x_eq'][3]:.2f}, {exp1['x_eq'][4]:.2f}, {exp1['x_eq'][5]:.2f}] km/h")
    print(f"               w = {exp1['x_eq'][6]:.2f} veh")
    print(f"  Eigenvalues (|λ|): {', '.join(f'{abs(ev):.4f}' for ev in exp1['eigvals_A'])}")
    if 'eigvals_Acl' in exp1:
        print(f"  CL eigenvalues: {', '.join(f'{abs(ev):.4f}' for ev in exp1['eigvals_Acl'])}")
    print(f"  Zonotopic RPI widths:")
    for j in range(7):
        print(f"    {STATE_NAMES[j]}: {exp1['rpi_widths'][j]:.4f}")
    print(f"  Result: {'PASS' if exp1['success'] else 'FAIL'}")

    print("\n--- Experiment 2: Conservativeness ---")
    print(f"  Average remaining fraction: {exp2['avg_frac']*100:.1f}%")
    print(f"  Queue remaining fraction: {exp2['queue_frac']*100:.1f}%")
    print(f"  Per-dimension: {', '.join(f'{f*100:.1f}%' for f in exp2['remaining_fracs'])}")
    print(f"  Verdict: {exp2['verdict']}")
    print(f"  Result: {'PASS' if exp2['success'] else 'FAIL'}")

    print("\n--- Experiment 3: SDD-TMPC ---")
    if 'ff_improvement' in exp3:
        print(f"  Free-flow SDD improvement: {exp3.get('ff_improvement', 0):.1f}%")
    print(f"  SDD tighter in free-flow: {exp3.get('sdd_tighter_in_ff', 'N/A')}")
    print(f"  Number of operating points tested: {len(exp3.get('actual_densities', []))}")
    print(f"  Result: {'PASS' if exp3['success'] else 'FAIL'}")

    # ========================
    # Feasibility Verdict
    # ========================
    all_pass = exp1['success'] and exp2['success'] and exp3['success']

    print("\n" + "=" * 70)
    print("FEASIBILITY VERDICT")
    print("=" * 70)

    if all_pass and exp2['avg_frac'] > 0.5:
        print("""
    RECOMMENDATION: YES — Proceed with Direction B

    All three experiments passed successfully:
    1. The RPI set is computable, finite, and well within constraints.
    2. The tube is NOT overly conservative — RL retains substantial room
       to optimize within the safety envelope.
    3. State-dependent tube computation is feasible and shows the expected
       behavior: tighter tubes in free-flow, wider near congestion.

    Key strengths of this direction:
    - Safety guarantees via tube MPC are computationally tractable
    - The 7-dimensional METANET state space is manageable
    - State-dependent tubes can reduce conservatism significantly
    - Clear separation of concerns: MPC ensures safety, RL optimizes performance
        """)
    elif all_pass:
        print("""
    RECOMMENDATION: CONDITIONAL YES — Proceed with caution

    All experiments passed, but the tube is moderately conservative.
    Consider:
    - Using state-dependent tubes (SDD-TMPC) to reduce conservatism
    - Focusing RL optimization on the dimensions with most remaining room
    - Investigating tighter RPI approximations (e.g., polytopic methods)
        """)
    elif exp1['success'] and not exp2['success']:
        print("""
    RECOMMENDATION: CONDITIONAL — Tube too conservative for naive approach

    The RPI set is computable but leaves insufficient room for RL.
    Potential fixes:
    - State-dependent tubes (SDD-TMPC) to reduce conservatism
    - Tighter disturbance characterization with online learning
    - Reduced disturbance bounds via better demand prediction
        """)
    else:
        print("""
    RECOMMENDATION: NO — Fundamental obstacles detected

    The RPI computation or stability analysis revealed issues that
    would make this thesis direction risky. Consider Direction A instead,
    or investigate whether a different MPC formulation could work.
        """)

    print(f"\nPlots saved to: {OUTPUT_DIR}/")
    print(f"  - plot1_rpi_projection.png")
    print(f"  - plot2_conservativeness.png")
    print(f"  - plot3_sdd_tube_width.png")

    return exp1, exp2, exp3


if __name__ == '__main__':
    exp1, exp2, exp3 = main()
