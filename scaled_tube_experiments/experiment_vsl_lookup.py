"""
Lookup table recomputation with VSL.
Compares RM-only vs RM+VSL for every operating point, paying special attention
to the previously-unstable congested regime (ρ/ρ_crit ≥ 1.0).
"""
import numpy as np
import time
import metanet6 as mn
import metanet6_vsl as vsl
import tube_tools as tt
from experiment2_lookup_table import RHO_FACTORS, make_operating_point

Delta = np.array([mn.DELTA_D, mn.DELTA_Q])


def run(verbose=True):
    table_rm = []   # RM-only (recomputed for the dense grid)
    table_vsl = []  # RM+VSL
    t0_all = time.time()

    if verbose:
        print(f"  {'f':>5} {'ρ_op':>6}  "
              f"{'ρ(A)_RM':>8} {'stab_RM':>8} {'hw_ρ1_RM':>9}  "
              f"{'ρ(A)_VSL':>9} {'stab_VSL':>9} {'hw_ρ1_VSL':>10}")
        print("  " + "-" * 80)

    for f in RHO_FACTORS:
        rho_op = f * mn.RHO_CRIT
        v_op = mn.Ve(rho_op)
        q_op = rho_op * v_op * mn.LAM
        x_op = mn.pack(np.full(mn.N_SEG, rho_op), np.full(mn.N_SEG, v_op), 0.0)
        d1 = mn.D_NOMINAL
        q_up = q_op

        # --- RM-only ---
        A_rm, Bu_rm, Bw_rm = mn.linearize(x_op, mn.R_NOMINAL, d1, q_up)
        rho_A_rm = tt.spectral_radius(A_rm)
        A_cl_rm, K_rm = tt.stabilize(A_rm, Bu_rm)
        rho_cl_rm = tt.spectral_radius(A_cl_rm)
        stable_rm = rho_cl_rm < 1.0 - 1e-9
        hw_rm = None
        if stable_rm:
            P, c, hw_rm = tt.ellipsoidal_rpi(A_cl_rm, Bw_rm, Delta)

        entry_rm = dict(factor=f, rho_op=rho_op, v_op=v_op, x_op=x_op,
                        q_up=q_up, d1=d1,
                        A=A_rm, Bu=Bu_rm, Bw=Bw_rm, A_cl=A_cl_rm, K=K_rm,
                        spectral_A=rho_A_rm, spectral_Acl=rho_cl_rm,
                        stable=stable_rm, hw=hw_rm,
                        P=P if stable_rm else None,
                        c=c if stable_rm else None)
        table_rm.append(entry_rm)

        # --- RM + VSL ---
        u_op = vsl.nominal_u(rho_op)
        A_v, Bu_v, Bw_v = vsl.linearize(x_op, u_op, d1, q_up)
        rho_A_v = tt.spectral_radius(A_v)
        A_cl_v, K_v = tt.stabilize(A_v, Bu_v,
                                    Q=np.eye(mn.NX),
                                    R=np.diag([10.0, 1.0, 1.0]))
        rho_cl_v = tt.spectral_radius(A_cl_v)
        stable_v = rho_cl_v < 1.0 - 1e-9
        hw_v = None
        P_v, c_v = None, None
        if stable_v:
            P_v, c_v, hw_v = tt.ellipsoidal_rpi(A_cl_v, Bw_v, Delta)

        entry_v = dict(factor=f, rho_op=rho_op, v_op=v_op, x_op=x_op,
                       q_up=q_up, d1=d1, u_op=u_op,
                       A=A_v, Bu=Bu_v, Bw=Bw_v, A_cl=A_cl_v, K=K_v,
                       spectral_A=rho_A_v, spectral_Acl=rho_cl_v,
                       stable=stable_v, hw=hw_v, P=P_v, c=c_v)
        table_vsl.append(entry_v)

        if verbose:
            h_rm = f"{hw_rm[0]:.3f}" if hw_rm is not None else "—"
            h_v = f"{hw_v[0]:.3f}" if hw_v is not None else "—"
            print(f"  {f:>5.2f} {rho_op:>6.1f}  "
                  f"{rho_A_rm:>8.4f} {'YES' if stable_rm else 'NO':>8} {h_rm:>9}  "
                  f"{rho_A_v:>9.4f} {'YES' if stable_v else 'NO':>9} {h_v:>10}")

    # Summary
    n_rm = sum(1 for e in table_rm if e['stable'])
    n_vsl = sum(1 for e in table_vsl if e['stable'])
    rescued = [(table_rm[i], table_vsl[i]) for i in range(len(RHO_FACTORS))
               if not table_rm[i]['stable'] and table_vsl[i]['stable']]

    if verbose:
        print(f"\n  Stable operating points:  RM-only {n_rm}/{len(RHO_FACTORS)},  "
              f"RM+VSL {n_vsl}/{len(RHO_FACTORS)}")
        if rescued:
            print(f"  VSL RESCUED {len(rescued)} operating point(s):")
            for r_rm, r_v in rescued:
                print(f"    factor={r_rm['factor']:.2f}  "
                      f"ρ(A) {r_rm['spectral_A']:.4f} → {r_v['spectral_Acl']:.4f}")
        else:
            print("  VSL did NOT rescue any additional operating points.")
        print(f"  Total time: {(time.time()-t0_all)*1000:.1f} ms")

    return table_rm, table_vsl


if __name__ == '__main__':
    print("=" * 90)
    print("  VSL Lookup Table — RM-only vs RM+VSL comparison")
    print("=" * 90)
    run()
