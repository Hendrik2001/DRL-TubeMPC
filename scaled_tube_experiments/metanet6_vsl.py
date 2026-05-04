"""
6-segment METANET with Variable Speed Limits on segments 3 and 4.

State  (13-dim): [ρ1..ρ6, v1..v6, w1]  (same as before — VSL is an input)
Action  (3-dim): [r1, v_vsl3, v_vsl4]
Disturbances: d1 (on-ramp demand), q_up (upstream flow)

The only change vs metanet6.py: the equilibrium speed on segments 3 and 4 is
replaced by  V_eff(ρ, v_vsl) = smooth_min(V(ρ), v_vsl).
"""
import numpy as np
import metanet6 as mn

# Re-export constants for convenience
T, L, LAM = mn.T, mn.L, mn.LAM
TAU, ETA, KAPPA = mn.TAU, mn.ETA, mn.KAPPA
V_FREE, RHO_CRIT, RHO_MAX, A_PARAM = mn.V_FREE, mn.RHO_CRIT, mn.RHO_MAX, mn.A_PARAM
C_O, W_MAX, BETA_SMOOTH, DELTA_MERGE = mn.C_O, mn.W_MAX, mn.BETA_SMOOTH, mn.DELTA_MERGE
N_SEG, NX = mn.N_SEG, mn.NX
D_NOMINAL, Q_UPSTREAM, R_NOMINAL = mn.D_NOMINAL, mn.Q_UPSTREAM, mn.R_NOMINAL
DELTA_D, DELTA_Q = mn.DELTA_D, mn.DELTA_Q

# VSL configuration
VSL_SEGS = [2, 3]          # 0-indexed → segments 3 and 4
VSL_MIN = 20.0             # km/h
VSL_MAX = V_FREE           # 102 km/h
N_VSL = len(VSL_SEGS)
NU = 1 + N_VSL             # 3 control inputs: r1, v_vsl3, v_vsl4


def smooth_min2(a, b, beta=BETA_SMOOTH):
    m = min(a, b)
    return m - (1.0 / beta) * np.log(
        np.exp(-beta * (a - m)) + np.exp(-beta * (b - m))
    )


def step(state, u, d1, q_up):
    """
    One 10 s METANET+VSL time step.
    u = [r1, v_vsl3, v_vsl4].
    """
    r1 = np.clip(u[0], 0.0, 1.0)
    v_vsl = np.clip(u[1:], VSL_MIN, VSL_MAX)

    rho, v, w1 = mn.unpack(state)
    rho = np.clip(rho, 0.01, RHO_MAX)
    v = np.clip(v, 0.01, V_FREE * 1.5)
    w1 = max(w1, 0.0)

    q = rho * v * LAM

    # On-ramp flow (segment 1)
    t1 = d1 + w1 / T
    t2 = C_O * r1
    t3 = C_O * (RHO_MAX - rho[0]) / (RHO_MAX - RHO_CRIT)
    q_on1 = max(mn.smooth_min3(t1, t2, t3), 0.0)

    v_up = min(q_up / (LAM * max(rho[0], 0.01)), V_FREE)

    # Equilibrium speeds — apply VSL where configured
    Veq = mn.Ve(rho)
    for k, seg in enumerate(VSL_SEGS):
        Veq[seg] = smooth_min2(Veq[seg], v_vsl[k])

    # Density updates
    rho_new = np.zeros(N_SEG)
    for i in range(N_SEG):
        q_in = q_up if i == 0 else q[i - 1]
        q_out = q[i]
        on = q_on1 if i == 0 else 0.0
        rho_new[i] = rho[i] + (T / (L * LAM)) * (q_in - q_out + on)

    # Speed updates
    v_new = np.zeros(N_SEG)
    for i in range(N_SEG):
        dv = (T / TAU) * (Veq[i] - v[i])
        v_prev = v_up if i == 0 else v[i - 1]
        dv += (T / L) * v[i] * (v_prev - v[i])
        if i < N_SEG - 1:
            dv -= (ETA * T / (TAU * L)) * (rho[i + 1] - rho[i]) / (rho[i] + KAPPA)
        v_new[i] = v[i] + dv
        if i == 0:
            v_new[i] -= DELTA_MERGE * T * q_on1 * v[i] / (L * LAM * (rho[i] + KAPPA))

    w1_new = w1 + T * (d1 - q_on1)

    rho_new = np.clip(rho_new, 0.01, RHO_MAX)
    v_new = np.clip(v_new, 0.01, V_FREE * 1.5)
    w1_new = max(w1_new, 0.0)

    return mn.pack(rho_new, v_new, w1_new)


def linearize(x_op, u_op, d1, q_up, delta=1e-6):
    """
    Finite-difference Jacobians.
    A = 13×13, Bu = 13×3, Bw = 13×2.
    """
    n = NX
    nu = NU
    A = np.zeros((n, n))
    Bu = np.zeros((n, nu))
    Bw = np.zeros((n, 2))

    f0 = step(x_op, u_op, d1, q_up)

    for j in range(n):
        dx = max(delta * abs(x_op[j]), delta)
        xp = x_op.copy(); xp[j] += dx
        xm = x_op.copy(); xm[j] -= dx
        A[:, j] = (step(xp, u_op, d1, q_up) - step(xm, u_op, d1, q_up)) / (2 * dx)

    for j in range(nu):
        du = delta if j == 0 else max(delta * abs(u_op[j]), delta)
        up = u_op.copy(); up[j] += du
        um = u_op.copy(); um[j] -= du
        Bu[:, j] = (step(x_op, up, d1, q_up) - step(x_op, um, d1, q_up)) / (2 * du)

    dd = max(delta * abs(d1), delta)
    Bw[:, 0] = (step(x_op, u_op, d1 + dd, q_up) - step(x_op, u_op, d1 - dd, q_up)) / (2 * dd)
    dq = max(delta * abs(q_up), delta)
    Bw[:, 1] = (step(x_op, u_op, d1, q_up + dq) - step(x_op, u_op, d1, q_up - dq)) / (2 * dq)

    return A, Bu, Bw


def nominal_u(rho_op):
    """Nominal action: moderate metering, VSL set to equilibrium speed."""
    v_eq = float(mn.Ve(rho_op))
    return np.array([R_NOMINAL, v_eq, v_eq])


def sanity_check():
    """Quick verification that VSL model is consistent."""
    print("  VSL Sanity Checks:")
    rho_op = 0.6 * RHO_CRIT
    x = mn.pack(np.full(N_SEG, rho_op), np.full(N_SEG, mn.Ve(rho_op)), 0.0)
    u_nolim = np.array([0.8, V_FREE, V_FREE])
    u_lim60 = np.array([0.8, 60.0, 60.0])

    x1_nolim = step(x, u_nolim, D_NOMINAL, Q_UPSTREAM)
    x1_orig = mn.metanet_step(x, 0.8, D_NOMINAL, Q_UPSTREAM)
    err = np.max(np.abs(x1_nolim - x1_orig))
    print(f"    VSL=102 vs original: max diff = {err:.2e}  "
          f"[{'OK' if err < 1e-6 else 'MISMATCH'}]")

    x1_lim = step(x, u_lim60, D_NOMINAL, Q_UPSTREAM)
    v3_nolim = x1_nolim[N_SEG + 2]
    v3_lim = x1_lim[N_SEG + 2]
    print(f"    seg3 speed: VSL=102 → v3={v3_nolim:.2f}, "
          f"VSL=60 → v3={v3_lim:.2f}  "
          f"[{'OK' if v3_lim < v3_nolim else 'NO EFFECT'}]")

    # Congested case: check VSL reduces upstream density
    rho_cong = 1.2 * RHO_CRIT
    x_cong = mn.pack(np.full(N_SEG, rho_cong),
                     np.full(N_SEG, mn.Ve(rho_cong)), 0.0)
    u_vsl40 = np.array([0.5, 40.0, 40.0])
    u_novsl = np.array([0.5, V_FREE, V_FREE])
    # Simulate 50 steps
    xn, xv = x_cong.copy(), x_cong.copy()
    for _ in range(50):
        xn = step(xn, u_novsl, D_NOMINAL, Q_UPSTREAM)
        xv = step(xv, u_vsl40, D_NOMINAL, Q_UPSTREAM)
    rho1_no = xn[0]
    rho1_vsl = xv[0]
    print(f"    Congested (50-step): no-VSL ρ1={rho1_no:.2f}, "
          f"VSL=40 ρ1={rho1_vsl:.2f}  "
          f"[{'OK — VSL helps' if rho1_vsl < rho1_no else 'NO BENEFIT'}]")


if __name__ == '__main__':
    sanity_check()
