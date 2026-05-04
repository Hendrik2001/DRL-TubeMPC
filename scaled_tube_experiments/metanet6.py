"""
6-segment METANET model for Sun et al. (IEEE TITS 2024) benchmark.

State (13-dim): [ρ1..ρ6, v1..v6, w1]
Control: r1 ∈ [0,1] (ramp metering)
Disturbances: d1 (on-ramp demand), q_up (upstream flow)

VSL is not modeled here (kept for a later iteration).
"""
import numpy as np

# --- Parameters (Kotsialos/Sun) ---
T = 10 / 3600          # 10 s in hours
L = 1.0                # km per segment
LAM = 2                # lanes
TAU = 18 / 3600        # h
ETA = 65               # km^2/h
KAPPA = 40             # veh/km/lane
V_FREE = 102           # km/h
RHO_CRIT = 33.5        # veh/km/lane
RHO_MAX = 180          # veh/km/lane
A_PARAM = 1.867
C_O = 2000             # on-ramp capacity (veh/h)
W_MAX = 200            # max queue (veh)
BETA_SMOOTH = 50
DELTA_MERGE = 0.0122

N_SEG = 6
NX = 2 * N_SEG + 1     # 13

# Nominal operating conditions
D_NOMINAL = 500
Q_UPSTREAM = 3000      # well below free-flow capacity (~4000 veh/h)
R_NOMINAL = 0.8

# Disturbance bounds
DELTA_D = 200
DELTA_Q = 500


def Ve(rho):
    rho = np.maximum(rho, 0.01)
    return V_FREE * np.exp(-(1.0 / A_PARAM) * (rho / RHO_CRIT) ** A_PARAM)


def smooth_min3(a, b, c, beta=BETA_SMOOTH):
    vals = np.array([a, b, c], dtype=float)
    m = float(np.min(vals))
    return m - (1.0 / beta) * np.log(
        np.exp(-beta * (a - m)) + np.exp(-beta * (b - m)) + np.exp(-beta * (c - m))
    )


def unpack(state):
    rho = state[:N_SEG].copy()
    v = state[N_SEG:2 * N_SEG].copy()
    w = state[2 * N_SEG]
    return rho, v, w


def pack(rho, v, w):
    out = np.zeros(NX)
    out[:N_SEG] = rho
    out[N_SEG:2 * N_SEG] = v
    out[2 * N_SEG] = w
    return out


def metanet_step(state, r1, d1, q_up):
    """One 10 s METANET time step for the 6-segment network."""
    rho, v, w1 = unpack(state)

    rho = np.clip(rho, 0.01, RHO_MAX)
    v = np.clip(v, 0.01, V_FREE * 1.5)
    w1 = max(w1, 0.0)

    # Mainline flows
    q = rho * v * LAM  # per segment

    # On-ramp flow (segment 1)
    t1 = d1 + w1 / T
    t2 = C_O * r1
    t3 = C_O * (RHO_MAX - rho[0]) / (RHO_MAX - RHO_CRIT)
    q_on1 = max(smooth_min3(t1, t2, t3), 0.0)

    # Upstream boundary speed
    v_up = min(q_up / (LAM * max(rho[0], 0.01)), V_FREE)

    # Equilibrium speeds
    Veq = Ve(rho)

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

    # Queue update
    w1_new = w1 + T * (d1 - q_on1)

    rho_new = np.clip(rho_new, 0.01, RHO_MAX)
    v_new = np.clip(v_new, 0.01, V_FREE * 1.5)
    w1_new = max(w1_new, 0.0)

    return pack(rho_new, v_new, w1_new)


def find_equilibrium(r1=R_NOMINAL, d1=D_NOMINAL, q_up=Q_UPSTREAM,
                     max_iter=200000, tol=1e-9):
    rho0 = q_up / (LAM * V_FREE)
    v0 = V_FREE * 0.95
    state = pack(np.full(N_SEG, rho0), np.full(N_SEG, v0), 0.0)
    for i in range(max_iter):
        new = metanet_step(state, r1, d1, q_up)
        if np.max(np.abs(new - state)) < tol:
            return new, i + 1
        state = new
    return state, max_iter


def linearize(x_eq, r1, d1, q_up, delta=1e-6):
    """Finite-difference Jacobians A=∂f/∂x, Bu=∂f/∂r, Bw=∂f/∂[d,q]."""
    n = NX
    A = np.zeros((n, n))
    Bu = np.zeros((n, 1))
    Bw = np.zeros((n, 2))

    for j in range(n):
        dx = max(delta * abs(x_eq[j]), delta)
        xp = x_eq.copy(); xp[j] += dx
        xm = x_eq.copy(); xm[j] -= dx
        A[:, j] = (metanet_step(xp, r1, d1, q_up) - metanet_step(xm, r1, d1, q_up)) / (2 * dx)

    dr = delta
    Bu[:, 0] = (metanet_step(x_eq, r1 + dr, d1, q_up)
                - metanet_step(x_eq, r1 - dr, d1, q_up)) / (2 * dr)

    dd = max(delta * abs(d1), delta)
    Bw[:, 0] = (metanet_step(x_eq, r1, d1 + dd, q_up)
                - metanet_step(x_eq, r1, d1 - dd, q_up)) / (2 * dd)

    dq = max(delta * abs(q_up), delta)
    Bw[:, 1] = (metanet_step(x_eq, r1, d1, q_up + dq)
                - metanet_step(x_eq, r1, d1, q_up - dq)) / (2 * dq)

    return A, Bu, Bw


def constraint_box():
    """Hard state constraints for the 13-dim state."""
    lo = np.zeros(NX)
    hi = np.zeros(NX)
    hi[:N_SEG] = RHO_MAX
    hi[N_SEG:2 * N_SEG] = V_FREE
    hi[2 * N_SEG] = W_MAX
    return lo, hi
