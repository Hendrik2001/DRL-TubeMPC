"""
METANET freeway traffic model for Direction A experiments.
3 segments, 1 on-ramp → state vector: [ρ1, ρ2, ρ3, v1, v2, v3, w1] (dim 7)

Parameters from Kotsialos et al. (1999), with η=60 as specified for Direction A.
"""
import numpy as np

# --- Parameters ---
T = 10 / 3600          # sampling time in hours (10 seconds)
L = 1.0                # segment length in km
LAMBDA = 2             # number of lanes
TAU = 18 / 3600        # relaxation time constant in hours
ETA = 60               # anticipation parameter (km²/h) — Direction A uses 60
KAPPA = 40             # density offset (veh/km/lane)
V_FREE = 102           # free-flow speed (km/h)
RHO_CRIT = 33.5        # critical density (veh/km/lane)
RHO_MAX = 180          # maximum density (veh/km/lane)
A_PARAM = 1.867        # exponent in speed-density function
C_O = 2000             # on-ramp capacity (veh/h)
C_LANE = 4000          # per-lane capacity (veh/h/lane)
W_MAX = 200            # max queue length (vehicles)
BETA_SMOOTH = 50       # smoothing parameter for smooth_min
DELTA_MERGE = 0.0122   # merge speed drop coefficient

# Nominal operating conditions
D_NOMINAL = 500        # nominal on-ramp demand (veh/h)
Q_UPSTREAM = 3000      # upstream mainline inflow (veh/h)
R_NOMINAL = 0.8        # nominal ramp metering rate


def equilibrium_speed(rho):
    """V_e(ρ) = v_free * exp(-(1/a) * (ρ/ρ_crit)^a)"""
    rho = np.maximum(rho, 0.01)
    return V_FREE * np.exp(-(1 / A_PARAM) * (rho / RHO_CRIT) ** A_PARAM)


def smooth_min3(a, b, c, beta=BETA_SMOOTH):
    """Smooth approximation of min(a, b, c) using log-sum-exp."""
    vals = np.array([a, b, c], dtype=float)
    m = np.min(vals)
    return m - (1 / beta) * np.log(
        np.exp(-beta * (a - m)) + np.exp(-beta * (b - m)) + np.exp(-beta * (c - m))
    )


def smooth_min2(a, b, beta=BETA_SMOOTH):
    """Smooth approximation of min(a, b) using log-sum-exp."""
    m = min(a, b)
    return m - (1 / beta) * np.log(
        np.exp(-beta * (a - m)) + np.exp(-beta * (b - m))
    )


def metanet_step(state, r_o, d_o, q_up):
    """
    One METANET time step (Euler forward, discrete-time).

    Args:
        state: [ρ1, ρ2, ρ3, v1, v2, v3, w1]
        r_o: ramp metering rate ∈ [0, 1]
        d_o: on-ramp demand (veh/h)
        q_up: upstream mainline inflow (veh/h)
    Returns:
        next_state: updated state vector
    """
    rho1, rho2, rho3, v1, v2, v3, w1 = state

    # Clamp to avoid numerical issues
    rho1, rho2, rho3 = max(rho1, 0.01), max(rho2, 0.01), max(rho3, 0.01)
    v1, v2, v3 = max(v1, 0.01), max(v2, 0.01), max(v3, 0.01)
    w1 = max(w1, 0.0)

    # Mainline flows
    q1 = rho1 * v1 * LAMBDA
    q2 = rho2 * v2 * LAMBDA
    q3 = rho3 * v3 * LAMBDA

    # On-ramp flow: min(demand+queue_drain, metered_capacity, space_available)
    term1 = d_o + w1 / T
    term2 = C_O * r_o
    term3 = C_O * (RHO_MAX - rho1) / (RHO_MAX - RHO_CRIT)
    q_on1 = smooth_min3(term1, term2, term3)
    q_on1 = max(q_on1, 0.0)

    # Upstream boundary: approximate upstream speed
    v_up = min(q_up / (LAMBDA * max(rho1, 0.01)), V_FREE)

    # Equilibrium speeds
    Ve1, Ve2, Ve3 = equilibrium_speed(rho1), equilibrium_speed(rho2), equilibrium_speed(rho3)

    # --- Density updates ---
    rho1_new = rho1 + (T / (L * LAMBDA)) * (q_up - q1 + q_on1)
    rho2_new = rho2 + (T / (L * LAMBDA)) * (q1 - q2)
    rho3_new = rho3 + (T / (L * LAMBDA)) * (q2 - q3)

    # --- Speed updates ---
    v1_new = (v1
              + (T / TAU) * (Ve1 - v1)
              + (T / L) * v1 * (v_up - v1)
              - (ETA * T / (TAU * L)) * (rho2 - rho1) / (rho1 + KAPPA))
    v1_new -= DELTA_MERGE * T * q_on1 * v1 / (L * LAMBDA * (rho1 + KAPPA))

    v2_new = (v2
              + (T / TAU) * (Ve2 - v2)
              + (T / L) * v2 * (v1 - v2)
              - (ETA * T / (TAU * L)) * (rho3 - rho2) / (rho2 + KAPPA))

    v3_new = (v3
              + (T / TAU) * (Ve3 - v3)
              + (T / L) * v3 * (v2 - v3)
              - (ETA * T / (TAU * L)) * (rho3 - rho3) / (rho3 + KAPPA))

    # --- Queue update ---
    w1_new = w1 + T * (d_o - q_on1)

    return np.array([rho1_new, rho2_new, rho3_new, v1_new, v2_new, v3_new, w1_new])


def find_equilibrium(r_o=R_NOMINAL, d_o=D_NOMINAL, q_up=Q_UPSTREAM,
                     max_iter=500000, tol=1e-8):
    """Find equilibrium by iterating METANET until convergence."""
    rho_init = q_up / (LAMBDA * V_FREE)
    v_init = V_FREE * 0.95
    state = np.array([rho_init, rho_init, rho_init, v_init, v_init, v_init, 0.0])

    for i in range(max_iter):
        state_new = metanet_step(state, r_o, d_o, q_up)
        if np.max(np.abs(state_new - state)) < tol:
            return state_new, i + 1, True
        state = state_new

    return state, max_iter, False


def linearize(x_eq, r_o, d_o, q_up, delta=1e-6):
    """Compute Jacobians A, B_u, B_w via central finite differences."""
    n = len(x_eq)
    A = np.zeros((n, n))
    B_u = np.zeros((n, 1))
    B_w = np.zeros((n, 2))

    for j in range(n):
        dx = max(delta * abs(x_eq[j]), delta)
        x_p, x_m = x_eq.copy(), x_eq.copy()
        x_p[j] += dx
        x_m[j] -= dx
        A[:, j] = (metanet_step(x_p, r_o, d_o, q_up) - metanet_step(x_m, r_o, d_o, q_up)) / (2 * dx)

    dr = delta
    B_u[:, 0] = (metanet_step(x_eq, r_o + dr, d_o, q_up) - metanet_step(x_eq, r_o - dr, d_o, q_up)) / (2 * dr)

    dd = max(delta * d_o, delta)
    B_w[:, 0] = (metanet_step(x_eq, r_o, d_o + dd, q_up) - metanet_step(x_eq, r_o, d_o - dd, q_up)) / (2 * dd)

    dq = delta * q_up
    B_w[:, 1] = (metanet_step(x_eq, r_o, d_o, q_up + dq) - metanet_step(x_eq, r_o, d_o, q_up - dq)) / (2 * dq)

    return A, B_u, B_w


def simulate_tts(state0, n_steps, r_o, d_o_sequence, q_up):
    """
    Simulate METANET for n_steps and compute Total Time Spent (TTS).
    TTS = Σ_k [T * L * λ * (ρ1 + ρ2 + ρ3) + T * w1]   (in veh·hours)
    """
    state = state0.copy()
    tts = 0.0
    states = [state.copy()]

    for k in range(n_steps):
        d_o_k = d_o_sequence[k] if hasattr(d_o_sequence, '__len__') else d_o_sequence
        tts += T * L * LAMBDA * (state[0] + state[1] + state[2]) + T * state[6]
        state = metanet_step(state, r_o, d_o_k, q_up)
        # Clamp queue to physical bounds
        state[6] = np.clip(state[6], 0, W_MAX)
        states.append(state.copy())

    return tts, np.array(states)
