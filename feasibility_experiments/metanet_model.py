"""
METANET freeway traffic model: 3 segments, 1 on-ramp.
State vector: [ρ1, ρ2, ρ3, v1, v2, v3, w1] (dim 7)
Control input: r_o (ramp metering rate)
Disturbances: w_d (demand), w_q (upstream flow)
"""
import numpy as np

# --- Parameters (Kotsialos et al. 1999) ---
T = 10 / 3600          # sampling time in hours (10 seconds)
L = 1.0                # segment length in km
LAMBDA = 2             # number of lanes
TAU = 18 / 3600        # relaxation time constant in hours
ETA = 65               # anticipation parameter (km²/h)
KAPPA = 40             # density offset (veh/km/lane)
V_FREE = 102           # free-flow speed (km/h)
RHO_CRIT = 33.5        # critical density (veh/km/lane)
RHO_MAX = 180          # maximum density (veh/km/lane)
A_PARAM = 1.867        # exponent in speed-density function
C_O = 2000             # on-ramp capacity (veh/h)
W_MAX = 200            # max queue length (vehicles)
BETA_SMOOTH = 50       # smoothing parameter for smooth_min

# Nominal operating conditions
D_NOMINAL = 500        # nominal on-ramp demand (veh/h)
Q_UPSTREAM = 3000      # upstream mainline inflow (veh/h)
R_NOMINAL = 0.8        # nominal ramp metering rate


def equilibrium_speed(rho):
    """V(ρ) = v_free * exp(-(1/a) * (ρ/ρ_crit)^a)"""
    return V_FREE * np.exp(-(1 / A_PARAM) * (rho / RHO_CRIT) ** A_PARAM)


def smooth_min3(a, b, c, beta=BETA_SMOOTH):
    """Smooth approximation of min(a, b, c) using log-sum-exp."""
    # Numerically stable version
    vals = np.array([a, b, c])
    m = np.min(vals)
    return m - (1 / beta) * np.log(
        np.exp(-beta * (a - m)) + np.exp(-beta * (b - m)) + np.exp(-beta * (c - m))
    )


def metanet_step(state, r_o, d_o, q_up):
    """
    One METANET time step.

    Args:
        state: [ρ1, ρ2, ρ3, v1, v2, v3, w1]
        r_o: ramp metering rate ∈ [0, 1]
        d_o: on-ramp demand (veh/h)
        q_up: upstream mainline inflow (veh/h)

    Returns:
        next_state: updated state vector
    """
    rho1, rho2, rho3, v1, v2, v3, w1 = state

    # Clamp densities and speeds to avoid numerical issues
    rho1 = max(rho1, 0.01)
    rho2 = max(rho2, 0.01)
    rho3 = max(rho3, 0.01)
    v1 = max(v1, 0.01)
    v2 = max(v2, 0.01)
    v3 = max(v3, 0.01)
    w1 = max(w1, 0.0)

    # Mainline flows: q_j = ρ_j * v_j * λ
    q1 = rho1 * v1 * LAMBDA
    q2 = rho2 * v2 * LAMBDA
    q3 = rho3 * v3 * LAMBDA

    # On-ramp flow (smooth min of three terms)
    term1 = d_o + w1 / T    # demand + queue drain
    term2 = C_O * r_o       # metered capacity
    term3 = C_O * (RHO_MAX - rho1) / (RHO_MAX - RHO_CRIT)  # space available
    q_on1 = smooth_min3(term1, term2, term3)
    q_on1 = max(q_on1, 0.0)

    # Upstream flow enters segment 1; use upstream speed from flow/density
    # For boundary: v_0 = q_up / (ρ_0 * λ), approximate ρ_0 from q_up and V(ρ_0)
    # Simple approach: use v1 as proxy for upstream speed (common in METANET)
    v_up = q_up / (LAMBDA * max(rho1, 0.01))  # approximate upstream speed
    v_up = min(v_up, V_FREE)  # cap at free-flow

    # For downstream boundary of segment 3: use ρ_4 ≈ ρ_3 (free outflow)
    rho_downstream = rho3

    # Equilibrium speeds
    Ve1 = equilibrium_speed(rho1)
    Ve2 = equilibrium_speed(rho2)
    Ve3 = equilibrium_speed(rho3)

    # --- Density updates ---
    # Segment 1: inflow = q_up, outflow = q1, on-ramp = q_on1
    rho1_new = rho1 + (T / (L * LAMBDA)) * (q_up - q1 + q_on1)

    # Segment 2: inflow = q1, outflow = q2, no ramp
    rho2_new = rho2 + (T / (L * LAMBDA)) * (q1 - q2)

    # Segment 3: inflow = q2, outflow = q3, no ramp
    rho3_new = rho3 + (T / (L * LAMBDA)) * (q2 - q3)

    # --- Speed updates ---
    # Segment 1
    v1_new = (v1
              + (T / TAU) * (Ve1 - v1)
              + (T / L) * v1 * (v_up - v1)
              - (ETA * T / (TAU * L)) * (rho2 - rho1) / (rho1 + KAPPA))

    # Segment 2
    v2_new = (v2
              + (T / TAU) * (Ve2 - v2)
              + (T / L) * v2 * (v1 - v2)
              - (ETA * T / (TAU * L)) * (rho3 - rho2) / (rho2 + KAPPA))

    # Segment 3 (downstream boundary: use ρ_downstream ≈ ρ_3, so anticipation = 0)
    v3_new = (v3
              + (T / TAU) * (Ve3 - v3)
              + (T / L) * v3 * (v2 - v3)
              - (ETA * T / (TAU * L)) * (rho_downstream - rho3) / (rho3 + KAPPA))

    # On-ramp merging effect on speed of segment 1
    # Standard METANET: speed drop due to merging = -(δ*T*q_on1*v1) / (L*λ*(ρ1+κ))
    # Use δ=0.0122 (typical value)
    delta_merge = 0.0122
    v1_new -= delta_merge * T * q_on1 * v1 / (L * LAMBDA * (rho1 + KAPPA))

    # --- Queue update ---
    w1_new = w1 + T * (d_o - q_on1)

    return np.array([rho1_new, rho2_new, rho3_new, v1_new, v2_new, v3_new, w1_new])


def find_equilibrium(r_o=R_NOMINAL, d_o=D_NOMINAL, q_up=Q_UPSTREAM,
                     max_iter=500000, tol=1e-8):
    """Find equilibrium by iterating METANET until convergence."""
    # Initial guess: free-flow conditions
    rho_init = q_up / (LAMBDA * V_FREE)  # from q = ρ*v*λ
    v_init = V_FREE * 0.95
    w_init = 0.0

    state = np.array([rho_init, rho_init, rho_init, v_init, v_init, v_init, w_init])

    for i in range(max_iter):
        state_new = metanet_step(state, r_o, d_o, q_up)
        if np.max(np.abs(state_new - state)) < tol:
            return state_new, i + 1
        state = state_new

    print(f"Warning: equilibrium not converged after {max_iter} iterations")
    print(f"Max residual: {np.max(np.abs(metanet_step(state, r_o, d_o, q_up) - state)):.2e}")
    return state, max_iter


def linearize(x_eq, r_o, d_o, q_up, delta=1e-6):
    """
    Compute Jacobians A (∂f/∂x), B_u (∂f/∂r), B_w (∂f/∂[d_o, q_up])
    via finite differences at equilibrium.
    """
    n = len(x_eq)
    A = np.zeros((n, n))
    B_u = np.zeros((n, 1))
    B_w = np.zeros((n, 2))

    f0 = metanet_step(x_eq, r_o, d_o, q_up)

    # A = ∂f/∂x
    for j in range(n):
        dx = max(delta * abs(x_eq[j]), delta)
        x_pert = x_eq.copy()
        x_pert[j] += dx
        fp = metanet_step(x_pert, r_o, d_o, q_up)
        x_pert[j] = x_eq[j] - dx
        fm = metanet_step(x_pert, r_o, d_o, q_up)
        A[:, j] = (fp - fm) / (2 * dx)

    # B_u = ∂f/∂r_o
    dr = delta
    fp = metanet_step(x_eq, r_o + dr, d_o, q_up)
    fm = metanet_step(x_eq, r_o - dr, d_o, q_up)
    B_u[:, 0] = (fp - fm) / (2 * dr)

    # B_w = ∂f/∂[d_o, q_up]
    dd = delta * d_o if d_o > 0 else delta
    fp = metanet_step(x_eq, r_o, d_o + dd, q_up)
    fm = metanet_step(x_eq, r_o, d_o - dd, q_up)
    B_w[:, 0] = (fp - fm) / (2 * dd)

    dq = delta * q_up
    fp = metanet_step(x_eq, r_o, d_o, q_up + dq)
    fm = metanet_step(x_eq, r_o, d_o, q_up - dq)
    B_w[:, 1] = (fp - fm) / (2 * dq)

    return A, B_u, B_w
