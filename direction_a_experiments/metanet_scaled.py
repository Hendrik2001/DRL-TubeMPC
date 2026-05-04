"""
Scalable METANET model: N segments, M on-ramps.

State vector: [ρ₁,...,ρ_N, v₁,...,v_N, w₁,...,w_M]  (dim = 2N + M)
On-ramps are placed at specified segment indices.

Same Kotsialos et al. (1999) parameters as the 3-segment model.
"""
import numpy as np

# --- Parameters ---
T = 10 / 3600          # sampling time in hours (10 seconds)
L = 1.0                # segment length in km
LAMBDA = 2             # lanes
TAU = 18 / 3600        # relaxation time in hours
ETA = 60               # anticipation (km²/h)
KAPPA = 40             # density offset (veh/km/lane)
V_FREE = 102           # free-flow speed (km/h)
RHO_CRIT = 33.5        # critical density (veh/km/lane)
RHO_MAX = 180          # max density (veh/km/lane)
A_PARAM = 1.867        # exponent
C_O = 2000             # on-ramp capacity (veh/h)
W_MAX = 200            # max queue (vehicles)
BETA_SMOOTH = 50       # smooth-min sharpness
DELTA_MERGE = 0.0122   # merge speed drop


def Ve(rho):
    """Equilibrium speed."""
    rho = np.maximum(rho, 0.01)
    return V_FREE * np.exp(-(1 / A_PARAM) * (rho / RHO_CRIT) ** A_PARAM)


def smooth_min3(a, b, c, beta=BETA_SMOOTH):
    vals = np.array([a, b, c], dtype=float)
    m = np.min(vals)
    return m - (1 / beta) * np.log(
        np.exp(-beta * (a - m)) + np.exp(-beta * (b - m)) + np.exp(-beta * (c - m))
    )


class ScaledMETANET:
    """
    N-segment METANET with on-ramps at specified segments.

    Args:
        n_seg: number of segments
        onramp_segs: list of segment indices (0-based) with on-ramps
    """

    def __init__(self, n_seg=10, onramp_segs=None):
        self.N = n_seg
        if onramp_segs is None:
            # Default: on-ramps at roughly equal spacing
            if n_seg <= 5:
                onramp_segs = [0]
            elif n_seg <= 10:
                onramp_segs = [0, n_seg // 3, 2 * n_seg // 3]
            else:
                onramp_segs = [0, n_seg // 4, n_seg // 2, 3 * n_seg // 4]
        self.onramp_segs = sorted(onramp_segs)
        self.M = len(onramp_segs)
        self.nx = 2 * n_seg + self.M  # state dimension

        # Map: segment index → queue index (or -1)
        self.seg_to_queue = {}
        for qi, si in enumerate(self.onramp_segs):
            self.seg_to_queue[si] = qi

    def unpack_state(self, state):
        """Split state into densities, speeds, queues."""
        rho = state[:self.N].copy()
        v = state[self.N:2 * self.N].copy()
        w = state[2 * self.N:].copy()
        return rho, v, w

    def pack_state(self, rho, v, w):
        return np.concatenate([rho, v, w])

    def step(self, state, r_vec, d_vec, q_up):
        """
        One METANET time step.

        Args:
            state: [ρ₁..ρ_N, v₁..v_N, w₁..w_M]
            r_vec: metering rates for each on-ramp (len M)
            d_vec: demand for each on-ramp (len M, veh/h)
            q_up: upstream mainline inflow (veh/h)
        """
        rho, v, w = self.unpack_state(state)
        N = self.N

        # Clamp
        rho = np.clip(rho, 0.01, RHO_MAX)
        v = np.clip(v, 0.01, V_FREE * 1.5)
        w = np.clip(w, 0.0, W_MAX)

        # Flows
        q = rho * v * LAMBDA  # mainline flow per segment

        # On-ramp flows
        q_on = np.zeros(N)
        for qi, si in enumerate(self.onramp_segs):
            t1 = d_vec[qi] + w[qi] / T
            t2 = C_O * r_vec[qi]
            t3 = C_O * (RHO_MAX - rho[si]) / (RHO_MAX - RHO_CRIT)
            q_on[si] = max(smooth_min3(t1, t2, t3), 0.0)

        # Upstream speed
        v_up = min(q_up / (LAMBDA * max(rho[0], 0.01)), V_FREE)

        # Equilibrium speeds
        Ve_all = Ve(rho)

        # New densities
        rho_new = np.zeros(N)
        for i in range(N):
            q_in = q_up if i == 0 else q[i - 1]
            q_out = q[i]
            rho_new[i] = rho[i] + (T / (L * LAMBDA)) * (q_in - q_out + q_on[i])

        # New speeds
        v_new = np.zeros(N)
        for i in range(N):
            # Relaxation
            dv = (T / TAU) * (Ve_all[i] - v[i])

            # Convection
            v_prev = v_up if i == 0 else v[i - 1]
            dv += (T / L) * v[i] * (v_prev - v[i])

            # Anticipation (look at downstream segment)
            if i < N - 1:
                dv -= (ETA * T / (TAU * L)) * (rho[i + 1] - rho[i]) / (rho[i] + KAPPA)

            v_new[i] = v[i] + dv

            # Merge speed drop at on-ramp segments
            if i in self.seg_to_queue:
                v_new[i] -= DELTA_MERGE * T * q_on[i] * v[i] / (L * LAMBDA * (rho[i] + KAPPA))

        # New queues
        w_new = np.zeros(self.M)
        for qi, si in enumerate(self.onramp_segs):
            w_new[qi] = w[qi] + T * (d_vec[qi] - q_on[si])

        # Clamp
        rho_new = np.clip(rho_new, 0.01, RHO_MAX)
        v_new = np.clip(v_new, 0.01, V_FREE * 1.5)
        w_new = np.clip(w_new, 0.0, W_MAX)

        return self.pack_state(rho_new, v_new, w_new)

    def compute_tts_step(self, state):
        """TTS contribution for one time step: T·L·λ·Σρ + T·Σw."""
        rho, v, w = self.unpack_state(state)
        return T * L * LAMBDA * np.sum(rho) + T * np.sum(w)

    def make_homogeneous_state(self, rho_val, w_val=0.0):
        """Create a state with uniform density and equilibrium speed."""
        rho = np.full(self.N, rho_val)
        v = np.full(self.N, Ve(rho_val))
        w = np.full(self.M, w_val)
        return self.pack_state(rho, v, w)
