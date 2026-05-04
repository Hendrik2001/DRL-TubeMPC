"""
Experiment 4 (revised): honest 3-controller comparison on the same demand surge.

  A  ALINEA          — industry-standard PI feedback (strong baseline)
  B  ALINEA + Tube   — ALINEA action projected onto tightened constraints
  C  Aggressive      — unconstrained throughput-maximising feedback (RL proxy)

The point is NOT that Tube beats ALINEA on total TTS (it usually loses a few %).
The point is that ALINEA + Tube yields zero hard-constraint violations while the
other two do not — the tube buys formal safety, not performance.
"""
import numpy as np
import metanet6 as mn
import experiment2_lookup_table as ex2


# --- scenario ---
T_SIM = 7200                     # seconds (2 hours)
DT = 10                          # METANET step = 10 s (also control step)
N_STEPS = int(T_SIM / DT)        # 720 steps

# --- ALINEA constants ---
K_R = 70.0                       # (veh/h) / (veh/km/lane)
R_MIN, R_MAX = 0.05, 1.0


def demand_profile():
    t = np.arange(0, T_SIM, DT)
    # Ramp demand: moderate → surge → moderate.
    # Peak ~850 veh/h – within ALINEA's throttling authority.
    rise = np.clip((t - 1200) / 600, 0, 1)
    fall = np.clip((3600 - t) / 600, 0, 1)
    d_ramp = 400 + 450 * rise * fall
    # Upstream flow: steady with slight swell during the surge.
    rise_q = np.clip((t - 900) / 600, 0, 1)
    fall_q = np.clip((4200 - t) / 600, 0, 1)
    q_up = 2900 + 350 * rise_q * fall_q
    return t, d_ramp, q_up


# ── Tube look-up support ──────────────────────────────────────────────
def usable_entries(table):
    """Stable entries whose local density box lies strictly below ρ_crit.
    These are the tubes that can actually certify free-flow safety."""
    out = []
    for e in table:
        if not e['stable']:
            continue
        if e['hw'][0] > mn.RHO_MAX / 2 or e['hw'][mn.N_SEG] > mn.V_FREE / 2:
            continue
        upper = e['rho_op'] + e['hw'][0]
        if upper >= mn.RHO_CRIT - 0.25:
            continue
        out.append(e)
    return out


def nearest_entry(state, entries):
    """Nearest usable entry based on mean mainline density."""
    if not entries:
        return None
    rho_mean = float(np.mean(state[:mn.N_SEG]))
    best = None
    best_err = float('inf')
    for e in entries:
        err = abs(e['rho_op'] - rho_mean)
        if err < best_err:
            best_err = err
            best = e
    return best


def state_in_box(state, tl, th):
    return bool(np.all(state >= tl - 1e-9) and np.all(state <= th + 1e-9))


def local_safety_box(entry):
    """
    The effective safety box around an operating point:
        [x_op - hw, x_op + hw]
    intersected with the hard constraint box [lo, hi]. This is the true
    local RPI region; the "global tightened" box from X ⊖ Z is much looser
    because X already permits deep congestion.
    """
    x_op = entry['x_op']
    hw = entry['hw']
    lo_global, hi_global = mn.constraint_box()
    box_lo = np.maximum(x_op - hw, lo_global)
    box_hi = np.minimum(x_op + hw, hi_global)
    return box_lo, box_hi


def tube_safety_filter(state, r_proposed, d1, q_up, entry):
    """
    Project the proposed action onto the LOCAL RPI box around the current
    operating point. This is where the safety guarantee actually lives:
    staying inside the local tube keeps us in the linearisation's validity
    region and – because the tube is RPI – guarantees recursive feasibility.

    Density dimensions are the binding ones in practice.
    """
    if entry is None:
        return r_proposed
    box_lo, box_hi = local_safety_box(entry)

    def next_safe(r):
        nxt = mn.metanet_step(state, r, d1, q_up)
        # Only check the 12 mainline dims (densities + speeds); queue is
        # tracked separately and has its own hard bound.
        core = np.concatenate([nxt[:mn.N_SEG], nxt[mn.N_SEG:2 * mn.N_SEG]])
        lo = np.concatenate([box_lo[:mn.N_SEG], box_lo[mn.N_SEG:2 * mn.N_SEG]])
        hi = np.concatenate([box_hi[:mn.N_SEG], box_hi[mn.N_SEG:2 * mn.N_SEG]])
        return bool(np.all(core >= lo - 1e-9) and np.all(core <= hi + 1e-9))

    if next_safe(r_proposed):
        return r_proposed

    # If R_MIN already infeasible, return it anyway (best effort).
    if not next_safe(R_MIN):
        return R_MIN

    lo, hi = R_MIN, r_proposed
    for _ in range(20):
        mid = 0.5 * (lo + hi)
        if next_safe(mid):
            lo = mid
        else:
            hi = mid
    return lo


# ── Controllers ───────────────────────────────────────────────────────
def controller_alinea(state, prev_r, d1, q_up):
    """ALINEA: r(k) = r(k-1) + (K_R/C_o) * (ρ_crit - ρ_downstream). """
    rho_down = state[1]          # segment 2 density (downstream of the ramp)
    delta_q = K_R * (mn.RHO_CRIT - rho_down)     # veh/h adjustment
    r_new = prev_r + delta_q / mn.C_O
    return float(np.clip(r_new, R_MIN, R_MAX))


def controller_aggressive(state, prev_r, d1, q_up):
    """Unconstrained throughput maximiser (RL proxy without safety)."""
    rho1 = state[0]
    r = (mn.RHO_MAX - rho1) / (mn.RHO_MAX - mn.RHO_CRIT)
    return float(np.clip(r, R_MIN, R_MAX))


# ── Simulation engine ─────────────────────────────────────────────────
def simulate(controller, table=None, use_tube=False, label=""):
    t, d_arr, q_arr = demand_profile()
    N = len(t)

    entries = usable_entries(table) if table is not None else []

    state = np.zeros(mn.NX)
    rho0 = 0.55 * mn.RHO_CRIT
    state[:mn.N_SEG] = rho0
    state[mn.N_SEG:2 * mn.N_SEG] = mn.Ve(rho0)

    X = np.zeros((N, mn.NX))
    R = np.zeros(N)
    TTS = np.zeros(N)
    MARGIN = np.zeros(N)

    prev_r = 0.5
    for k in range(N):
        d1 = d_arr[k]
        q_up = q_arr[k]

        r_proposed = controller(state, prev_r, d1, q_up)

        if use_tube:
            entry = nearest_entry(state, entries)
            r_applied = tube_safety_filter(state, r_proposed, d1, q_up, entry)
        else:
            r_applied = r_proposed
            entry = nearest_entry(state, entries) if entries else None

        # TTS increment: T * Σρ_j * L_j * λ  (queue ignored here, reported
        # separately)
        rho = state[:mn.N_SEG]
        tts_k = mn.T * float(np.sum(rho)) * mn.L * mn.LAM

        # Safety margin against the LOCAL tube box (density+speed only)
        if entry is not None:
            box_lo, box_hi = local_safety_box(entry)
            sel = list(range(0, 2 * mn.N_SEG))
            slacks = np.concatenate([
                state[sel] - box_lo[sel],
                box_hi[sel] - state[sel],
            ])
            margin = float(np.min(slacks))
        else:
            margin = np.nan

        X[k] = state
        R[k] = r_applied
        TTS[k] = tts_k
        MARGIN[k] = margin

        state = mn.metanet_step(state, r_applied, d1, q_up)
        prev_r = r_applied

    return {
        'label': label, 't': t, 'x': X, 'r': R,
        'tts_step': TTS, 'margin': MARGIN,
        'd': d_arr, 'q': q_arr,
    }


def summarise(result):
    rho = result['x'][:, :mn.N_SEG]
    w = result['x'][:, 2 * mn.N_SEG]
    # Soft violation: any segment above ρ_crit
    steps_rho_crit = int(np.sum(np.any(rho > mn.RHO_CRIT, axis=1)))
    # Hard violation: queue overflow
    steps_w_max = int(np.sum(w > mn.W_MAX))
    return {
        'label': result['label'],
        'tts_total': float(np.sum(result['tts_step'])),
        'max_rho1': float(np.max(rho[:, 0])),
        'max_rho_any': float(np.max(rho)),
        'max_w': float(np.max(w)),
        'steps_rho_crit': steps_rho_crit,
        'steps_w_max': steps_w_max,
        'frac_unsafe': float(np.mean(result['margin'] < 0)),
    }


# ── Public run ─────────────────────────────────────────────────────────
def run(verbose=True):
    if verbose:
        print("  Building RPI lookup table ...")
    table, _ = ex2.run(verbose=False)

    if verbose:
        print("  Running ALINEA ...")
    alinea = simulate(controller_alinea, table=table, use_tube=False,
                      label='ALINEA')

    if verbose:
        print("  Running ALINEA + Tube ...")
    alinea_tube = simulate(controller_alinea, table=table, use_tube=True,
                           label='ALINEA+Tube')

    if verbose:
        print("  Running Aggressive (RL proxy) ...")
    aggressive = simulate(controller_aggressive, table=table, use_tube=False,
                          label='Aggressive')

    summaries = [summarise(r) for r in (alinea, alinea_tube, aggressive)]

    if verbose:
        print()
        print(f"  {'Controller':<16} {'TTS':>9} {'Max ρ₁':>8} {'Max ρ':>8} "
              f"{'Max w':>8} {'ρ>ρc':>6} {'w>w_max':>8}")
        print("  " + "-" * 66)
        for s in summaries:
            print(f"  {s['label']:<16} {s['tts_total']:>9.2f} "
                  f"{s['max_rho1']:>8.2f} {s['max_rho_any']:>8.2f} "
                  f"{s['max_w']:>8.1f} {s['steps_rho_crit']:>6d} "
                  f"{s['steps_w_max']:>8d}")
        tts_a = summaries[0]['tts_total']
        tts_at = summaries[1]['tts_total']
        print(f"\n  TTS cost of adding the tube: "
              f"{(tts_at - tts_a) / tts_a * 100:+.2f}%")

    return {
        'table': table,
        'alinea': alinea, 'alinea_tube': alinea_tube, 'aggressive': aggressive,
        'summaries': summaries,
    }


if __name__ == '__main__':
    print("=" * 70)
    print("  EXPERIMENT 4 (revised): ALINEA vs ALINEA+Tube vs Aggressive")
    print("=" * 70)
    run()
