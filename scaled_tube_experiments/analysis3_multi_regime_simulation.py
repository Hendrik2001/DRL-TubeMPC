"""
Analysis 3: Multi-regime 3-hour simulation with tube switching.

Demand profile traverses five phases:
  1  free-flow   (0–45 min)
  2  building    (45–90 min)
  3  near-crit   (90–120 min)
  4  recovery    (120–150 min)
  5  return      (150–180 min)

ALINEA + tube filter, tube looked up from lookup table at each control step.
Records: state trajectory, tube index, switch events, safety margins.
"""
import numpy as np
import metanet6 as mn
import tube_tools as tt
import experiment2_lookup_table as ex2

T_SIM = 10800      # 3 hours
DT_CTRL = 60       # control every 60 s
DT_SIM = 10        # metanet step
INNER = DT_CTRL // DT_SIM

K_R = 70.0
R_MIN, R_MAX = 0.05, 1.0
RNG = np.random.RandomState(99)

Delta = np.array([mn.DELTA_D, mn.DELTA_Q])


def demand_profile(t_sec):
    """Five-phase demand + 10% noise."""
    # Ramp demand
    if t_sec < 2700:
        d = 400.0
    elif t_sec < 5400:
        d = 400 + 500 * (t_sec - 2700) / 2700
    elif t_sec < 7200:
        d = 900.0
    elif t_sec < 9000:
        d = 900 - 500 * (t_sec - 7200) / 1800
    else:
        d = 400.0
    # Upstream
    if t_sec < 2700:
        q = 2800.0
    elif t_sec < 5400:
        q = 2800 + 450 * (t_sec - 2700) / 2700
    elif t_sec < 7200:
        q = 3250.0
    elif t_sec < 9000:
        q = 3250 - 450 * (t_sec - 7200) / 1800
    else:
        q = 2800.0
    d *= (1 + 0.10 * RNG.uniform(-1, 1))
    q *= (1 + 0.10 * RNG.uniform(-1, 1))
    return max(d, 50.0), max(q, 500.0)


# ── classify + safety filter (same logic as experiment 4) ─────────────
def usable_entries(table):
    out = []
    for e in table:
        if not e['stable']:
            continue
        if e['hw'][0] > mn.RHO_MAX / 2 or e['hw'][mn.N_SEG] > mn.V_FREE / 2:
            continue
        if e['rho_op'] + e['hw'][0] >= mn.RHO_CRIT - 0.25:
            continue
        out.append(e)
    return out


def nearest(state, entries):
    if not entries:
        return None
    rho_mean = float(np.mean(state[:mn.N_SEG]))
    return min(entries, key=lambda e: abs(e['rho_op'] - rho_mean))


def local_box(entry):
    x = entry['x_op']
    hw = entry['hw']
    lo, hi = mn.constraint_box()
    return np.maximum(x - hw, lo), np.minimum(x + hw, hi)


def safety_filter(state, r_prop, d1, q_up, entry):
    if entry is None:
        return r_prop
    blo, bhi = local_box(entry)
    sel = list(range(2 * mn.N_SEG))

    def ok(r):
        nxt = mn.metanet_step(state, r, d1, q_up)
        return bool(np.all(nxt[sel] >= blo[sel] - 1e-9) and
                    np.all(nxt[sel] <= bhi[sel] + 1e-9))

    if ok(r_prop):
        return r_prop
    if not ok(R_MIN):
        return R_MIN
    lo, hi = R_MIN, r_prop
    for _ in range(20):
        mid = 0.5 * (lo + hi)
        if ok(mid):
            lo = mid
        else:
            hi = mid
    return lo


def alinea(state, prev_r):
    delta_q = K_R * (mn.RHO_CRIT - state[1])
    return float(np.clip(prev_r + delta_q / mn.C_O, R_MIN, R_MAX))


# ── simulation ────────────────────────────────────────────────────────
def run(verbose=True):
    table, _ = ex2.run(verbose=False)
    entries = usable_entries(table)

    if verbose:
        print(f"  Usable (strictly sub-ρ_crit) entries: {len(entries)}")
        for e in entries:
            ub = e['rho_op'] + e['hw'][0]
            print(f"    factor={e['factor']:.2f}, ρ_op={e['rho_op']:.1f}, "
                  f"upper={ub:.1f}")

    n_ctrl = T_SIM // DT_CTRL

    state = np.zeros(mn.NX)
    rho0 = 0.5 * mn.RHO_CRIT
    state[:mn.N_SEG] = rho0
    state[mn.N_SEG:2 * mn.N_SEG] = mn.Ve(rho0)

    T_arr = np.zeros(n_ctrl)
    X_arr = np.zeros((n_ctrl, mn.NX))
    R_arr = np.zeros(n_ctrl)
    IDX_arr = np.full(n_ctrl, -1, dtype=int)
    MARGIN_arr = np.zeros(n_ctrl)
    D_arr = np.zeros(n_ctrl)
    Q_arr = np.zeros(n_ctrl)

    switches = []    # list of (step, old_idx, new_idx, state_in_new)
    prev_r = 0.5
    prev_idx = -1

    for k in range(n_ctrl):
        t_sec = k * DT_CTRL

        entry = nearest(state, entries)
        idx = entry['factor'] if entry else -1

        # Detect tube switch
        if idx != prev_idx and prev_idx != -1:
            blo, bhi = local_box(entry) if entry else (None, None)
            sel = list(range(2 * mn.N_SEG))
            inside = (entry is not None and
                      bool(np.all(state[sel] >= blo[sel] - 1e-9) and
                           np.all(state[sel] <= bhi[sel] + 1e-9)))
            switches.append((k, prev_idx, idx, inside))

        d1, q_up = demand_profile(t_sec)
        r_prop = alinea(state, prev_r)
        r_app = safety_filter(state, r_prop, d1, q_up, entry)

        # Margin
        if entry is not None:
            blo, bhi = local_box(entry)
            sel = list(range(2 * mn.N_SEG))
            slacks = np.concatenate([state[sel] - blo[sel],
                                     bhi[sel] - state[sel]])
            margin = float(np.min(slacks))
        else:
            margin = np.nan

        T_arr[k] = t_sec + DT_CTRL
        X_arr[k] = state
        R_arr[k] = r_app
        IDX_arr[k] = idx if entry else -1
        MARGIN_arr[k] = margin
        D_arr[k] = d1
        Q_arr[k] = q_up

        for _ in range(INNER):
            state = mn.metanet_step(state, r_app, d1, q_up)
        prev_r = r_app
        prev_idx = idx

    rho_all = X_arr[:, :mn.N_SEG]
    w_all = X_arr[:, 2 * mn.N_SEG]
    max_rho = float(np.max(rho_all))
    max_w = float(np.max(w_all))
    steps_over = int(np.sum(np.any(rho_all > mn.RHO_CRIT, axis=1)))
    tts = float(np.sum(mn.T * mn.L * mn.LAM * np.sum(rho_all, axis=1)))
    n_switches = len(switches)
    n_safe_sw = sum(1 for s in switches if s[3])
    n_unsafe_sw = n_switches - n_safe_sw

    if verbose:
        print(f"\n  3-hour simulation results:")
        print(f"    max density   = {max_rho:.2f} (ρ_crit = {mn.RHO_CRIT})")
        print(f"    max queue     = {max_w:.1f} (w_max = {mn.W_MAX})")
        print(f"    steps ρ>ρ_c   = {steps_over}/{n_ctrl}")
        print(f"    total TTS     = {tts:.2f} veh·h")
        print(f"    tube switches = {n_switches}")
        print(f"    safe switches = {n_safe_sw}/{n_switches}")
        if n_unsafe_sw:
            print(f"    ⚠ UNSAFE switches = {n_unsafe_sw}")
        else:
            print(f"    ✓ all switches safe")

    return {
        't': T_arr, 'x': X_arr, 'r': R_arr, 'idx': IDX_arr,
        'margin': MARGIN_arr, 'd': D_arr, 'q': Q_arr,
        'switches': switches,
        'stats': {
            'max_rho': max_rho, 'max_w': max_w,
            'steps_over': steps_over, 'tts': tts,
            'n_switches': n_switches, 'n_safe': n_safe_sw,
            'n_unsafe': n_unsafe_sw,
        },
        'entries': entries, 'table': table,
    }


if __name__ == '__main__':
    print("=" * 78)
    print("  ANALYSIS 3: Multi-Regime 3-hour Simulation")
    print("=" * 78)
    run()
