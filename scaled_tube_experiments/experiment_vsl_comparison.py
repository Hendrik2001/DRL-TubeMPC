"""
3-controller comparison WITH VSL on the same demand surge scenario.

  A  ALINEA + VSL-feedback
  B  ALINEA + VSL-feedback + Tube safety filter
  C  Aggressive (r=1, no VSL restriction)
"""
import numpy as np
import metanet6 as mn
import metanet6_vsl as vsl
import tube_tools as tt
import experiment_vsl_lookup as vsl_lut

T_SIM = 7200
DT = 10
N_STEPS = T_SIM // DT

K_R = 70.0
K_VSL = 6.0          # (km/h) / (veh/km/lane) — gentle reduction
R_MIN, R_MAX = 0.05, 1.0


def demand_profile():
    t = np.arange(0, T_SIM, DT)
    rise = np.clip((t - 1200) / 600, 0, 1)
    fall = np.clip((3600 - t) / 600, 0, 1)
    d_ramp = 400 + 450 * rise * fall
    rise_q = np.clip((t - 900) / 600, 0, 1)
    fall_q = np.clip((4200 - t) / 600, 0, 1)
    q_up = 2900 + 350 * rise_q * fall_q
    return t, d_ramp, q_up


# ── tube helpers (same logic as experiment4, now using VSL table) ──────
def usable_entries(table):
    """Stable entries whose local density box lies strictly below ρ_crit."""
    out = []
    for e in table:
        if not e['stable'] or e['hw'] is None:
            continue
        if e['hw'][0] > mn.RHO_MAX / 2 or e['hw'][mn.N_SEG] > mn.V_FREE / 2:
            continue
        upper = e['rho_op'] + e['hw'][0]
        if upper >= mn.RHO_CRIT - 0.25:
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


def tube_filter_vsl(state, u_prop, d1, q_up, entry):
    """Project ramp rate so next mainline state stays inside local tube.

    Matches experiment4 approach: bisect on ramp rate, check density+speed
    dimensions only (queue tracked separately).  VSL inputs pass through
    unchanged — they already help via the ALINEA+VSL feedback law.
    """
    if entry is None:
        return u_prop
    blo, bhi = local_box(entry)

    def next_safe(u):
        nxt = vsl.step(state, u, d1, q_up)
        core = nxt[:2 * mn.N_SEG]
        lo = blo[:2 * mn.N_SEG]
        hi = bhi[:2 * mn.N_SEG]
        return bool(np.all(core >= lo - 1e-9) and np.all(core <= hi + 1e-9))

    if next_safe(u_prop):
        return u_prop

    # Bisect on ramp rate only, keeping VSL unchanged
    u = u_prop.copy()
    u_min = u_prop.copy()
    u_min[0] = R_MIN
    if not next_safe(u_min):
        return u_min  # best effort

    lo_r, hi_r = R_MIN, u_prop[0]
    for _ in range(20):
        mid = 0.5 * (lo_r + hi_r)
        u[0] = mid
        if next_safe(u):
            lo_r = mid
        else:
            hi_r = mid
    u[0] = lo_r
    return u


# ── controllers ───────────────────────────────────────────────────────
def ctrl_alinea_vsl(state, prev_u, d1, q_up):
    """ALINEA ramp + simple VSL feedback."""
    rho_down = state[1]
    delta_q = K_R * (mn.RHO_CRIT - rho_down)
    r_new = float(np.clip(prev_u[0] + delta_q / mn.C_O, R_MIN, R_MAX))
    # VSL: gently reduce speed when seg 3/4 density exceeds ρ_crit
    rho3 = state[2]
    rho4 = state[3]
    v3 = max(vsl.VSL_MIN, mn.V_FREE - K_VSL * max(0, rho3 - mn.RHO_CRIT))
    v4 = max(vsl.VSL_MIN, mn.V_FREE - K_VSL * max(0, rho4 - mn.RHO_CRIT))
    return np.array([r_new, v3, v4])


def ctrl_aggressive(state, prev_u, d1, q_up):
    return np.array([1.0, mn.V_FREE, mn.V_FREE])


# ── simulation ────────────────────────────────────────────────────────
def simulate(controller, entries=None, use_tube=False, label=""):
    t, d_arr, q_arr = demand_profile()
    N = len(t)

    state = np.zeros(mn.NX)
    rho0 = 0.55 * mn.RHO_CRIT
    state[:mn.N_SEG] = rho0
    state[mn.N_SEG:2 * mn.N_SEG] = mn.Ve(rho0)

    X = np.zeros((N, mn.NX))
    U = np.zeros((N, vsl.NU))
    TTS = np.zeros(N)

    prev_u = np.array([0.5, mn.V_FREE, mn.V_FREE])
    for k in range(N):
        d1, q_up = d_arr[k], q_arr[k]
        u_prop = controller(state, prev_u, d1, q_up)
        if use_tube and entries:
            entry = nearest(state, entries)
            u_app = tube_filter_vsl(state, u_prop, d1, q_up, entry)
        else:
            u_app = u_prop

        rho = state[:mn.N_SEG]
        TTS[k] = mn.T * mn.L * mn.LAM * float(np.sum(rho))

        X[k] = state
        U[k] = u_app
        state = vsl.step(state, u_app, d1, q_up)
        prev_u = u_app

    return {'label': label, 't': t, 'x': X, 'u': U, 'tts': TTS,
            'd': d_arr, 'q': q_arr}


def summarise(r):
    rho = r['x'][:, :mn.N_SEG]
    w = r['x'][:, 2 * mn.N_SEG]
    return {
        'label': r['label'],
        'tts_total': float(np.sum(r['tts'])),
        'max_rho': float(np.max(rho)),
        'max_w': float(np.max(w)),
        'steps_rho_crit': int(np.sum(np.any(rho > mn.RHO_CRIT, axis=1))),
        'steps_w_max': int(np.sum(w > mn.W_MAX)),
    }


def run(verbose=True):
    if verbose:
        print("  Building RM+VSL lookup table ...")
    _, table_vsl = vsl_lut.run(verbose=False)
    entries = usable_entries(table_vsl)
    if verbose:
        print(f"  {len(entries)} usable tube entries")

    results = {}
    for name, ctrl, tube in [
        ('ALINEA+VSL',      ctrl_alinea_vsl, False),
        ('ALINEA+VSL+Tube', ctrl_alinea_vsl, True),
        ('Aggressive',      ctrl_aggressive,  False),
    ]:
        if verbose:
            print(f"  Running {name} ...")
        results[name] = simulate(ctrl, entries, tube, name)

    sums = {k: summarise(v) for k, v in results.items()}
    if verbose:
        print()
        print(f"  {'Controller':<20} {'TTS':>9} {'MaxRho':>8} {'MaxQ':>7} "
              f"{'ρ>ρc':>6} {'w>wm':>6}")
        print("  " + "-" * 58)
        for s in sums.values():
            print(f"  {s['label']:<20} {s['tts_total']:>9.2f} "
                  f"{s['max_rho']:>8.2f} {s['max_w']:>7.1f} "
                  f"{s['steps_rho_crit']:>6d} {s['steps_w_max']:>6d}")
        tts_a = sums['ALINEA+VSL']['tts_total']
        tts_t = sums['ALINEA+VSL+Tube']['tts_total']
        print(f"\n  TTS cost of tube filter: {(tts_t - tts_a)/tts_a*100:+.2f}%")

    return results, sums, table_vsl


if __name__ == '__main__':
    print("=" * 70)
    print("  VSL Comparison: ALINEA+VSL vs +Tube vs Aggressive")
    print("=" * 70)
    run()
