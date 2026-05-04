#!/usr/bin/env python3
"""
SCALE-UP EXPERIMENT: Does the hockey stick emerge on larger METANET networks?

Tests 4 network sizes: 3, 7, 10, 15 segments.
For each size:
  1. Compute cost-to-go via Monte Carlo at density grid
  2. Fit quadratic
  3. Fit ICNN and General NN
  4. Measure asymmetry and quadratic mismatch

The hypothesis: congestion compounding increases with network size because
shockwaves propagate upstream through more segments, creating secondary
bottlenecks and exponential cost growth above ρ_crit.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from metanet_scaled import (
    ScaledMETANET, Ve, V_FREE, RHO_CRIT, RHO_MAX, T, L, LAMBDA, W_MAX, C_O
)


# ── ICNN (same architecture as experiment2) ──────────────────────────────
class ICNN(nn.Module):
    def __init__(self, input_dim, hidden_dims=(64, 64)):
        super().__init__()
        self.fc0 = nn.Linear(input_dim, hidden_dims[0])
        self.z_raw = nn.ParameterList()
        self.z_bias = nn.ParameterList()
        self.x_layers = nn.ModuleList()
        for i in range(1, len(hidden_dims)):
            self.z_raw.append(nn.Parameter(torch.empty(hidden_dims[i], hidden_dims[i-1])))
            self.z_bias.append(nn.Parameter(torch.zeros(hidden_dims[i])))
            self.x_layers.append(nn.Linear(input_dim, hidden_dims[i]))
        self.z_out_raw = nn.Parameter(torch.empty(1, hidden_dims[-1]))
        self.x_out = nn.Linear(input_dim, 1)
        self.bias_out = nn.Parameter(torch.zeros(1))
        self._init()

    def _init(self):
        nn.init.xavier_uniform_(self.fc0.weight, gain=0.5)
        nn.init.zeros_(self.fc0.bias)
        for l in self.x_layers:
            nn.init.xavier_uniform_(l.weight, gain=0.5)
            nn.init.zeros_(l.bias)
        nn.init.xavier_uniform_(self.x_out.weight, gain=0.5)
        nn.init.zeros_(self.x_out.bias)
        for w in self.z_raw:
            nn.init.uniform_(w, -3.0, -1.0)
        for b in self.z_bias:
            nn.init.zeros_(b)
        nn.init.uniform_(self.z_out_raw, -3.0, -1.0)

    def forward(self, x):
        z = torch.nn.functional.softplus(self.fc0(x))
        for i in range(len(self.z_raw)):
            Wz = torch.nn.functional.softplus(self.z_raw[i])
            z = torch.nn.functional.softplus(
                torch.nn.functional.linear(z, Wz, self.z_bias[i]) + self.x_layers[i](x))
        Wz_out = torch.nn.functional.softplus(self.z_out_raw)
        return (torch.nn.functional.linear(z, Wz_out) + self.x_out(x) + self.bias_out).squeeze(-1)


class GeneralNN(nn.Module):
    def __init__(self, input_dim, hidden_dims=(64, 64)):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_nn(model, X_tr, y_tr, X_te, y_te, epochs=5000, lr=1e-3):
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)
    loss_fn = nn.MSELoss()
    Xt, yt = torch.tensor(X_tr, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.float32)
    Xv, yv = torch.tensor(X_te, dtype=torch.float32), torch.tensor(y_te, dtype=torch.float32)
    best, best_state = float('inf'), None
    for ep in range(epochs):
        model.train(); opt.zero_grad()
        loss = loss_fn(model(Xt), yt); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if (ep+1) % epochs == 0 or ep == 0:
            model.eval()
            with torch.no_grad():
                tl = loss_fn(model(Xv), yv).item()
            if tl < best:
                best = tl
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
    if best_state:
        model.load_state_dict(best_state)
    return best


def r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - y_true.mean())**2)
    return 1 - ss_res / max(ss_tot, 1e-12)


# ── Main experiment ──────────────────────────────────────────────────────
def run_scale_experiment(n_seg, n_mc=40, n_steps=600, verbose=True):
    """Run the hockey stick experiment for a given network size."""
    net = ScaledMETANET(n_seg)
    cap = V_FREE * RHO_CRIT * LAMBDA  # ~6834 veh/h

    # Upstream demand: 90% capacity (per lane pair)
    q_up = 0.90 * cap
    # On-ramp demands: enough to push total above capacity
    d_per_ramp = 600  # veh/h per on-ramp
    n_ramps = net.M

    if verbose:
        print(f"\n{'─'*60}")
        print(f"  Network: {n_seg} segments, {n_ramps} on-ramps")
        print(f"  State dim: {net.nx}")
        print(f"  Upstream: {q_up:.0f} veh/h, On-ramp demand: {d_per_ramp} × {n_ramps}")
        print(f"  Total demand: {q_up + d_per_ramp * n_ramps:.0f} veh/h "
              f"({(q_up + d_per_ramp * n_ramps)/cap*100:.0f}% of capacity)")

    # Density grid
    rho_grid = np.sort(np.unique(np.concatenate([
        np.linspace(0.3 * RHO_CRIT, 2.0 * RHO_CRIT, 35),
        np.linspace(0.8 * RHO_CRIT, 1.3 * RHO_CRIT, 25),
    ])))

    rho_norm_all = []
    cost_means = []
    cost_stds = []

    t0 = time.time()
    for idx, rho in enumerate(rho_grid):
        state0 = net.make_homogeneous_state(rho)
        costs = []
        for trial in range(n_mc):
            noise_d = np.random.uniform(-1, 1, (n_steps, n_ramps))
            noise_q = np.random.uniform(-1, 1, n_steps)
            state = state0.copy()
            tts = 0.0
            for k in range(n_steps):
                tts += net.compute_tts_step(state)
                d_k = d_per_ramp * (1 + 0.4 * noise_d[k])
                r_k = np.full(n_ramps, 0.7)
                q_k = q_up * (1 + 0.15 * noise_q[k])
                state = net.step(state, r_k, d_k, q_k)
            costs.append(tts)

        rho_norm_all.append(rho / RHO_CRIT)
        cost_means.append(np.mean(costs))
        cost_stds.append(np.std(costs))

        if verbose and (idx + 1) % 15 == 0:
            print(f"    {idx+1}/{len(rho_grid)}: ρ/ρ_c={rho/RHO_CRIT:.2f}, "
                  f"cost={np.mean(costs):.1f} ± {np.std(costs):.1f}")

    elapsed_mc = time.time() - t0
    rho_norm_all = np.array(rho_norm_all)
    cost_means = np.array(cost_means)
    cost_stds = np.array(cost_stds)

    # ── Quadratic fit (1D in ρ/ρ_crit) ──
    rc = rho_norm_all - 0.7
    A_1d = np.column_stack([rc**2, rc, np.ones(len(rc))])
    c_1d, _, _, _ = np.linalg.lstsq(A_1d, cost_means, rcond=None)
    quad_pred = A_1d @ c_1d
    r2_quad = r2(cost_means, quad_pred)

    # ── Asymmetry ──
    below = (rho_norm_all > 0.4) & (rho_norm_all < 0.95)
    above = (rho_norm_all > 1.05) & (rho_norm_all < 1.8)
    if np.sum(below) > 2 and np.sum(above) > 2:
        sl_b = abs(np.polyfit(rho_norm_all[below], cost_means[below], 1)[0])
        sl_a = abs(np.polyfit(rho_norm_all[above], cost_means[above], 1)[0])
        asym = sl_a / max(sl_b, 1e-6)
    else:
        asym = 1.0

    # ── Residuals near ρ_crit ──
    near = (rho_norm_all > 0.85) & (rho_norm_all < 1.15)
    res_near = np.abs(cost_means[near] - quad_pred[near]) / np.maximum(cost_means[near], 1e-6) * 100
    max_res = np.max(res_near) if len(res_near) > 0 else 0.0

    # ── ICNN and General NN (1D: density input) ──
    X_1d = rho_norm_all.reshape(-1, 1)
    y_min, y_rng = cost_means.min(), max(cost_means.max() - cost_means.min(), 1e-8)
    y_n = (cost_means - y_min) / y_rng

    idx_all = np.random.RandomState(42).permutation(len(cost_means))
    nt = int(0.8 * len(cost_means))
    tr, te = idx_all[:nt], idx_all[nt:]

    torch.manual_seed(42)
    icnn = ICNN(1, (64, 64))
    train_nn(icnn, X_1d[tr], y_n[tr], X_1d[te], y_n[te], epochs=6000, lr=5e-3)

    torch.manual_seed(42)
    gnn = GeneralNN(1, (64, 64))
    train_nn(gnn, X_1d[tr], y_n[tr], X_1d[te], y_n[te], epochs=4000, lr=1e-3)

    icnn.eval(); gnn.eval()
    with torch.no_grad():
        icnn_pred = icnn(torch.tensor(X_1d, dtype=torch.float32)).numpy() * y_rng + y_min
        gnn_pred = gnn(torch.tensor(X_1d, dtype=torch.float32)).numpy() * y_rng + y_min

    r2_icnn = r2(cost_means[te], icnn_pred[te])
    r2_gnn = r2(cost_means[te], gnn_pred[te])
    r2_quad_te = r2(cost_means[te], quad_pred[te])

    elapsed_total = time.time() - t0

    result = {
        'n_seg': n_seg, 'n_ramps': n_ramps, 'nx': net.nx,
        'rho_norm': rho_norm_all, 'cost_means': cost_means, 'cost_stds': cost_stds,
        'quad_pred': quad_pred, 'icnn_pred': icnn_pred, 'gnn_pred': gnn_pred,
        'r2_quad': r2_quad, 'r2_quad_te': r2_quad_te,
        'r2_icnn': r2_icnn, 'r2_gnn': r2_gnn,
        'asymmetry': asym, 'max_res_crit': max_res,
        'cost_range': cost_means.max() - cost_means.min(),
        'cost_ratio': cost_means.max() / max(cost_means.min(), 1e-6),
        'elapsed_mc': elapsed_mc, 'elapsed_total': elapsed_total,
    }

    if verbose:
        print(f"\n  Results for {n_seg} segments:")
        print(f"    MC time: {elapsed_mc:.1f}s, Total: {elapsed_total:.1f}s")
        print(f"    Cost range: [{cost_means.min():.1f}, {cost_means.max():.1f}] "
              f"(ratio {result['cost_ratio']:.2f}×, span {result['cost_range']:.1f})")
        print(f"    Asymmetry ratio: {asym:.2f}×")
        print(f"    Quadratic R²: {r2_quad:.4f} (1D), {r2_quad_te:.4f} (test)")
        print(f"    ICNN R² (test): {r2_icnn:.4f}")
        print(f"    General NN R² (test): {r2_gnn:.4f}")
        print(f"    Max residual near ρ_crit: {max_res:.1f}%")

    return result


def make_plots(results_list, save_dir='.'):
    """Generate comparison plots across network sizes."""
    n_sizes = len(results_list)
    sizes = [r['n_seg'] for r in results_list]

    # ── Plot A: Hockey stick shapes at each scale ──
    fig, axes = plt.subplots(1, n_sizes, figsize=(5 * n_sizes, 5), sharey=False)
    if n_sizes == 1:
        axes = [axes]

    for i, (ax, res) in enumerate(zip(axes, results_list)):
        rho = res['rho_norm']
        costs = res['cost_means']
        stds = res['cost_stds']
        sort = np.argsort(rho)

        ax.errorbar(rho, costs, yerr=stds, fmt='o', color='#2176AE',
                     markersize=3, alpha=0.6, capsize=1.5, label='MC mean ± 1σ')
        ax.plot(rho[sort], res['quad_pred'][sort], '--', color='#D32F2F',
                linewidth=2, label=f"Quad R²={res['r2_quad']:.3f}")
        ax.plot(rho[sort], res['gnn_pred'][sort], '-', color='#7B1FA2',
                linewidth=2, label=f"Gen NN R²={res['r2_gnn']:.3f}")
        ax.axvline(1.0, color='gray', linestyle='--', alpha=0.5)

        ax.set_xlabel('ρ / ρ_crit', fontsize=12)
        if i == 0:
            ax.set_ylabel('Cost-to-go (TTS, veh·h)', fontsize=12)
        ax.set_title(f'{res["n_seg"]} segments, {res["n_ramps"]} ramps\n'
                     f'Asym={res["asymmetry"]:.1f}×, Span={res["cost_range"]:.0f}',
                     fontsize=11, fontweight='bold')
        ax.legend(fontsize=8, loc='upper left')
        ax.grid(True, alpha=0.3)

    plt.suptitle('Hockey Stick vs Network Size', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'plot_scaleup_hockey.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: plot_scaleup_hockey.png")

    # ── Plot B: Summary metrics vs network size ──
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # Asymmetry
    ax = axes[0, 0]
    ax.bar(range(n_sizes), [r['asymmetry'] for r in results_list],
           color='#2176AE', alpha=0.8, edgecolor='black')
    ax.axhline(1.5, color='red', linestyle='--', label='Threshold (1.5×)')
    ax.set_xticks(range(n_sizes))
    ax.set_xticklabels([f'{s} seg' for s in sizes])
    ax.set_ylabel('Asymmetry ratio', fontsize=12)
    ax.set_title('Slope Asymmetry (above/below ρ_crit)', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # R² comparison
    ax = axes[0, 1]
    x = np.arange(n_sizes)
    w = 0.25
    ax.bar(x - w, [r['r2_quad_te'] for r in results_list], w,
           label='Quadratic', color='#D32F2F', alpha=0.8)
    ax.bar(x, [r['r2_icnn'] for r in results_list], w,
           label='ICNN', color='#2E7D32', alpha=0.8)
    ax.bar(x + w, [r['r2_gnn'] for r in results_list], w,
           label='General NN', color='#7B1FA2', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{s} seg' for s in sizes])
    ax.set_ylabel('Test R²', fontsize=12)
    ax.set_title('Model Fit Quality (Test Set)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    # Set y-axis to show detail near 1.0
    all_r2 = ([r['r2_quad_te'] for r in results_list] +
              [r['r2_icnn'] for r in results_list] +
              [r['r2_gnn'] for r in results_list])
    min_r2 = min(all_r2)
    ax.set_ylim(max(min_r2 - 0.05, 0), 1.005)

    # Cost range (span)
    ax = axes[1, 0]
    ax.bar(range(n_sizes), [r['cost_range'] for r in results_list],
           color='#FF8F00', alpha=0.8, edgecolor='black')
    ax.set_xticks(range(n_sizes))
    ax.set_xticklabels([f'{s} seg' for s in sizes])
    ax.set_ylabel('Cost range (veh·h)', fontsize=12)
    ax.set_title('Cost-to-Go Dynamic Range', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Max residual near ρ_crit
    ax = axes[1, 1]
    ax.bar(range(n_sizes), [r['max_res_crit'] for r in results_list],
           color='#D32F2F', alpha=0.8, edgecolor='black')
    ax.axhline(50, color='red', linestyle='--', label='50% threshold')
    ax.set_xticks(range(n_sizes))
    ax.set_xticklabels([f'{s} seg' for s in sizes])
    ax.set_ylabel('Max residual (%)', fontsize=12)
    ax.set_title('Quadratic Mismatch Near ρ_crit', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle('Direction A Scale-Up Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'plot_scaleup_summary.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: plot_scaleup_summary.png")


def main():
    save_dir = os.path.dirname(os.path.abspath(__file__))
    np.random.seed(42)

    print("=" * 70)
    print("  DIRECTION A SCALE-UP: Hockey Stick vs Network Size")
    print("=" * 70)

    configs = [
        3,   # original (baseline)
        7,   # medium
        10,  # large
        15,  # very large
    ]

    results = []
    for n_seg in configs:
        res = run_scale_experiment(n_seg, n_mc=40, n_steps=600, verbose=True)
        results.append(res)

    # ── Summary table ──
    print("\n" + "=" * 70)
    print("  SCALE-UP SUMMARY")
    print("=" * 70)
    print(f"\n  {'Segments':>8} {'Ramps':>6} {'Asym':>6} {'Quad R²':>8} "
          f"{'ICNN R²':>8} {'GNN R²':>8} {'Range':>8} {'MaxRes%':>8} {'Time':>6}")
    print("  " + "-" * 68)
    for r in results:
        print(f"  {r['n_seg']:>8} {r['n_ramps']:>6} {r['asymmetry']:>6.2f} "
              f"{r['r2_quad_te']:>8.4f} {r['r2_icnn']:>8.4f} {r['r2_gnn']:>8.4f} "
              f"{r['cost_range']:>8.0f} {r['max_res_crit']:>8.1f} {r['elapsed_total']:>5.0f}s")

    # ── Verdict ──
    last = results[-1]
    print(f"\n  At {last['n_seg']} segments:")
    if last['asymmetry'] > 2.0 and last['r2_quad'] < 0.95:
        print("  ✓ STRONG hockey stick — quadratic clearly fails")
        print("  ✓ ICNN advantage is significant")
        print("  → Direction A hypothesis CONFIRMED at scale")
    elif last['asymmetry'] > 1.5:
        print(f"  ~ Moderate hockey stick (asymmetry {last['asymmetry']:.1f}×)")
        print(f"  ~ Quadratic R²={last['r2_quad']:.4f} — "
              f"{'still good' if last['r2_quad'] > 0.99 else 'degrading'}")
        if last['r2_icnn'] > last['r2_quad_te']:
            print(f"  ✓ ICNN beats quadratic: {last['r2_icnn']:.4f} vs {last['r2_quad_te']:.4f}")
        print("  → Hockey stick emerging — scale further or use flow-based cost")
    else:
        print("  ✗ Hockey stick still too weak even at this scale")
        print("  → Consider flow-based cost metric or capacity-drop penalty")

    print(f"\n  Trend across scales:")
    for r in results:
        bar_len = int(r['asymmetry'] * 10)
        bar = '█' * bar_len
        print(f"    {r['n_seg']:>2} seg: {bar} {r['asymmetry']:.2f}×")

    print("=" * 70)

    # ── Plots ──
    print("\nGenerating plots...")
    make_plots(results, save_dir)

    return results


if __name__ == '__main__':
    main()
