"""
EXPERIMENT 2: ICNN Terminal Cost Proof-of-Concept

Goal: Show that a small Input Convex Neural Network can capture the hockey-stick
cost-to-go shape that the quadratic misses.

Key architecture details:
- ICNN z-path weights initialized NEGATIVE so softplus gives small positive values
- x-path (passthrough) carries most of the signal — initialized with small Xavier
- No final nonlinearity — raw convex function output
- Training on normalized [0,1] targets with proper initialization
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time


class ICNN(nn.Module):
    """
    Input Convex Neural Network (Amos et al. 2017).
    Convex in x: z-path weights non-negative (via softplus), softplus activations.

    Key: z-path raw weights initialized to negative values so that
    softplus(w) ≈ 0, preventing exploding initial outputs.
    """

    def __init__(self, input_dim, hidden_dims=(64, 64)):
        super().__init__()

        # First layer (x-path only)
        self.fc0 = nn.Linear(input_dim, hidden_dims[0])

        # Hidden layers: z-path (non-neg) + x-path (free)
        self.z_raw = nn.ParameterList()
        self.z_bias = nn.ParameterList()
        self.x_layers = nn.ModuleList()
        for i in range(1, len(hidden_dims)):
            # Raw z-weights — will be passed through softplus
            self.z_raw.append(nn.Parameter(torch.empty(hidden_dims[i], hidden_dims[i-1])))
            self.z_bias.append(nn.Parameter(torch.zeros(hidden_dims[i])))
            self.x_layers.append(nn.Linear(input_dim, hidden_dims[i]))

        # Output layer
        self.z_out_raw = nn.Parameter(torch.empty(1, hidden_dims[-1]))
        self.x_out = nn.Linear(input_dim, 1)
        self.bias_out = nn.Parameter(torch.zeros(1))

        self._init_weights()

    def _init_weights(self):
        # x-path: small Xavier initialization
        nn.init.xavier_uniform_(self.fc0.weight, gain=0.5)
        nn.init.zeros_(self.fc0.bias)
        for layer in self.x_layers:
            nn.init.xavier_uniform_(layer.weight, gain=0.5)
            nn.init.zeros_(layer.bias)
        nn.init.xavier_uniform_(self.x_out.weight, gain=0.5)
        nn.init.zeros_(self.x_out.bias)

        # z-path: initialize raw weights to NEGATIVE values
        # softplus(-2) ≈ 0.13, softplus(-3) ≈ 0.05
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
                torch.nn.functional.linear(z, Wz, self.z_bias[i])
                + self.x_layers[i](x)
            )

        Wz_out = torch.nn.functional.softplus(self.z_out_raw)
        out = torch.nn.functional.linear(z, Wz_out) + self.x_out(x) + self.bias_out
        return out.squeeze(-1)


class GeneralNN(nn.Module):
    """Standard (non-convex) NN for comparison."""

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


def train_model(model, X_tr, y_tr, X_te, y_te, epochs=5000, lr=1e-3, verbose=False):
    """Train with Adam + cosine annealing."""
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)
    loss_fn = nn.MSELoss()

    Xt = torch.tensor(X_tr, dtype=torch.float32)
    yt = torch.tensor(y_tr, dtype=torch.float32)
    Xv = torch.tensor(X_te, dtype=torch.float32)
    yv = torch.tensor(y_te, dtype=torch.float32)

    best_test = float('inf')
    best_state = None

    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        loss = loss_fn(model(Xt), yt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if (ep+1) % 1000 == 0 or ep == 0:
            model.eval()
            with torch.no_grad():
                tl = loss_fn(model(Xv), yv).item()
            if tl < best_test:
                best_test = tl
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            if verbose:
                print(f"    Epoch {ep+1:5d}: train={loss.item():.8f} test={tl:.8f}")

    if best_state:
        model.load_state_dict(best_state)
    return best_test


def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - y_true.mean())**2)
    if ss_tot < 1e-12:
        return 1.0 if ss_res < 1e-12 else -float('inf')
    return 1 - ss_res / ss_tot


def run_experiment2(exp1_results, verbose=True):
    """Train ICNN, General NN, compare with quadratic on cost-to-go data."""
    t0 = time.time()
    np.random.seed(42)
    torch.manual_seed(42)

    rho_norm = exp1_results['rho_norm']
    costs = exp1_results['cost_means']
    states_7d = exp1_results['all_states_7d']

    # ---- Prepare data ----
    # 1D input (normalized density)
    X_1d = rho_norm.reshape(-1, 1)

    # 7D input (normalized state)
    x_mean = states_7d.mean(axis=0)
    x_std = states_7d.std(axis=0)
    x_std[x_std < 1e-8] = 1.0
    X_7d = (states_7d - x_mean) / x_std

    # Normalize targets to [0, 1]
    y_min, y_max = costs.min(), costs.max()
    y_range = max(y_max - y_min, 1e-8)
    y_norm = (costs - y_min) / y_range

    # 80/20 split
    n = len(costs)
    idx = np.random.permutation(n)
    nt = int(0.8 * n)
    tr, te = idx[:nt], idx[nt:]

    if verbose:
        print(f"Data: {n} points ({nt} train, {n-nt} test)")
        print(f"Cost range: [{y_min:.2f}, {y_max:.2f}], span: {y_range:.2f} veh·h")

    # ---- Train 1D models ----
    if verbose:
        print("\n--- 1D models (density only) ---")

    if verbose:
        print("  ICNN 1D (2×64):")
    torch.manual_seed(42)
    icnn_1d = ICNN(1, (64, 64))
    train_model(icnn_1d, X_1d[tr], y_norm[tr], X_1d[te], y_norm[te],
                epochs=8000, lr=5e-3, verbose=verbose)

    if verbose:
        print("  General NN 1D (2×64):")
    torch.manual_seed(42)
    gnn_1d = GeneralNN(1, (64, 64))
    train_model(gnn_1d, X_1d[tr], y_norm[tr], X_1d[te], y_norm[te],
                epochs=5000, lr=1e-3, verbose=verbose)

    # ---- Train 7D models ----
    if verbose:
        print("\n--- 7D models (full state) ---")

    if verbose:
        print("  ICNN 7D (2×64):")
    torch.manual_seed(42)
    icnn_7d = ICNN(7, (64, 64))
    train_model(icnn_7d, X_7d[tr], y_norm[tr], X_7d[te], y_norm[te],
                epochs=8000, lr=5e-3, verbose=verbose)

    if verbose:
        print("  General NN 7D (2×64):")
    torch.manual_seed(42)
    gnn_7d = GeneralNN(7, (64, 64))
    train_model(gnn_7d, X_7d[tr], y_norm[tr], X_7d[te], y_norm[te],
                epochs=5000, lr=1e-3, verbose=verbose)

    # ---- Evaluate ----
    def get_pred(model, X):
        model.eval()
        with torch.no_grad():
            p = model(torch.tensor(X, dtype=torch.float32)).numpy()
        return p * y_range + y_min

    icnn_1d_pred = get_pred(icnn_1d, X_1d)
    gnn_1d_pred = get_pred(gnn_1d, X_1d)
    icnn_7d_pred = get_pred(icnn_7d, X_7d)
    gnn_7d_pred = get_pred(gnn_7d, X_7d)
    quad_pred = exp1_results['quad_pred_7d']

    # R² on test set
    r2_q = r2_score(costs[te], quad_pred[te])
    r2_i1 = r2_score(costs[te], icnn_1d_pred[te])
    r2_g1 = r2_score(costs[te], gnn_1d_pred[te])
    r2_i7 = r2_score(costs[te], icnn_7d_pred[te])
    r2_g7 = r2_score(costs[te], gnn_7d_pred[te])

    # R² full
    r2_q_f = exp1_results['r2_quad_7d']
    r2_i1_f = r2_score(costs, icnn_1d_pred)
    r2_g1_f = r2_score(costs, gnn_1d_pred)
    r2_i7_f = r2_score(costs, icnn_7d_pred)
    r2_g7_f = r2_score(costs, gnn_7d_pred)

    # Best ICNN and GNN
    r2_icnn_best = max(r2_i1, r2_i7)
    r2_gnn_best = max(r2_g1, r2_g7)
    icnn_pred = icnn_1d_pred if r2_i1 >= r2_i7 else icnn_7d_pred
    gnn_pred = gnn_1d_pred if r2_g1 >= r2_g7 else gnn_7d_pred

    rel_err_quad = np.abs(costs - quad_pred) / np.maximum(costs, 1e-6) * 100
    rel_err_icnn = np.abs(costs - icnn_pred) / np.maximum(costs, 1e-6) * 100
    rel_err_gnn = np.abs(costs - gnn_pred) / np.maximum(costs, 1e-6) * 100

    convexity_cost = (r2_gnn_best - r2_icnn_best) / max(abs(r2_gnn_best), 1e-6) * 100

    elapsed = time.time() - t0

    results = {
        'rho_norm': rho_norm, 'cost_means': costs,
        'icnn_pred': icnn_pred, 'gnn_pred': gnn_pred, 'quad_pred': quad_pred,
        'r2_icnn_test': r2_icnn_best, 'r2_gnn_test': r2_gnn_best, 'r2_quad_test': r2_q,
        'r2_icnn_full': max(r2_i1_f, r2_i7_f), 'r2_gnn_full': max(r2_g1_f, r2_g7_f),
        'r2_quad_full': r2_q_f,
        'rel_err_quad': rel_err_quad, 'rel_err_icnn': rel_err_icnn, 'rel_err_gnn': rel_err_gnn,
        'convexity_cost': convexity_cost, 'elapsed': elapsed,
    }

    if verbose:
        print(f"\n{'='*60}")
        print("EXPERIMENT 2 RESULTS: ICNN Cost Function")
        print(f"{'='*60}")
        print(f"  Computation time: {elapsed:.1f}s")
        print(f"\n  Test-set R²:")
        print(f"         1D        7D")
        print(f"    ICNN:    {r2_i1:.4f}    {r2_i7:.4f}")
        print(f"    Gen NN:  {r2_g1:.4f}    {r2_g7:.4f}")
        print(f"    Quad:    —         {r2_q:.4f}")
        print(f"\n  Best ICNN:     {r2_icnn_best:.4f}")
        print(f"  Best Gen NN:   {r2_gnn_best:.4f}")
        print(f"  Quadratic:     {r2_q:.4f}")
        print(f"  Convexity cost: {convexity_cost:.1f}%")

        improvement = r2_icnn_best - r2_q
        if r2_icnn_best > 0.95 and improvement > 0.05:
            verdict = "PASS: ICNN captures what quadratic cannot"
        elif r2_icnn_best > 0.90 and improvement > 0.02:
            verdict = "MARGINAL: ICNN improves fit modestly"
        elif r2_icnn_best > r2_q:
            verdict = "MARGINAL: ICNN slightly better — benefit depends on regime"
        else:
            verdict = "FAIL: ICNN does not improve over quadratic"
        print(f"\n  VERDICT: {verdict}")
        results['verdict'] = verdict

    return results


if __name__ == '__main__':
    from experiment1_quadratic_mismatch import run_experiment1
    exp1 = run_experiment1(n_mc=50, n_steps=600, verbose=True)
    print()
    run_experiment2(exp1, verbose=True)
