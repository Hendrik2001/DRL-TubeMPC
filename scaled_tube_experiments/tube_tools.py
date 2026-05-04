"""
Tube computation utilities: ellipsoidal RPI, zonotopic RPI, LQR stabilization.
"""
import numpy as np
from scipy.linalg import solve_discrete_lyapunov, solve_discrete_are


def spectral_radius(A):
    return np.max(np.abs(np.linalg.eigvals(A)))


def lqr_gain(A, B, Q=None, R=None):
    """Discrete LQR. Returns K such that A - B K is stable (u = -K x)."""
    n = A.shape[0]
    m = B.shape[1]
    if Q is None:
        Q = np.eye(n)
    if R is None:
        R = 10.0 * np.eye(m)
    P = solve_discrete_are(A, B, Q, R)
    K = np.linalg.solve(B.T @ P @ B + R, B.T @ P @ A)
    return K, P


def stabilize(A, B, Q=None, R=None):
    """If A is not Schur, build A_cl = A - B K. Returns (A_cl, K_or_None)."""
    if spectral_radius(A) < 1.0 - 1e-9:
        return A, None
    try:
        K, _ = lqr_gain(A, B, Q, R)
        A_cl = A - B @ K
        return A_cl, K
    except Exception:
        return A, None


def ellipsoidal_rpi(A_cl, Bw, Delta):
    """
    Ellipsoidal RPI via discrete Lyapunov equation.

    P = A_cl P A_cl^T + Bw Σ_w Bw^T
    Ellipsoid {e : e^T P^{-1} e ≤ c}, widths = sqrt(c * diag(P)).

    Delta = [Δ_d, Δ_q] disturbance half-widths.
    Returns (P, widths, bounding_box_halfwidths).
    """
    Sigma_w = np.diag(Delta ** 2)  # uniform approximation
    Q = Bw @ Sigma_w @ Bw.T
    try:
        P = solve_discrete_lyapunov(A_cl, Q)
    except Exception:
        return None, None, None
    # Scale factor c: use c = n (chi-square-ish bound for n-dim ellipsoid)
    n = A_cl.shape[0]
    c = float(n)
    halfwidths = np.sqrt(c * np.diag(P))
    return P, c, halfwidths


def zonotopic_rpi(A_cl, Bw, Delta, eps=0.01, max_iter=200):
    """
    Zonotopic RPI: Z_s = ⊕_{i=0}^{s-1} A_cl^i W.
    W is a box with generator matrix Gw = Bw @ diag(Delta).
    Returns (generators, box_halfwidths, s_used).
    """
    Gw = Bw * Delta  # (n, 2) – each column scaled
    G = Gw.copy()
    Ai = np.eye(A_cl.shape[0])
    prev_bbox = np.sum(np.abs(G), axis=1)
    for s in range(1, max_iter + 1):
        Ai = A_cl @ Ai
        new_gen = Ai @ Gw
        G = np.concatenate([G, new_gen], axis=1)
        bbox = np.sum(np.abs(G), axis=1)
        rel = np.linalg.norm(bbox - prev_bbox) / max(np.linalg.norm(prev_bbox), 1e-12)
        if rel < eps:
            return G, bbox, s
        prev_bbox = bbox
    return G, np.sum(np.abs(G), axis=1), max_iter


def reachable_box_from_ellipsoid(A_cl, P_i, c_i, Bw, Delta):
    """
    One-step reachable set from an ellipsoidal RPI under the same dynamics,
    plus a fresh disturbance box.

    Returns bounding-box halfwidths of R_i = A_cl · Z_i ⊕ W.
    """
    # Halfwidths of A_cl · Z_i (ellipsoid with covariance A P A^T)
    P_reach = A_cl @ P_i @ A_cl.T
    hw_state = np.sqrt(c_i * np.diag(P_reach))
    # Add disturbance box
    hw_dist = np.abs(Bw) @ Delta
    return hw_state + hw_dist
