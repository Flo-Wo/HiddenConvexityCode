from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from problem_smooth_highDim import proj_box


class GradCounter:
    def __init__(self) -> None:
        self.count = 0

    def wrap(
        self, f: Callable[[np.ndarray], np.ndarray]
    ) -> Callable[[np.ndarray], np.ndarray]:
        def g(x: np.ndarray) -> np.ndarray:
            self.count += 1
            return f(x)

        return g


InnerSolver = Callable[
    [
        Callable[[np.ndarray], float],
        Callable[[np.ndarray], np.ndarray],
        Callable[[np.ndarray], float],
        Callable[[np.ndarray], np.ndarray],
        np.ndarray,
        int,
        float,
        float,
        Callable[[np.ndarray], np.ndarray],
        Optional[Callable[[int], float]],
        float,
    ],
    Tuple[np.ndarray, Dict],
]


@dataclass
class SwSGSolver:
    stepsize_scale: float = 0.2

    def __call__(
        self,
        phi1,
        dphi1,
        phi2,
        dphi2,
        xk,
        Tin,
        tau,
        eps_in,
        proj,
        stepsizes=None,
        alpha: float = 1.0,
    ):
        if stepsizes is None:
            stepsizes = lambda t: self.stepsize_scale / (t + 1.0)

        def phi2_sh(x: np.ndarray) -> float:
            return phi2(x) - tau + (alpha * tau) / 3.0

        z = xk.copy()
        z_hist: List[np.ndarray] = [z.copy()]
        F_idx: List[int] = []

        for t in range(1, Tin + 1):
            if phi2_sh(z) <= eps_in:
                g = dphi1(z)
                F_idx.append(t)
            else:
                g = dphi2(z)
            z = proj(z - stepsizes(t - 1) * g)
            z_hist.append(z.copy())

        if F_idx:
            w = np.array(F_idx, dtype=float)
            zbar = sum(w[i] * z_hist[F_idx[i]] for i in range(len(F_idx))) / w.sum()
        else:
            zbar = z

        return zbar, {"feasible_indices": F_idx, "Tin": Tin}


@dataclass
class ACGDSolver:
    rho_hat: float
    box_l: float
    box_u: float
    L_init: Optional[float] = None
    safety_factor: float = 1.5
    shift_scale: float = 0.25  # controls how conservative the constraint shift is
    L_cap: float = 10.0

    def _estimate_L(
        self, phi1, dphi1, x: np.ndarray, L_guess: Optional[float]
    ) -> float:
        L = 1.0 if L_guess is None else float(L_guess)
        fx = phi1(x)
        g = dphi1(x)
        for _ in range(20):
            L_eff = min(L, self.L_cap)
            x_new = proj_box(x - g / L_eff, self.box_l, self.box_u)
            if phi1(x_new) <= fx - (np.dot(g, g)) / (2.0 * L_eff):
                return min(L_eff * self.safety_factor, self.L_cap)
            if L_eff >= self.L_cap:
                return self.L_cap
            L = L_eff * 2.0
        return min(L, self.L_cap)

    def _solve_qp(self, z_prev, z_curr, pi_t, nu_t, phi2_sh_at_z, eta_t):
        x0 = z_prev - (1.0 / eta_t) * pi_t
        a = nu_t
        b = -phi2_sh_at_z + float(a @ z_curr)

        x = np.clip(x0, self.box_l, self.box_u)
        if float(a @ x) <= b + 1e-12:
            return x

        def x_of_lambda(lmbd: float) -> np.ndarray:
            return np.clip(x0 - (lmbd / eta_t) * a, self.box_l, self.box_u)

        def h(lmbd: float) -> float:
            return float(a @ x_of_lambda(lmbd)) - b

        lo, hi = 0.0, 1.0
        while h(hi) > 0.0 and hi < 1e6:
            hi *= 2.0
        for _ in range(50):
            mid = 0.5 * (lo + hi)
            if h(mid) > 0.0:
                lo = mid
            else:
                hi = mid
        return x_of_lambda(hi)

    def __call__(
        self,
        phi1,
        dphi1,
        phi2,
        dphi2,
        xk,
        Tin,
        tau,
        eps_in,
        proj,
        stepsizes=None,
        alpha: float = 1.0,
    ):
        shift_margin = tau * max(self.shift_scale * alpha, 0.0)
        b = max((alpha * tau) / 3.0 - shift_margin, 0.0)

        def phi2_sh(x: np.ndarray) -> float:
            return phi2(x) - b

        L_r = self._estimate_L(phi1, dphi1, xk, self.L_init)
        kappa_r = L_r / max(self.rho_hat, 1e-12)
        sqrt_k = math.sqrt(kappa_r)
        self.L_init = L_r

        z_minus2 = xk.copy()
        z_minus1 = xk.copy()
        tau_prev = 0.0
        omega_prev = 1.0
        weights: List[float] = []
        zs: List[np.ndarray] = []

        for t in range(1, Tin + 1):
            tau_t = min((t - 1) / 2.0, sqrt_k)
            theta_t = tau_t / (tau_prev + 1.0)
            omega_t = 1.0 if t == 1 else omega_prev / max(theta_t, 1e-12)
            omega_prev, tau_prev = omega_t, tau_t

            z_tilde = z_minus1 + theta_t * (z_minus1 - z_minus2)
            z_mid = (tau_t * z_minus1 + z_tilde) / (1.0 + tau_t)

            pi_t = dphi1(z_mid)
            nu_t = dphi2(z_mid)
            phi2_val = phi2_sh(z_mid)

            tau_next = min(t / 2.0, sqrt_k)
            eta_t = L_r / max(tau_next, 1e-12)

            z_new = self._solve_qp(z_minus1, z_mid, pi_t, nu_t, phi2_val, eta_t)

            z_minus2, z_minus1 = z_minus1, z_new
            zs.append(z_new.copy())
            weights.append(omega_t)

        W = max(sum(weights), 1e-12)
        zbar = sum(w * z for w, z in zip(weights, zs)) / W
        return zbar, {"L_r": L_r, "kappa_r": kappa_r, "Tin": Tin}


def _project_onto_halfspace(x: np.ndarray, a: np.ndarray, b: float) -> np.ndarray:
    viol = float(np.dot(a, x) - b)
    if viol <= 0.0:
        return x
    denom = float(np.dot(a, a))
    if denom <= 1e-12:
        return x
    return x - (viol / denom) * a


def _project_onto_box(x: np.ndarray, box_l: float, box_u: float) -> np.ndarray:
    return np.clip(x, box_l, box_u)


def _project_onto_intersection(
    z: np.ndarray,
    halfspaces: List[Tuple[np.ndarray, float]],
    box_l: float,
    box_u: float,
    tol: float = 1e-6,
    max_iters: int = 100,
) -> Tuple[np.ndarray, int]:
    if not halfspaces:
        x = _project_onto_box(z, box_l, box_u)
        return x, 1

    sets: List[Tuple[str, Optional[Tuple[np.ndarray, float]]]] = [("box", None)]
    for a, b in halfspaces:
        sets.append(("halfspace", (a.copy(), float(b))))

    x = z.copy()
    residuals = [np.zeros_like(z) for _ in sets]

    for it in range(1, max_iters + 1):
        x_prev = x.copy()
        for idx, (kind, data) in enumerate(sets):
            y = x + residuals[idx]
            if kind == "box":
                x_new = _project_onto_box(y, box_l, box_u)
            else:
                assert data is not None
                a, b = data
                x_new = _project_onto_halfspace(y, a, b)
            residuals[idx] = y - x_new
            x = x_new
        if np.linalg.norm(x - x_prev) <= tol:
            break
    return _project_onto_box(x, box_l, box_u), it


def IPPM_with_counts(
    F0: Callable[[np.ndarray], float],
    dF0: Callable[[np.ndarray], np.ndarray],
    F1: Callable[[np.ndarray], float],
    dF1: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    tau: float,
    N: int,
    rho_hat: float,
    inner_solver: InnerSolver,
    Tin: int,
    eps_in: float,
    proj: Callable[[np.ndarray], np.ndarray],
    alpha: float = 1.0,
):
    xk = x0.copy()
    F0_vals = [F0(xk)]
    F1_viols = [F1(xk) - 1.0]
    grad_calls = [0]

    c0, c1 = GradCounter(), GradCounter()
    dF0_w = c0.wrap(dF0)
    dF1_w = c1.wrap(dF1)

    for _ in range(N):

        def phi1(x, xk=xk):
            return F0(x) + 0.5 * rho_hat * np.sum((x - xk) ** 2)

        def dphi1(x, xk=xk):
            return dF0_w(x) + rho_hat * (x - xk)

        def phi2(x, xk=xk):
            return F1(x) - 1.0 + 0.5 * rho_hat * np.sum((x - xk) ** 2)

        def dphi2(x, xk=xk):
            return dF1_w(x) + rho_hat * (x - xk)

        xk, _ = inner_solver(
            phi1, dphi1, phi2, dphi2, xk, Tin, tau, eps_in, proj, None, alpha
        )
        F0_vals.append(F0(xk))
        F1_viols.append(F1(xk) - 1.0)
        grad_calls.append(c0.count + c1.count)

    return np.array(F0_vals), np.array(F1_viols), np.array(grad_calls)


def shifted_bundle_level(
    F0: Callable[[np.ndarray], float],
    dF0: Callable[[np.ndarray], np.ndarray],
    F2: Callable[[np.ndarray], float],
    dF2: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    eta: float,
    T: int,
    tau: float,
    alpha: float,
    beta: float,
    lam: float,
    box_l: float,
    box_u: float,
    proj_tol: float = 1e-6,
    proj_max_iters: int = 200,
) -> Tuple[np.ndarray, Dict[str, object]]:
    x_curr = _project_onto_box(np.array(x0, dtype=float), box_l, box_u)

    iterates: List[np.ndarray] = [x_curr.copy()]
    F0_vals: List[float] = []
    F2_vals: List[float] = []
    penalties: List[float] = []
    qp_iters: List[int] = []

    def _record(x: np.ndarray) -> Tuple[float, float, float]:
        f0 = float(F0(x))
        f2 = float(F2(x))
        pen = f0 + lam * max(f2, 0.0)
        F0_vals.append(f0)
        F2_vals.append(f2)
        penalties.append(pen)
        return f0, f2, pen

    f0_curr, f2_curr, _ = _record(x_curr)

    for _ in range(T):
        g0 = dF0(x_curr)
        g2 = dF2(x_curr)

        rhs1 = (
            (1.0 - alpha * beta) * f0_curr
            + alpha * beta * eta
            + (1.0 - beta) * alpha * lam * max(f2_curr, 0.0)
            + tau
        )
        rhs2 = (1.0 - alpha) * f2_curr + tau

        halfspaces: List[Tuple[np.ndarray, float]] = []

        if np.linalg.norm(g0) > 1e-12:
            b1 = rhs1 - f0_curr + float(np.dot(g0, x_curr))
            halfspaces.append((g0, b1))
        else:
            if f0_curr > rhs1:
                raise RuntimeError(
                    "Infeasible minorant for objective in shifted bundle-level step."
                )

        if np.linalg.norm(g2) > 1e-12:
            b2 = rhs2 - f2_curr + float(np.dot(g2, x_curr))
            halfspaces.append((g2, b2))
        else:
            if f2_curr > rhs2:
                raise RuntimeError(
                    "Infeasible minorant for constraint in shifted bundle-level step."
                )

        x_next, proj_iters = _project_onto_intersection(
            x_curr, halfspaces, box_l, box_u, tol=proj_tol, max_iters=proj_max_iters
        )
        qp_iters.append(proj_iters)
        x_curr = x_next

        iterates.append(x_curr.copy())
        f0_curr, f2_curr, _ = _record(x_curr)

    penalties_np = np.array(penalties, dtype=float)
    best_idx = int(np.argmin(penalties_np))
    x_best = iterates[best_idx]

    stats: Dict[str, object] = {
        "iterates": iterates,
        "F0": np.array(F0_vals, dtype=float),
        "F2": np.array(F2_vals, dtype=float),
        "penalties": penalties_np,
        "best_index": best_idx,
        "best_penalty": float(penalties_np[best_idx]),
        "eta": float(eta),
        "alpha": float(alpha),
        "beta": float(beta),
        "lambda": float(lam),
        "tau": float(tau),
        "qp_iterations": qp_iters,
    }
    return x_best.copy(), stats


def adaptive_shifted_bundle_level(
    F0: Callable[[np.ndarray], float],
    dF0: Callable[[np.ndarray], np.ndarray],
    F2: Callable[[np.ndarray], float],
    dF2: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    eta0: float,
    N: int,
    T: int,
    tau: float,
    alpha: float,
    beta: float,
    lam: float,
    box_l: float,
    box_u: float,
    proj_tol: float = 1e-6,
    proj_max_iters: int = 200,
) -> Dict[str, object]:
    c0, c2 = GradCounter(), GradCounter()
    dF0_w = c0.wrap(dF0)
    dF2_w = c2.wrap(dF2)

    eta_vals = [float(eta0)]
    outer_solutions: List[np.ndarray] = []
    outer_stats: List[Dict[str, object]] = []
    penalties: List[float] = []
    F0_final: List[float] = []
    F2_final: List[float] = []
    grad_calls = [0]

    eta_k = float(eta0)
    x_best: Optional[np.ndarray] = None
    best_penalty = float("inf")
    best_index = -1

    for k in range(1, N + 1):
        x_k, stats_k = shifted_bundle_level(
            F0,
            dF0_w,
            F2,
            dF2_w,
            x0,
            eta=eta_k,
            T=T,
            tau=tau,
            alpha=alpha,
            beta=beta,
            lam=lam,
            box_l=box_l,
            box_u=box_u,
            proj_tol=proj_tol,
            proj_max_iters=proj_max_iters,
        )

        f0_val = float(F0(x_k))
        f2_val = float(F2(x_k))
        penalty_val = f0_val + lam * max(f2_val, 0.0)

        outer_solutions.append(x_k.copy())
        outer_stats.append(stats_k)
        F0_final.append(f0_val)
        F2_final.append(f2_val)
        penalties.append(penalty_val)

        if penalty_val < best_penalty:
            x_best = x_k.copy()
            best_penalty = penalty_val
            best_index = k - 1

        eta_k = beta * eta_k + (1.0 - beta) * penalty_val
        eta_vals.append(float(eta_k))
        grad_calls.append(c0.count + c2.count)

    if x_best is None:
        x_best = np.array(x0, dtype=float)

    return {
        "x_best": x_best,
        "best_penalty": best_penalty,
        "best_outer_index": best_index,
        "outer_solutions": outer_solutions,
        "outer_stats": outer_stats,
        "penalties": np.array(penalties, dtype=float),
        "F0": np.array(F0_final, dtype=float),
        "F2": np.array(F2_final, dtype=float),
        "eta_values": np.array(eta_vals, dtype=float),
        "grad_calls": np.array(grad_calls, dtype=int),
        "params": {
            "eta0": float(eta0),
            "N": int(N),
            "T": int(T),
            "tau": float(tau),
            "alpha": float(alpha),
            "beta": float(beta),
            "lambda": float(lam),
        },
    }
