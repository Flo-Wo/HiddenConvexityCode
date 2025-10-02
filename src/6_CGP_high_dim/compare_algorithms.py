from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Tuple

import sys

import matplotlib.pyplot as plt
import numpy as np

from algorithms_smooth_highDim import (
    ACGDSolver,
    IPPM_with_counts,
    SwSGSolver,
    adaptive_shifted_bundle_level,
    shifted_bundle_level,
)
from main import solve_with_cvxpy
from problem_smooth_highDim import build_near_tight_instance

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.colors_setting import (
    col_PPM_SwSG,
    col_PPM_penalty,
    eth_bronze,
    eth_green,
    use_latex,
)

# Apply global plotting style harmonised with other project figures.
plt.rcParams.update(
    {
        "text.usetex": use_latex,
        "font.family": "serif",
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "figure.figsize": (5.2, 3.4),
        "lines.linewidth": 2.0,
        "savefig.dpi": 300,
        "axes.edgecolor": "black",
        "axes.linewidth": 1.0,
    }
)


def _prepare_problem(
    d: int,
    seed: int,
    box: Tuple[float, float],
    tau: float,
    rho_hat: float,
) -> Tuple:
    problem, metadata = build_near_tight_instance(
        d=d, seed=seed, box=box, tau=tau, rho_hat=rho_hat
    )
    cvx_solution = solve_with_cvxpy(metadata, problem.box)
    if cvx_solution is None or cvx_solution.get("objective") is None:
        raise RuntimeError("CVXPY solution is required to estimate F1*.")
    return problem, cvx_solution


def compare_algorithms_equal_budget(
    outdir: str = "plots_compare",
    d: int = 100,
    N_outer: int = 10,
    per_stage_budget: int = 60,
    tau: float = 1e-3,
    rho_hat: float = 0.02,
    box: Tuple[float, float] = (0.5, 2.0),
    seed: int = 21,
    alpha_swsg: float = 0.1,
    alpha_acgd: float = 0.1,
    acgd_shift_scale: float = 0.0,
    alpha_star: float = 0.3,
    alpha_sbl: float = 0.3,
    beta_sbl: float = 0.5,
    lam_penalty: float = 0.25,
    eps_in: float = 1e-4,
) -> Dict[str, Dict[str, float]]:
    """
    Compare IPPM+SwSG, IPPM+ACGD, S-StarBL, and S-BL+AdaLS under an equal gradient-call budget.
    """
    os.makedirs(outdir, exist_ok=True)
    if N_outer % 2 != 0:
        raise ValueError("N_outer must be even to keep all budgets integral.")

    problem, cvx_solution = _prepare_problem(d, seed, box, tau, rho_hat)

    F1_star = float(cvx_solution["objective"])
    l, u = problem.box
    proj = problem.projection
    x0 = np.clip(problem.x0.copy(), l, u)

    # All methods share the same total gradient-call budget.
    # Choose per-stage budget M so that total budget B = 2 * N_outer * M.
    M = max(1, per_stage_budget)
    Tin_acgd = M
    total_grad_budget = N_outer + 2 * N_outer * Tin_acgd

    Tin_swsg = total_grad_budget // N_outer
    T_star = total_grad_budget // 2
    N_sbl = max(1, N_outer // 2)
    Tin_sbl = total_grad_budget // (2 * N_sbl)

    constraint_violation = lambda x: problem.F1(x) - 1.0

    # IPPM + SwSG
    swsg_solver = SwSGSolver(stepsize_scale=0.05)
    F0_swsg, viol_swsg, calls_swsg = IPPM_with_counts(
        problem.F0,
        problem.dF0,
        problem.F1,
        problem.dF1,
        x0,
        problem.tau,
        N_outer,
        problem.rho_hat,
        swsg_solver,
        Tin_swsg,
        eps_in,
        proj,
        alpha_swsg,
    )

    # IPPM + ACGD
    acgd_solver = ACGDSolver(
        rho_hat=problem.rho_hat,
        box_l=l,
        box_u=u,
        shift_scale=acgd_shift_scale,
    )
    F0_acgd, viol_acgd, calls_acgd = IPPM_with_counts(
        problem.F0,
        problem.dF0,
        problem.F1,
        problem.dF1,
        x0,
        problem.tau,
        N_outer,
        problem.rho_hat,
        acgd_solver,
        Tin_acgd,
        eps_in,
        proj,
        alpha_acgd,
    )

    # S-StarBL (beta = 1, eta = F1*)
    x_star, stats_star = shifted_bundle_level(
        problem.F0,
        problem.dF0,
        constraint_violation,
        problem.dF1,
        x0,
        eta=F1_star,
        T=T_star,
        tau=problem.tau,
        alpha=alpha_star,
        beta=1.0,
        lam=lam_penalty,
        box_l=l,
        box_u=u,
    )
    F0_star = np.array(stats_star["F0"], dtype=float)
    viol_star = np.array(stats_star["F2"], dtype=float)
    calls_star = np.arange(len(F0_star), dtype=float) * 2.0

    # S-BL + AdaLS
    eta0 = 0.5 * F1_star
    ada_stats = adaptive_shifted_bundle_level(
        problem.F0,
        problem.dF0,
        constraint_violation,
        problem.dF1,
        x0,
        eta0=eta0,
        N=N_sbl,
        T=Tin_sbl,
        tau=problem.tau,
        alpha=alpha_sbl,
        beta=beta_sbl,
        lam=lam_penalty,
        box_l=l,
        box_u=u,
    )
    F0_sbl = np.concatenate(([problem.F0(x0)], np.array(ada_stats["F0"], dtype=float)))
    viol_sbl = np.concatenate(
        ([constraint_violation(x0)], np.array(ada_stats["F2"], dtype=float))
    )
    calls_sbl = np.array(ada_stats["grad_calls"], dtype=float)

    # Sanity check: ensure equal budgets.
    final_calls = {
        "IPPM+SwSG": int(calls_swsg[-1]),
        "IPPM+ACGD": int(calls_acgd[-1]),
        "S-StarBL": int(calls_star[-1]),
        "S-BL+AdaLS": int(calls_sbl[-1]),
    }
    target_budget = int(total_grad_budget)
    for name, count in final_calls.items():
        if count != target_budget:
            raise RuntimeError(
                f"{name} used {count} gradient calls (target {target_budget}). "
                "Adjust per_stage_budget or N_outer for exact matching."
            )

    labels = {
        "IPPM+SwSG": r"IPPM+SwSG",
        "IPPM+ACGD": r"IPPM+ACGD",
        "S-StarBL": r"S-StarBL",
        "S-BL+AdaLS": r"S-BL+AdaLS",
    }
    colors = {
        "IPPM+SwSG": col_PPM_SwSG,
        "IPPM+ACGD": eth_green,
        "S-StarBL": eth_bronze,
        "S-BL+AdaLS": col_PPM_penalty,
    }
    linestyles = {
        "IPPM+SwSG": "-",
        "IPPM+ACGD": "--",
        "S-StarBL": ":",
        "S-BL+AdaLS": "-.",
    }

    # Objective plot (log-scale y-axis)
    fig_obj, ax_obj = plt.subplots()
    ax_obj.plot(
        calls_swsg,
        F0_swsg,
        label=labels["IPPM+SwSG"],
        color=colors["IPPM+SwSG"],
        linestyle=linestyles["IPPM+SwSG"],
    )
    ax_obj.plot(
        calls_acgd,
        F0_acgd,
        label=labels["IPPM+ACGD"],
        color=colors["IPPM+ACGD"],
        linestyle=linestyles["IPPM+ACGD"],
    )
    ax_obj.plot(
        calls_star,
        F0_star,
        label=labels["S-StarBL"],
        color=colors["S-StarBL"],
        linestyle=linestyles["S-StarBL"],
    )
    ax_obj.plot(
        calls_sbl,
        F0_sbl,
        label=labels["S-BL+AdaLS"],
        color=colors["S-BL+AdaLS"],
        linestyle=linestyles["S-BL+AdaLS"],
    )
    ax_obj.set_xlabel(r"Oracle Calls")
    ax_obj.set_ylabel(r"$F_1(x)$")
    ax_obj.set_yscale("log")
    ax_obj.set_title("")
    ax_obj.legend(loc="best", frameon=False)
    fig_obj.tight_layout()
    obj_path = os.path.join(outdir, "compare_objective.png")
    fig_obj.savefig(obj_path, bbox_inches="tight")
    plt.close(fig_obj)

    # Constraint violation plot
    fig_viol, ax_viol = plt.subplots()
    ax_viol.plot(
        calls_swsg,
        viol_swsg,
        label=labels["IPPM+SwSG"],
        color=colors["IPPM+SwSG"],
        linestyle=linestyles["IPPM+SwSG"],
    )
    ax_viol.plot(
        calls_acgd,
        viol_acgd,
        label=labels["IPPM+ACGD"],
        color=colors["IPPM+ACGD"],
        linestyle=linestyles["IPPM+ACGD"],
    )
    ax_viol.plot(
        calls_star,
        viol_star,
        label=labels["S-StarBL"],
        color=colors["S-StarBL"],
        linestyle=linestyles["S-StarBL"],
    )
    ax_viol.plot(
        calls_sbl,
        viol_sbl,
        label=labels["S-BL+AdaLS"],
        color=colors["S-BL+AdaLS"],
        linestyle=linestyles["S-BL+AdaLS"],
    )
    ax_viol.set_xlabel(r"Oracle Calls")
    ax_viol.set_ylabel(r"$F_2(x) - 1$")
    ax_viol.set_title("")
    ax_viol.axhline(0.0, color="black", linewidth=1.0, alpha=0.3, label="Tolerance")
    ax_viol.legend(loc="best", frameon=False)
    fig_viol.tight_layout()
    viol_path = os.path.join(outdir, "compare_violation.png")
    fig_viol.savefig(viol_path, bbox_inches="tight")
    plt.close(fig_viol)

    return {
        "budget": target_budget,
        "objective_plot": obj_path,
        "violation_plot": viol_path,
        "final_values": {
            "IPPM+SwSG": (float(F0_swsg[-1]), float(viol_swsg[-1])),
            "IPPM+ACGD": (float(F0_acgd[-1]), float(viol_acgd[-1])),
            "S-StarBL": (float(F0_star[-1]), float(viol_star[-1])),
            "S-BL+AdaLS": (float(F0_sbl[-1]), float(viol_sbl[-1])),
        },
    }


if __name__ == "__main__":
    stats = compare_algorithms_equal_budget()
    print("Comparison complete.")
    print(f"Total gradient budget: {stats['budget']}")
    print(f"Objective plot: {stats['objective_plot']}")
    print(f"Violation plot: {stats['violation_plot']}")
    print("Final objective/violation pairs:")
    for name, (obj, viol) in stats["final_values"].items():
        print(f"  {name}: F0={obj:.6f}, F1-1={viol:.3e}")
