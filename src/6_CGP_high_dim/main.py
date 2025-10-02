import os
from typing import Dict

import numpy as np

from algorithms_smooth_highDim import (
    ACGDSolver,
    IPPM_with_counts,
    SwSGSolver,
    adaptive_shifted_bundle_level,
)
from problem_smooth_highDim import build_near_tight_instance


def solve_with_cvxpy(metadata: Dict[str, np.ndarray], box):
    try:
        import cvxpy as cp  # type: ignore
    except ImportError:
        return None

    A0 = metadata["A0"]
    A1 = metadata["A1"]
    log_b0 = np.log(metadata["b0"])
    log_b1 = np.log(metadata["b1"])

    d = A0.shape[1]
    u = cp.Variable(d)
    l, u_box = box

    objective = cp.Minimize(cp.sum(cp.exp(log_b0 + A0 @ u)))
    constraints = [
        cp.sum(cp.exp(log_b1 + A1 @ u)) <= 1.0,
        u >= np.log(l),
        u <= np.log(u_box),
    ]

    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.SCS, verbose=False)

    if problem.status not in ("optimal", "optimal_inaccurate"):
        return None

    u_opt = u.value
    x_opt = np.exp(u_opt)
    F1_val = float(np.sum(np.exp(log_b1 + A1 @ u_opt)))

    return {
        "status": problem.status,
        "objective": float(objective.value),
        "violation": F1_val - 1.0,
        "x_opt": x_opt,
    }


def run_experiment_and_plot(
    outdir: str = "../figs",
    d: int = 100,
    N: int = 10,
    grad_budget_per_outer: int = 600,
    tau: float = 1e-3,
    rho_hat: float = 1.0,
    box=(0.5, 2.0),
    seed: int = 21,
    initial_points: list[np.ndarray] | None = None,
    alpha: float = 0.1,
    Tin_swsg_override: int | None = None,
    Tin_acgd_override: int | None = None,
    make_plots: bool = True,
    run_sbl: bool = True,
    sbl_options: Dict[str, object] | None = None,
) -> Dict[str, Dict[str, object]]:
    os.makedirs(outdir, exist_ok=True)
    problem, metadata = build_near_tight_instance(
        d=d, seed=seed, box=box, tau=tau, rho_hat=rho_hat
    )

    l, u = problem.box
    proj = problem.projection
    x0 = problem.x0.copy()
    eps_in = 1e-4
    alpha = alpha

    if sbl_options is None:
        sbl_options = {}

    sbl_config = {
        "N": int(sbl_options.get("epochs", 5)),
        "T": int(sbl_options.get("inner_iters", 50)),
        "alpha": float(sbl_options.get("alpha", 0.5)),
        "beta": float(sbl_options.get("beta", 0.5)),
        "lambda": float(sbl_options.get("lambda", 1.0)),
        "eta0": float(sbl_options.get("eta0", 0.0)),
        "tau": float(sbl_options.get("tau", tau)),
        "proj_tol": float(sbl_options.get("proj_tol", 1e-6)),
        "proj_max_iters": int(sbl_options.get("proj_max_iters", 200)),
    }

    if Tin_swsg_override is not None and Tin_acgd_override is not None:
        Tin_swsg = int(Tin_swsg_override)
        Tin_acgd = int(Tin_acgd_override)
    else:
        Tin_swsg = grad_budget_per_outer
        Tin_acgd = max(1, int(0.4 * grad_budget_per_outer))
    target_calls: Dict[str, int] = {
        "SwSG": int(N * Tin_swsg),
        "ACGD": int(N * Tin_acgd),
    }
    if run_sbl:
        target_calls["SBL"] = int(sbl_config["N"] * sbl_config["T"] * 2)

    swsg = SwSGSolver(stepsize_scale=0.05)
    acgd = ACGDSolver(rho_hat=problem.rho_hat, box_l=l, box_u=u)

    cvxpy_solution = solve_with_cvxpy(metadata, box)

    if initial_points is None:
        rng = np.random.default_rng(seed + 1)
        if cvxpy_solution and cvxpy_solution.get("x_opt") is not None:
            x_opt = cvxpy_solution["x_opt"]
            noise = lambda scale: np.clip(x_opt + rng.normal(0.0, scale, size=d), l, u)
            initial_points = [
                np.clip(x_opt, l, u),
                noise(0.05),
                noise(0.1),
                noise(0.15),
            ]
        else:
            spread = u - l
            initial_points = [
                problem.x0.copy(),
                np.full(d, l + 0.3 * spread),
                np.full(d, l + 0.6 * spread),
                rng.uniform(l, u, size=d),
            ]

    def constraint_violation(x: np.ndarray) -> float:
        return problem.F1(x) - 1.0

    objective_tol = 1e-1
    violation_tol = 1e-2

    runs: list[Dict[str, object]] = []
    trajectories = None

    for idx, x_init in enumerate(initial_points):
        x_start = np.clip(np.array(x_init, dtype=float), l, u)

        F0_swsg, viol_swsg, calls_swsg = IPPM_with_counts(
            problem.F0,
            problem.dF0,
            problem.F1,
            problem.dF1,
            x_start,
            problem.tau,
            N,
            problem.rho_hat,
            swsg,
            Tin_swsg,
            eps_in,
            proj,
            alpha,
        )
        F0_acgd, viol_acgd, calls_acgd = IPPM_with_counts(
            problem.F0,
            problem.dF0,
            problem.F1,
            problem.dF1,
            x_start,
            problem.tau,
            N,
            problem.rho_hat,
            acgd,
            Tin_acgd,
            eps_in,
            proj,
            alpha,
        )

        sbl_result = None
        if run_sbl:
            sbl_result = adaptive_shifted_bundle_level(
                problem.F0,
                problem.dF0,
                constraint_violation,
                problem.dF1,
                x_start,
                eta0=sbl_config["eta0"],
                N=sbl_config["N"],
                T=sbl_config["T"],
                tau=sbl_config["tau"],
                alpha=sbl_config["alpha"],
                beta=sbl_config["beta"],
                lam=sbl_config["lambda"],
                box_l=l,
                box_u=u,
                proj_tol=sbl_config["proj_tol"],
                proj_max_iters=sbl_config["proj_max_iters"],
            )

        if cvxpy_solution:
            reference_obj = cvxpy_solution["objective"]
            reference_viol = cvxpy_solution["violation"]
        else:
            reference_obj = None
            reference_viol = None

        objective_gap_swsg = (
            abs(F0_swsg[-1] - reference_obj) if reference_obj is not None else None
        )
        objective_gap_acgd = (
            abs(F0_acgd[-1] - reference_obj) if reference_obj is not None else None
        )
        violation_gap_swsg = (
            abs(viol_swsg[-1] - reference_viol) if reference_viol is not None else None
        )
        violation_gap_acgd = (
            abs(viol_acgd[-1] - reference_viol) if reference_viol is not None else None
        )
        if run_sbl and sbl_result is not None:
            x_sbl_best = np.array(sbl_result["x_best"], dtype=float)
            sbl_f0 = float(problem.F0(x_sbl_best))
            sbl_viol = float(constraint_violation(x_sbl_best))
            sbl_calls = int(sbl_result["grad_calls"][-1])
            objective_gap_sbl = (
                abs(sbl_f0 - reference_obj) if reference_obj is not None else None
            )
            violation_gap_sbl = (
                abs(sbl_viol - reference_viol) if reference_viol is not None else None
            )
            sbl_summary = {
                "final_F0": sbl_f0,
                "final_viol": sbl_viol,
                "calls": sbl_calls,
                "penalty": float(sbl_result["best_penalty"]),
                "objective_gap": (
                    None if objective_gap_sbl is None else float(objective_gap_sbl)
                ),
                "violation_gap": (
                    None if violation_gap_sbl is None else float(violation_gap_sbl)
                ),
                "close_to_opt": (
                    None
                    if objective_gap_sbl is None or violation_gap_sbl is None
                    else bool(
                        objective_gap_sbl <= objective_tol
                        and violation_gap_sbl <= violation_tol
                    )
                ),
            }
        else:
            sbl_summary = None

        run_summary = {
            "initial_index": idx,
            "initial_stats": {
                "min": float(x_start.min()),
                "max": float(x_start.max()),
                "mean": float(x_start.mean()),
            },
            "SwSG": {
                "final_F0": float(F0_swsg[-1]),
                "final_viol": float(viol_swsg[-1]),
                "calls": int(calls_swsg[-1]),
                "objective_gap": (
                    None if objective_gap_swsg is None else float(objective_gap_swsg)
                ),
                "violation_gap": (
                    None if violation_gap_swsg is None else float(violation_gap_swsg)
                ),
                "close_to_opt": (
                    None
                    if objective_gap_swsg is None or violation_gap_swsg is None
                    else bool(
                        objective_gap_swsg <= objective_tol
                        and violation_gap_swsg <= violation_tol
                    )
                ),
            },
            "ACGD": {
                "final_F0": float(F0_acgd[-1]),
                "final_viol": float(viol_acgd[-1]),
                "calls": int(calls_acgd[-1]),
                "objective_gap": (
                    None if objective_gap_acgd is None else float(objective_gap_acgd)
                ),
                "violation_gap": (
                    None if violation_gap_acgd is None else float(violation_gap_acgd)
                ),
                "close_to_opt": (
                    None
                    if objective_gap_acgd is None or violation_gap_acgd is None
                    else bool(
                        objective_gap_acgd <= objective_tol
                        and violation_gap_acgd <= violation_tol
                    )
                ),
            },
            "SBL": sbl_summary,
        }
        runs.append(run_summary)

        if trajectories is None:
            trajectories = {
                "SwSG": {
                    "F0": F0_swsg,
                    "viol": viol_swsg,
                    "calls": calls_swsg,
                },
                "ACGD": {
                    "F0": F0_acgd,
                    "viol": viol_acgd,
                    "calls": calls_acgd,
                },
            }
            if run_sbl and sbl_result is not None:
                trajectories["SBL"] = {
                    "F0": np.array(sbl_result["F0"], dtype=float),
                    "viol": np.array(sbl_result["F2"], dtype=float),
                    "penalties": np.array(sbl_result["penalties"], dtype=float),
                    "calls": np.array(sbl_result["grad_calls"][1:], dtype=float),
                }

    obj_path = ""
    viol_path = ""
    if make_plots and trajectories is not None:
        import matplotlib.pyplot as plt

        plt.figure()
        plt.plot(
            trajectories["SwSG"]["calls"], trajectories["SwSG"]["F0"], label="IPPM+SwSG"
        )
        plt.plot(
            trajectories["ACGD"]["calls"], trajectories["ACGD"]["F0"], label="IPPM+ACGD"
        )
        if "SBL" in trajectories:
            plt.plot(
                trajectories["SBL"]["calls"],
                trajectories["SBL"]["F0"],
                label="Ada-LS+SBL",
            )
        plt.xlabel("cumulative gradient calls")
        plt.ylabel("F0(x^(k))")
        plt.title("Outer objective vs gradient calls")
        plt.legend()
        obj_path = os.path.join(outdir, "objective_vs_gradcalls.png")
        plt.savefig(obj_path, bbox_inches="tight")

        plt.figure()
        plt.plot(
            trajectories["SwSG"]["calls"],
            trajectories["SwSG"]["viol"],
            label="IPPM+SwSG",
        )
        plt.plot(
            trajectories["ACGD"]["calls"],
            trajectories["ACGD"]["viol"],
            label="IPPM+ACGD",
        )
        if "SBL" in trajectories:
            plt.plot(
                trajectories["SBL"]["calls"],
                trajectories["SBL"]["viol"],
                label="Ada-LS+SBL",
            )
        plt.xlabel("cumulative gradient calls")
        plt.ylabel("F1(x^(k)) - 1")
        plt.title("Outer constraint violation vs gradient calls")
        plt.legend()
        viol_path = os.path.join(outdir, "violation_vs_gradcalls.png")
        plt.savefig(viol_path, bbox_inches="tight")

    first_run = runs[0]

    tin_dict: Dict[str, object] = {"SwSG": Tin_swsg, "ACGD": Tin_acgd}
    if run_sbl:
        tin_dict["SBL"] = {"outer": sbl_config["N"], "inner": sbl_config["T"]}

    return {
        "SwSG": first_run["SwSG"],
        "ACGD": first_run["ACGD"],
        "SBL": first_run["SBL"] if run_sbl else None,
        "CVXPY": cvxpy_solution,
        "runs": runs,
        "target_calls": target_calls,
        "Tin": tin_dict,
        "plots": {"objective": obj_path, "violation": viol_path},
        "tolerances": {"objective": objective_tol, "violation": violation_tol},
        "params": {
            "rho_hat": rho_hat,
            "alpha": alpha,
            "Tin_swsg": Tin_swsg,
            "Tin_acgd": Tin_acgd,
            "N": N,
            "shift_scale": acgd.shift_scale,
            "SBL": sbl_config if run_sbl else None,
        },
    }


def grid_search_parameters(
    outdir: str,
    d: int,
    N: int,
    tau: float,
    box,
    seed: int,
    alpha_values: list[float],
    rho_hat_values: list[float],
    Tin_swsg_values: list[int],
    Tin_acgd_values: list[int],
    initial_points: list[np.ndarray] | None = None,
    run_sbl: bool = True,
    sbl_options: Dict[str, object] | None = None,
) -> list[Dict[str, object]]:
    results = []
    for rho_hat in rho_hat_values:
        for alpha in alpha_values:
            for Tin_swsg in Tin_swsg_values:
                for Tin_acgd in Tin_acgd_values:
                    stats = run_experiment_and_plot(
                        outdir=outdir,
                        d=d,
                        N=N,
                        grad_budget_per_outer=Tin_swsg,
                        tau=tau,
                        rho_hat=rho_hat,
                        box=box,
                        seed=seed,
                        initial_points=initial_points,
                        alpha=alpha,
                        Tin_swsg_override=Tin_swsg,
                        Tin_acgd_override=Tin_acgd,
                        make_plots=False,
                        run_sbl=run_sbl,
                        sbl_options=sbl_options,
                    )

                    def _valid(values):
                        return [
                            v
                            for v in values
                            if isinstance(v, (int, float)) and not np.isnan(v)
                        ]

                    acgd_gaps = _valid(
                        run["ACGD"]["objective_gap"] for run in stats["runs"]
                    )
                    acgd_viol_gaps = _valid(
                        run["ACGD"]["violation_gap"] for run in stats["runs"]
                    )
                    swsg_gaps = _valid(
                        run["SwSG"]["objective_gap"] for run in stats["runs"]
                    )
                    swsg_viol_gaps = _valid(
                        run["SwSG"]["violation_gap"] for run in stats["runs"]
                    )
                    if run_sbl:
                        sbl_entries = [
                            run["SBL"] for run in stats["runs"] if run.get("SBL")
                        ]
                        sbl_gaps = _valid(
                            entry["objective_gap"] for entry in sbl_entries
                        )
                        sbl_viol_gaps = _valid(
                            entry["violation_gap"] for entry in sbl_entries
                        )
                    else:
                        sbl_gaps = []
                        sbl_viol_gaps = []

                    def _aggregate(values):
                        if values:
                            return float(max(values)), float(np.mean(values))
                        return float("inf"), float("inf")

                    acgd_max_gap, acgd_mean_gap = _aggregate(acgd_gaps)
                    acgd_max_viol, acgd_mean_viol = _aggregate(acgd_viol_gaps)
                    swsg_max_gap, swsg_mean_gap = _aggregate(swsg_gaps)
                    swsg_max_viol, swsg_mean_viol = _aggregate(swsg_viol_gaps)
                    sbl_max_gap, sbl_mean_gap = _aggregate(sbl_gaps)
                    sbl_max_viol, sbl_mean_viol = _aggregate(sbl_viol_gaps)

                    aggregate = {
                        "ACGD_max_obj_gap": acgd_max_gap,
                        "ACGD_mean_obj_gap": acgd_mean_gap,
                        "ACGD_max_viol_gap": acgd_max_viol,
                        "ACGD_mean_viol_gap": acgd_mean_viol,
                        "SwSG_max_obj_gap": swsg_max_gap,
                        "SwSG_mean_obj_gap": swsg_mean_gap,
                        "SwSG_max_viol_gap": swsg_max_viol,
                        "SwSG_mean_viol_gap": swsg_mean_viol,
                    }
                    if run_sbl:
                        aggregate.update(
                            {
                                "SBL_max_obj_gap": sbl_max_gap,
                                "SBL_mean_obj_gap": sbl_mean_gap,
                                "SBL_max_viol_gap": sbl_max_viol,
                                "SBL_mean_viol_gap": sbl_mean_viol,
                            }
                        )

                    results.append(
                        {
                            "params": stats["params"],
                            "aggregate": aggregate,
                            "runs": stats["runs"],
                        }
                    )
    results.sort(
        key=lambda r: (
            r["aggregate"]["ACGD_max_obj_gap"],
            r["aggregate"]["ACGD_max_viol_gap"],
            r["aggregate"]["ACGD_mean_obj_gap"],
        )
    )
    return results


if __name__ == "__main__":

    alpha_values = [0.1, 0.2]
    rho_hat_values = [0.01, 0.02, 0.05]
    Tin_swsg_values = [200, 400]
    Tin_acgd_values = [80, 120]

    base_problem, base_metadata = build_near_tight_instance(
        d=100, seed=21, box=(0.5, 2.0), tau=1e-3, rho_hat=min(rho_hat_values)
    )
    base_solution = solve_with_cvxpy(base_metadata, base_problem.box)
    rng = np.random.default_rng(21 + 123)
    l, u = base_problem.box
    spread = u - l
    initial_points = [
        np.clip(base_problem.x0.copy(), l, u),
        np.full(base_problem.x0.shape, l + 0.3 * spread),
        np.full(base_problem.x0.shape, l + 0.6 * spread),
        rng.uniform(l, u, size=base_problem.x0.shape),
        rng.uniform(l, u, size=base_problem.x0.shape),
    ]
    if base_solution and base_solution.get("x_opt") is not None:
        x_opt = base_solution["x_opt"]
        initial_points.append(np.clip(x_opt, l, u))
        initial_points.append(np.clip(x_opt * 1.1, l, u))

    tuning_results = grid_search_parameters(
        outdir="../figs",
        d=100,
        N=15,
        tau=1e-3,
        box=(0.5, 2.0),
        seed=21,
        alpha_values=alpha_values,
        rho_hat_values=rho_hat_values,
        Tin_swsg_values=Tin_swsg_values,
        Tin_acgd_values=Tin_acgd_values,
        initial_points=initial_points,
    )

    top_k = tuning_results[:3]
    print("Top configurations (sorted by ACGD max objective gap):")
    for idx, cfg in enumerate(top_k, start=1):
        print(f"{idx}. params={cfg['params']}, aggregate={cfg['aggregate']}")

    if tuning_results:
        best = tuning_results[0]
        print("\nPer-run stats for best configuration:")
        for run in best["runs"]:
            print(run)

        # Generate plots for best configuration
        best_params = best["params"]
        final_stats = run_experiment_and_plot(
            outdir="../figs",
            d=100,
            N=15,
            grad_budget_per_outer=best_params["Tin_swsg"],
            tau=1e-3,
            rho_hat=best_params["rho_hat"],
            box=(0.5, 2.0),
            seed=21,
            initial_points=initial_points,
            alpha=best_params["alpha"],
            Tin_swsg_override=best_params["Tin_swsg"],
            Tin_acgd_override=best_params["Tin_acgd"],
            make_plots=True,
        )
        print("\nFinal stats with plots for best configuration:")
        print(final_stats)
