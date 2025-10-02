from pathlib import Path
import numpy as np

from problem_smooth import ProblemSmooth
from src.parser import Parser
from src.algorithms import HC_PMM_osqp, IPPM_ACGD

# plotting
from src.create_plots import minimal_example, trace_plot, func_value_const_viol_shared

global_path = Path(__file__).parent.parent / "figs"


def min_example_levelSet_only(show_feasible_set: bool = True):
    constraint_file_name = "" if show_feasible_set else "_unconstrained"

    path_to_folder = global_path
    n_points = 1000
    x_domain = np.array([0.4, 3])
    problem = ProblemSmooth(0.1)

    ## ===== MIN EXAMPLE =====
    minimal_example(
        problem=problem,
        dom_1=x_domain,
        dom_2=x_domain,
        n_points=n_points,
        # title=r"Constrained Geometric Programming, $\mathcal{X}$-domain",
        title=r"$\mathcal{X}$-domain",
        show_colorbar=False,
        path=path_to_folder
        / "smooth_geometric_programming_X{}".format(constraint_file_name),
        show_feasible_set=show_feasible_set,
    )
    minimal_example(
        problem=problem,
        dom_1=np.log(x_domain),
        dom_2=np.log(x_domain),
        n_points=n_points,
        trafo=True,
        # title=r"Constrained Geometric Programming, $\mathcal{U}$-domain",
        title=r"$\mathcal{U}$-domain",
        show_colorbar=True,
        path=path_to_folder
        / "smooth_geometric_programming_U{}".format(constraint_file_name),
        show_feasible_set=show_feasible_set,
    )
    return


def min_example_iterates():

    path_to_folder = global_path
    # Parameters
    step_size = 0.02
    epsilon_target = 1e-2
    const_step_size = False
    # Initial guess
    # x0 = np.array([0.4, 0.4])
    problem = ProblemSmooth(epsilon=epsilon_target)
    x0 = problem.c_inv(np.array([-0.5, 0.5 + (epsilon_target**2 / 2)]))
    # x0 = problem.optimum_XX.copy()
    # x0 *= np.sqrt(1 - epsilon_target**2 / 2)
    print("initial feasibility: ", problem.F2(x0))
    print("tolerance: ", epsilon_target**2 / 2)
    iter_PPM = int(0.18 * 1 / epsilon_target**2)
    max_iter = int(0.18 * 1 / epsilon_target**3)
    n_points = 1000
    # plotting
    x_domain = np.array([0.4, 3])
    y_domain = np.array([0.3, 3])

    print("\nRunning PPM + ACGD")
    res_PPM_ACGD = IPPM_ACGD(
        x0=x0,
        problem=problem,
        max_iter=max_iter,
        iter_PPM=iter_PPM,
    )
    print("Solution using PPM+ACGD: ", res_PPM_ACGD.x_final)
    print("function value: ", problem.F1(res_PPM_ACGD.x_final))
    print("constraint value: ", problem.F2(res_PPM_ACGD.x_final))

    print("\nRunning HC-PMM with KNOWN F_1^*")

    res_PMM_known_Fstar = HC_PMM_osqp(
        x0=x0,
        problem=problem,
        max_iter=int(1e4),
        _max_iter_global=max_iter,
    )
    print("Solution using HC—PMM: ", res_PMM_known_Fstar.x_final)
    print("function value: ", problem.F1(res_PMM_known_Fstar.x_final))

    print("\nRunning HC-PMM with UNKNOWN F_1^*")
    res_PMM_unknown_Fstar = HC_PMM_osqp(
        x0=x0,
        problem=problem,
        max_iter=int(4e4),
        _max_iter_global=max_iter,
        eta=0,
    )
    print("Solution using HC—PMM: ", res_PMM_unknown_Fstar.x_final)
    print("function value: ", problem.F1(res_PMM_unknown_Fstar.x_final))

    print("\n=======")
    print("Solution of the optimum: ", problem.optimum_XX)
    print("Function value optimum: ", problem.F1(problem.optimum_XX))
    print("=======")

    # Figure 5a) iterates in the X-domain
    trace_plot(
        problem=problem,
        dom_1=x_domain,
        dom_2=y_domain,
        n_points=n_points,
        results=[res_PPM_ACGD, res_PMM_known_Fstar, res_PMM_unknown_Fstar],
        title=(
            # "Smooth Constrained Geometric Programming\n"
            # + r"SwSG vs. IPPM vs. IPPPM, $\mathcal{X}$-domain"
            r"$\mathcal{X}$-domain"
        ),
        show_colorbar=False,
        show_last_iterate=False,
        path=path_to_folder / "smooth_trace_plot_X",
        legend_loc="upper right",
        # num_levels_colorbar=100,
    )

    # Figure 5b) iterates in the U-domain
    trace_plot(
        problem=problem,
        dom_1=np.log(x_domain),
        dom_2=np.log(y_domain),
        n_points=n_points,
        results=[res_PPM_ACGD, res_PMM_known_Fstar, res_PMM_unknown_Fstar],
        trafo=True,
        title=(
            # "Smooth Constrained Geometric Programming\n"
            # + r"SwSG vs. IPPM vs. IPPPM, $\mathcal{U}$-domain"
            r"$\mathcal{U}$-domain"
        ),
        show_colorbar=True,
        show_last_iterate=False,
        path=path_to_folder / "smooth_trace_plot_U",
        legend_loc="upper left",
        # num_levels_colorbar=100,
    )

    # Fig 5c): constraint violation and function value plot
    func_value_const_viol_shared(
        problem=problem,
        results=[res_PPM_ACGD, res_PMM_known_Fstar, res_PMM_unknown_Fstar],
        constraint_tol=epsilon_target,
        path=path_to_folder / "smooth_func_value_const_viol",
    )


if __name__ == "__main__":
    parser = Parser()
    args = parser.parse_args()

    if args.run_algos:
        min_example_iterates()
    else:
        min_example_levelSet_only(show_feasible_set=not args.hide_feasible_set)
