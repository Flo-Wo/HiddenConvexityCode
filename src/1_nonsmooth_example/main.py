import numpy as np
from problem_nonsmooth import ProblemNonSmooth
from src.parser import Parser
from src.algorithms import IPPM_switching_subgradient

# from trace_plots_nonsmooth import trace_plot
from src.create_plots import minimal_example, trace_plot, func_value_const_viol_shared

# from func_val_nonsmooth import func_value_const_viol_shared
from pathlib import Path

# script to create figurs of the non-smooth examples
global_path = Path(__file__).parent.parent / "figs"


def min_example_levelSet_only(show_feasible_set: bool = True):
    constraint_file_name = "" if show_feasible_set else "_unconstrained"
    path_to_folder = global_path
    n_points = 1000
    problem = ProblemNonSmooth(0.1)

    ## ===== MIN EXAMPLE =====
    minimal_example(
        problem=problem,
        dom_1=np.array([-1, 1.5]),
        dom_2=np.array([-1, 2.5]),
        n_points=n_points,
        # title=r"Constrained Non-linear Least Squares, $\mathcal{X}$-domain",
        title=r"$\mathcal{X}$-domain",
        show_colorbar=False,
        path=path_to_folder
        / "non_smooth_least_squares_X{}".format(constraint_file_name),
        show_feasible_set=show_feasible_set,
    )
    minimal_example(
        problem=problem,
        dom_1=np.array([-1.5, 0.75]),
        dom_2=np.array([-1.5, 0.75]),
        n_points=n_points,
        trafo=True,
        # title=r"Constrained Non-linear Least Squares, $\mathcal{U}$-domain",
        title=r"$\mathcal{U}$-domain",
        show_colorbar=True,
        path=path_to_folder
        / "non_smooth_least_squares_U{}".format(constraint_file_name),
        show_feasible_set=show_feasible_set,
    )
    return


def min_example_iterates():

    path_to_folder = global_path  # / "1_nonsmooth_example/"
    # path_to_folder = (
    #     Path(__file__).parent.parent.parent / "MathProg/figures/1_toy_example/"
    # )
    # Parameters
    step_size = 0.01
    epsilon_target = 1e-2
    const_step_size = False
    # Initial guess
    x0 = np.array([-0.3, 0.2 + epsilon_target**2 / 2])
    iter_PPM = int(1 / epsilon_target**2)
    max_iter = int(1 / epsilon_target**3)
    n_points = 1000

    problem = ProblemNonSmooth(epsilon=epsilon_target)
    print("initial feasibility: ", problem.F2(x0))
    print("tolerance: ", epsilon_target**2 / 2)

    ## ===== SOLUTION =====

    print("\nRunning PPM + Switching Subgradient")
    res_PPM_SwSG = IPPM_switching_subgradient(
        x0=x0,
        step_size=step_size,
        max_iter=max_iter,
        iter_PPM=iter_PPM,
        constraint_viol_tol=epsilon_target,
        const_step_size=const_step_size,
        problem=problem,
    )
    print("Solution using PPM+SwSG: ", res_PPM_SwSG.x_final)
    print("function value: ", problem.F1(res_PPM_SwSG.x_final))

    print("\n=======")
    print("Solution of the optimum: ", problem.optimum_XX)
    print("Function value optimum: ", problem.F1(problem.optimum_XX))
    print("=======")

    # Figure 4a) iterates in the X-domain
    trace_plot(
        problem=problem,
        dom_1=np.array([-0.6, 1.5]),
        dom_2=np.array([-1, 2.5]),
        n_points=n_points,
        results=[res_PPM_SwSG],
        title=(
            # "Non-smooth Constrained Non-Linear Least Squares\n"
            # r"SwSG vs. IPPM vs. IPPPM, $\mathcal{X}$-domain"
            r"$\mathcal{X}$-domain"
        ),
        show_colorbar=False,
        show_last_iterate=False,
        path=path_to_folder / "nonsmooth_lsq_trace_plot_X",
        legend_loc="upper left",
        num_levels_colorbar=100,
    )

    # Figure 4b) iterates in the U-domain
    trace_plot(
        problem=problem,
        dom_1=np.array([-1.5, 0.5]),
        dom_2=np.array([-1.5, 0.5]),
        n_points=n_points,
        results=[res_PPM_SwSG],
        trafo=True,
        title=(
            # "Non-smooth Constrained Non-Linear Least Squares\n"
            # r"SwSG vs. IPPM vs. IPPPM, $\mathcal{U}$-domain"
            r"$\mathcal{U}$-domain"
        ),
        show_colorbar=True,
        show_last_iterate=False,
        path=path_to_folder / "nonsmooth_lsq_trace_plot_U",
        legend_loc="upper right",
        num_levels_colorbar=100,
    )

    # Fig 4c): constraint violation and function value plot

    func_value_const_viol_shared(
        problem=problem,
        results=[res_PPM_SwSG],
        constraint_tol=epsilon_target,
        path=path_to_folder / "nonsmooth_lsq_func_value_const_viol",
    )


if __name__ == "__main__":
    parser = Parser()
    args = parser.parse_args()

    if args.run_algos:
        min_example_iterates()
    else:
        min_example_levelSet_only(show_feasible_set=not args.hide_feasible_set)
