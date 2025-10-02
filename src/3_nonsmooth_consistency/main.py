import numpy as np
from problem_nonsmooth_consistency import ProblemNonSmoothConsistency
from src.parser import Parser

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
    problem = ProblemNonSmoothConsistency(0.1)

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
        / "non_smooth_consistency_least_squares_X{}".format(constraint_file_name),
        show_feasible_set=show_feasible_set,
        ncol_legend=4,
    )
    minimal_example(
        problem=problem,
        dom_1=np.array([-1.5, 0.75]),
        dom_2=np.array([-3.5, 0.5]),
        n_points=n_points,
        trafo=True,
        # title=r"Constrained Non-linear Least Squares, $\mathcal{U}$-domain",
        title=r"$\mathcal{U}$-domain",
        show_colorbar=True,
        path=path_to_folder
        / "non_smooth_consistency_least_squares_U{}".format(constraint_file_name),
        show_feasible_set=show_feasible_set,
        ncol_legend=4,
    )
    return


if __name__ == "__main__":
    parser = Parser()
    args = parser.parse_args()

    min_example_levelSet_only(show_feasible_set=not args.hide_feasible_set)
