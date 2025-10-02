import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from src.base import OptimizationResult, ProblemAbstract
from src.colors_setting import *

if use_latex:
    font_size = 26
    plt.rcParams.update(
        {
            "text.usetex": use_latex,
            "font.size": font_size,
            "font.family": "serif",
            "axes.labelsize": font_size,
            "xtick.labelsize": font_size,
            "ytick.labelsize": font_size,
            "legend.fontsize": font_size,
        }
    )


def minimal_example(
    problem: ProblemAbstract,
    dom_1: np.ndarray,
    dom_2: np.ndarray,
    n_points: int,
    trafo: bool = False,
    path: Path = None,
    title: str = "",
    show_colorbar: float = True,
    show_feasible_set: bool = True,
    ncol_legend: int = 3,
) -> None:

    optimum = problem.optimum_XX
    optimum_global = problem.optimum_global_XX
    if trafo:
        optimum_global = problem.c_func(optimum_global)
        optimum = problem.c_func(optimum)

    x1 = np.linspace(dom_1[0], dom_1[1], n_points)
    x2 = np.linspace(dom_2[0], dom_2[1], n_points)
    X1, X2 = np.meshgrid(x1, x2)
    if trafo:
        eval_F1 = np.array(
            [
                [problem.F1_UU([x1, x2]) for x1, x2 in zip(row_x1, row_x2)]
                for row_x1, row_x2 in zip(X1, X2)
            ]
        )
        eval_F2 = np.array(
            [
                [problem.F2_UU([x1, x2]) for x1, x2 in zip(row_x1, row_x2)]
                for row_x1, row_x2 in zip(X1, X2)
            ]
        )
    else:
        eval_F1 = np.array(
            [
                [problem.F1([x1, x2]) for x1, x2 in zip(row_x1, row_x2)]
                for row_x1, row_x2 in zip(X1, X2)
            ]
        )
        eval_F2 = np.array(
            [
                [problem.F2([x1, x2]) for x1, x2 in zip(row_x1, row_x2)]
                for row_x1, row_x2 in zip(X1, X2)
            ]
        )

    # width x height
    plt.figure(figsize=(10 if not trafo else 12, 8))

    contourf = plt.contourf(
        X1,
        X2,
        eval_F1,
        levels=problem.levels_colorbar,
        # levels=200,
        # cmap="turbo",
        cmap=cmap_levelSets,
        # vmin=problem.vmin,
        # vmax=problem.vmax,
    )
    contour = plt.contour(
        X1,
        X2,
        eval_F1,
        # levels=problem.levels_colorbar,
        levels=20,
        # colors="gray",
        linewidths=0.3,
        linestyles="--",
        colors="k",
        # linewidths=1.0,
        # cmap="RdBu_r",  # "turbo"
        # vmin=problem.vmin,
        # vmax=problem.vmax,
    )
    if show_colorbar:
        plt.colorbar(contourf, ticks=problem.ticks_colorbar)

    plt.clim(problem.vmin, problem.vmax)
    if show_feasible_set:
        plt.plot(
            [],
            [],
            color=col_feas,
            linestyle="-",
            label=(r"$\{F_2 \leq 0\}$" if not trafo else r"$\{H_2 \leq 0\}$"),
        )
        # feasible set
        plt.contour(
            X1,
            X2,
            eval_F2 <= 0,
            levels=[0],
            linewidths=2,
            colors=[col_feas],
            # alpha=0.5,
        )
        plt.contourf(
            X1,
            X2,
            eval_F2,
            levels=[eval_F2.min(), 0],
            colors=[col_feas],
            alpha=alpha_col_feas,
        )
    plt.title(title)
    if trafo:
        plt.xlabel(axes_label_U[0])
        plt.ylabel(axes_label_U[1])
    else:
        plt.xlabel(axes_label_X[0])
        plt.ylabel(axes_label_X[1])
    plt.scatter(
        *optimum,
        marker=marker_opt,
        # edgecolor="white",  # fill color
        # facecolor="black",  # border color
        color=col_opt,
        s=scatter_marker_size,
        label=r"$x^*$" if not trafo else r"$u^{*}$",
        zorder=5,
    )
    plt.scatter(
        *optimum_global,
        marker=marker_opt,
        facecolors=facecolors_opt_global,
        color=col_opt,
        s=scatter_marker_size,
        label=r"$x^*_\mathrm{uncon}$" if not trafo else r"$u^{*}_\mathrm{uncon}$",
        zorder=5,
    )
    if problem.poi:
        for idx in range(len(problem.poi)):
            poi_point = problem.poi[idx]
            if not trafo:
                poi_label = problem.poi_label_XX[idx]
            else:
                poi_label = problem.poi_label_UU[idx]
            poi_marker = problem.poi_marker[idx]

            _poi_point = poi_point if not trafo else problem.c_func(poi_point)
            plt.scatter(
                *_poi_point,
                marker=poi_marker,
                # facecolors=facecolors_opt_global,
                color=col_opt,
                s=scatter_marker_size,
                label=poi_label,
                zorder=5,
            )

    # plt.legend(loc=legend_loc, framealpha=alpha_transparency_legend)
    plt.legend(
        loc="lower center",
        # bbox_to_anchor=(0.5 if trafo else 0.45, -0.25),
        bbox_to_anchor=(0.5, -0.3 if show_feasible_set else -0.3),
        ncol=ncol_legend,
        framealpha=alpha_transparency_legend,
    )
    # plt.legend(handles=legend_elements)
    plt.plot()
    if path is not None:
        for ext in [".pdf", ".png"]:
            plt.savefig(path.with_suffix(ext), bbox_inches="tight")
    plt.show()


def trace_plot(
    problem: ProblemAbstract,
    dom_1: np.ndarray,
    dom_2: np.ndarray,
    n_points: int,
    results: list[OptimizationResult],
    trafo: bool = False,
    path: Path = None,
    title: str = "",
    show_colorbar: float = True,
    show_init: bool = True,
    show_optima: bool = True,
    show_last_iterate: bool = True,
    legend_loc: str = "best",
    num_levels_colorbar: int = 50,
) -> None:

    optimum = problem.optimum_XX
    optimum_global = problem.optimum_global_XX
    if trafo:
        _results = []
        # transform all iterates
        for _res in results:
            _results.append(
                OptimizationResult(
                    x_final=problem.c_func(_res.x_final),
                    iter_history=_res.iter_history,
                    history=np.array([problem.c_func(x) for x in _res.history]),
                    label=_res.label,
                    marker=_res.marker,
                    linestyle=_res.linestyle,
                    color=_res.color,
                    _skip_iterates=_res._skip_iterates,
                )
            )
        # transform optima
        optimum = problem.c_func(optimum)
        optimum_global = problem.c_func(optimum_global)
    else:
        _results = results

    x1 = np.linspace(dom_1[0], dom_1[1], n_points)
    x2 = np.linspace(dom_2[0], dom_2[1], n_points)
    X1, X2 = np.meshgrid(x1, x2)
    if trafo:
        eval_F1 = np.array(
            [
                [problem.F1_UU([x1, x2]) for x1, x2 in zip(row_x1, row_x2)]
                for row_x1, row_x2 in zip(X1, X2)
            ]
        )
        eval_F2 = np.array(
            [
                [problem.F2_UU([x1, x2]) for x1, x2 in zip(row_x1, row_x2)]
                for row_x1, row_x2 in zip(X1, X2)
            ]
        )
    else:
        eval_F1 = np.array(
            [
                [problem.F1([x1, x2]) for x1, x2 in zip(row_x1, row_x2)]
                for row_x1, row_x2 in zip(X1, X2)
            ]
        )
        eval_F2 = np.array(
            [
                [problem.F2([x1, x2]) for x1, x2 in zip(row_x1, row_x2)]
                for row_x1, row_x2 in zip(X1, X2)
            ]
        )

    # width X height
    plt.figure(figsize=(10 if not trafo else 12, 8))

    contourf = plt.contourf(
        X1,
        X2,
        eval_F1,
        levels=problem.levels_colorbar,
        # levels=200,
        # cmap="turbo",
        cmap=cmap_levelSets,
        # vmin=problem.vmin,
        # vmax=problem.vmax,
    )
    contour = plt.contour(
        X1,
        X2,
        eval_F1,
        # levels=problem.levels_colorbar,
        levels=20,
        # colors="gray",
        linewidths=0.3,
        linestyles="--",
        colors="k",
        # linewidths=1.0,
        # cmap="RdBu_r",  # "turbo"
        # vmin=problem.vmin,
        # vmax=problem.vmax,
    )

    if show_colorbar:
        plt.colorbar(contourf, ticks=problem.ticks_colorbar)

    # plt.clim(vmin, vmax)
    plt.clim(problem.vmin, problem.vmax)
    plt.plot(
        [],
        [],
        color=col_feas,
        linestyle="-",
        label=(r"$\{F_2 \leq 0\}$" if not trafo else r"$\{H_2 \leq 0\}$"),
    )
    # contour plots for the objective and the feasible region
    plt.contour(
        X1,
        X2,
        eval_F2 <= 0,
        levels=[0],
        linewidths=2,
        colors=[col_feas],
        # alpha=0.5,
    )
    plt.contourf(
        X1,
        X2,
        eval_F2,
        levels=[eval_F2.min(), 0],
        colors=[col_feas],
        alpha=alpha_col_feas,
    )
    for res in _results:
        plt.plot(
            res.skipped_history[:, 0],
            res.skipped_history[:, 1],
            marker=res.marker,
            linestyle=res.linestyle,
            markersize=marker_size,
            color=res.color,
            label=res.label,
        )
        if show_last_iterate:
            plt.scatter(
                res.history[-1, 0],
                res.history[-1, 1],
                marker=marker_last_iterate,
                color=res.color,
                s=scatter_marker_size,
                label=(
                    rf"$x^{{(N)}}_{{\mathrm{{{res.label}}}}}$"
                    if not trafo
                    else rf"$u^{{(N)}}_{{\mathrm{{{res.label}}}}} = c(x^{{(N)}}_{{\mathrm{{{res.label}}}}})$"
                ),
                zorder=5,
            )

    if show_init:
        plt.scatter(
            _results[0].history[0, 0],
            _results[0].history[0, 1],
            marker=marker_start,
            color=col_opt,
            s=scatter_marker_size,
            label=r"$x^{(0)}$" if not trafo else r"$u^{(0)} = c(x^{(0)})$",
            zorder=5,
        )

    if show_optima:
        plt.scatter(
            *optimum,
            marker=marker_opt,
            color=col_opt,
            s=scatter_marker_size,
            label=r"$x^*$" if not trafo else r"$u^{*}$",
            zorder=5,
        )
        plt.scatter(
            *optimum_global,
            marker=marker_opt,
            facecolors=facecolors_opt_global,
            color=col_opt,
            label=r"$x^*_\mathrm{uncon}$" if not trafo else r"$u^{*}_\mathrm{uncon}$",
            s=scatter_marker_size,
            zorder=5,
        )
    plt.title(title)
    if trafo:
        plt.xlabel(axes_label_U[0])
        plt.ylabel(axes_label_U[1])
    else:
        plt.xlabel(axes_label_X[0])
        plt.ylabel(axes_label_X[1])
    # plt.legend(
    #     loc="lower center",
    #     # bbox_to_anchor=(0.5 if trafo else 0.45, -0.35),
    #     bbox_to_anchor=(0.5, -0.4),
    #     ncol=3,
    #     framealpha=alpha_transparency_legend,
    # )
    plt.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=3,
        framealpha=alpha_transparency_legend,
        borderaxespad=0.0,
    )
    plt.subplots_adjust(bottom=0.3)

    # plt.tight_layout()
    plt.plot()
    if path is not None:
        for ext in [".pdf", ".png"]:
            plt.savefig(path.with_suffix(ext), bbox_inches="tight")
    plt.show()


def func_value_const_viol_shared(
    problem: ProblemAbstract,
    results: list[OptimizationResult],
    constraint_tol: float,
    path: Path = None,
    x_axis_log_scale: bool = True,
) -> None:
    # Create subplots with a shared x-axis
    # width X height
    # fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    for _res in results:
        # F1: function value
        axes[0].plot(
            _res.skipped_iter_history,
            [problem.F1(x) for x in _res.skipped_history],
            linestyle=_res.linestyle,
            label=_res.label,
            marker=_res.marker,
            color=_res.color,
        )
        # F2: constraint violation
        axes[1].plot(
            _res.skipped_iter_history,
            [problem.F2(x) for x in _res.skipped_history],
            linestyle=_res.linestyle,
            label=_res.label,
            marker=_res.marker,
            color=_res.color,
        )

    # axes[0].set_title("Function Value")
    axes[0].set_ylabel(r"$F_1(x^{(k)})$")
    # axes[0].legend()

    # axes[1].axhline(0, color="green", linestyle="--", label="Constraint")
    axes[1].axhline(
        constraint_tol,
        color=col_feas_tol,
        linestyle=linestyle_feas,
        label="Tolerance",
    )
    # axes[1].set_title("Constraint Violation")
    axes[1].set_xlabel(r"Oracle Calls")
    axes[1].set_ylabel(r"$F_2(x^{(k)})$")
    # axes[1].legend()

    # shared legend
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=5,
        bbox_to_anchor=(0.5, -0.05),
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if x_axis_log_scale:
        axes[1].set_xscale("log")

    # plt.suptitle("Non-smooth Constrained Non-Linear Least Squares")
    # plt.tight_layout()
    # Save the plot if a path is provided
    if path is not None:
        for ext in [".pdf", ".png"]:
            plt.savefig(path.with_suffix(ext), bbox_inches="tight")
    plt.show()
