from pathlib import Path
from problem_smooth import ProblemSmooth
import numpy as np
import matplotlib.pyplot as plt
from src.colors_setting import *

"""
Script to visualize the shifting in the linearized constraint, Figure 3b) in the paper.
"""


font_size = font_size_general + 5
ncolumns_label = 2
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

problem = ProblemSmooth(0.1)

global_path = Path(__file__).parent.parent / "figs"
path = global_path / "PMM_constraint_illustration_smooth_problem"


def l_F1(x_t: np.ndarray) -> callable:
    l_F1_x_t = lambda x: problem.F1(x_t) + (problem.partial_F1(x_t)).T @ (x - x_t)
    return l_F1_x_t


def l_F2(x_t: np.ndarray) -> callable:
    l_F2_x_t = lambda x: problem.F2(x_t) + (problem.partial_F2(x_t)).T @ (x - x_t)
    return l_F2_x_t


x_domain = np.array([0.4, 3])
y_domain = np.array([0.4, 1.2])


epsilon_target = problem.epsilon

# x0 = problem.c_inv(np.array([-0.5, 0.5 + (epsilon_target**2 / 2)]))
x_current = np.array([1.13, 0.88])


l_F2_at_xCurrent = l_F2(x_current)

dom_1 = x_domain
dom_2 = y_domain

n_points = 1000
x1 = np.linspace(dom_1[0], dom_1[1], n_points)
x2 = np.linspace(dom_2[0], dom_2[1], n_points)
X1, X2 = np.meshgrid(x1, x2)

plt.figure(figsize=(10, 8))

# evaluate linearized function, F_1 and F_2
eval_l_F2 = np.array(
    [
        [l_F2_at_xCurrent([x1, x2]) for x1, x2 in zip(row_x1, row_x2)]
        for row_x1, row_x2 in zip(X1, X2)
    ]
)
# shifted field
alpha = 0.1
tau = problem.rho * alpha**2 * problem.D_U**2 / (2 * problem.mu_c**2)
c = (1 - alpha) * problem.F2(x_current) + tau

x_t_alpha = problem.c_inv(
    (1 - 5 * alpha) * problem.c_func(x_current)
    + 5 * alpha * problem.c_func(problem.optimum_XX)
)

eval_l_F2_shifted = eval_l_F2 - c
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

# contour plots and level sets
contourf = plt.contourf(
    X1,
    X2,
    eval_F1,
    levels=problem.levels_colorbar,
    cmap=cmap_levelSets,
)
contour = plt.contour(
    X1,
    X2,
    eval_F1,
    levels=20,
    linewidths=0.3,
    linestyles="--",
    colors="k",
)
# feasible set
plt.contour(
    X1,
    X2,
    eval_F2 <= 0,
    levels=[0],
    linewidths=2,
    colors=[col_feas],
)
plt.contourf(
    X1,
    X2,
    eval_F2,
    levels=[eval_F2.min(), 0],
    colors=[col_feas],
    alpha=alpha_col_feas,
)

# fill region where shifted <= 0
contourf_shifted = plt.contourf(
    X1,
    X2,
    eval_l_F2_shifted,
    levels=[-np.inf, 0],
    colors=[dark_orange],
    hatches=["\\"],  # diagonal stripes
    alpha=0.0,
    label=f"Shifted set (c={c})",
)
contourf_shifted._hatch_color = dark_orange

# plot boundary of shifted set
plt.contour(
    X1,
    X2,
    eval_l_F2_shifted,
    levels=[0],
    colors=dark_orange,
    linewidths=2,
    linestyles="-",
)


contourf_non_shifted = plt.contourf(
    X1,
    X2,
    eval_l_F2,
    levels=[-np.inf, 0],
    colors=[eth_purple],
    hatches=["/"],
    alpha=0.0,
    label="Original set",
)
contourf_non_shifted._hatch_color = eth_purple

plt.contour(
    X1,
    X2,
    eval_l_F2,
    levels=[0],
    colors=eth_purple,
    linewidths=2,
    linestyles="--",
)

# LEGEND
plt.plot(
    [],
    [],
    color=col_feas,
    linestyle="-",
    label=(r"$\{F_2 \leq 0\}$"),
)
plt.plot(
    [],
    [],
    color=dark_orange,
    linestyle="-",
    # label=r"$\{F_2 \leq 0\}$",
    label=r"$\{\ell_{F_2}(x,x^{(t)}) - (1-\alpha)F_2(x^{(t)}) - \tau \leq 0\}$",
)
plt.plot(
    [],
    [],
    color=eth_purple,
    linestyle="--",
    label=r"$\{\ell_{F_2}(x,x^{(t)}) \leq 0\}$",
)

# current iterate, optimum and global optimum
plt.scatter(
    *x_current,
    marker=marker_start,
    # edgecolor="white",  # fill color
    # facecolor="black",  # border color
    color=col_opt,
    s=scatter_marker_size,
    label=r"$x^{(t)}$",
    zorder=5,
)

plt.scatter(
    *x_t_alpha,
    marker="o",
    facecolors=facecolors_opt_global,
    edgecolors=col_opt,
    s=scatter_marker_size,
    label=r"$x^{(t)}_\alpha$",
    zorder=5,
)

plt.scatter(
    *problem.optimum_XX,
    marker=marker_opt,
    color=col_opt,
    s=scatter_marker_size,
    label=r"$x^*$",
    zorder=5,
)


plt.colorbar(contourf, ticks=problem.ticks_colorbar)
plt.clim(problem.vmin, problem.vmax)

plt.xlabel(r"$x_1$")
plt.ylabel(r"$x_2$")
plt.legend(
    loc="lower center",
    bbox_to_anchor=(0.5, -0.55),
    ncol=ncolumns_label,
    framealpha=alpha_transparency_legend,
)
plt.title(r"Shifted Linearized Constraint $\ell_{F_2}(x, x^{(t)})$", pad=14)
plt.plot()
if path is not None:
    for ext in [".pdf", ".png"]:
        plt.savefig(path.with_suffix(ext), bbox_inches="tight")
plt.show()
