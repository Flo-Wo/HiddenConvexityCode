from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from src.colors_setting import *

"""
PMM Objective Illustration (1D): Figure 3a and 3c
- Function: F1(x) = -cos(pi x) + 1
- Tangent at x0 chosen so that the tangent's x-axis intersection is x1 = -0.95
- Second figure: tangent shifted downward by 0.75, new intersection x1_shifted
The style follows src/colors_setting.py to stay consistent across the project.
"""


# --------------------------------------------
# Utility
# --------------------------------------------
def F1(x):
    return -np.cos(np.pi * x) + 1.0


def F1_prime(x):
    return np.pi * np.sin(np.pi * x)


def x1_from_x0(x0):
    return x0 - F1(x0) / F1_prime(x0)


def find_x0_for_target_x1(target, a=0.88, b=0.905, tol=1e-14, itmax=200):
    # Bisection on h(x0) = x1(x0) - target
    def h(x):
        return x1_from_x0(x) - target

    fa, fb = h(a), h(b)
    if fa * fb > 0:
        raise RuntimeError("Bracket does not contain a root for the requested target.")
    for _ in range(itmax):
        m = 0.5 * (a + b)
        fm = h(m)
        if fa * fm <= 0:
            b, fb = m, fm
        else:
            a, fa = m, fm
        if abs(b - a) < tol:
            break
    return 0.5 * (a + b)


# --------------------------------------------
# Saving Path
# --------------------------------------------
global_path = Path(__file__).parent.parent / "figs"
path = global_path
x_ticks = [-1, -0.5, 0, 0.5, 1]
ncol_legend = 2

# --------------------------------------------
# Styling
# --------------------------------------------
font_size = font_size_general - 6
plt.rcParams.update(
    {
        "text.usetex": use_latex,
        "font.size": font_size,
        "font.family": "serif",
        "axes.titlesize": font_size_axes,
        "axes.labelsize": font_size_label,
        "legend.fontsize": font_size_general,
        "xtick.labelsize": font_size_general,
        "ytick.labelsize": font_size_general,
    }
)

# Colors / styles
func_color = dark_blue
tangent_color = eth_purple
tangent_color_shifted = dark_orange

grid_alpha = 0.25

# Domain and key parameters
x_min, x_max = -1.2, 1.2
y_min, y_max = -0.5, 2.5
x_domain = np.linspace(-0.95, 0.95, 600)  # function drawn only here
x_plot = np.linspace(x_min, x_max, 800)  # for lines

# Choose x0 so that x1 = -0.95
target_x1 = -0.95
x0 = find_x0_for_target_x1(target_x1)

y0 = F1(x0)
slope = F1_prime(x0)


# Tangent and intersection
def tangent_vals(x):
    return slope * (x - x0) + y0


x1 = target_x1  # by construction
shift = 0.75


# --------------------------------------------
# Figure 1: Original tangent
# --------------------------------------------
def figure_original(path: Path | None):
    plt.figure(figsize=(10, 5))

    # Function on [-0.95, 0.95]
    plt.plot(x_domain, F1(x_domain), color=func_color, label=r"$F_1(x)$")

    # Tangent
    plt.plot(
        x_plot,
        tangent_vals(x_plot),
        linestyle="--",
        color=tangent_color,
        label=r"$\ell_{F_1}(x,x^{(t)})$",
    )
    # Orange highlighted segment from -1.2 to x1
    plt.plot(
        [x_min, x1],
        [0, 0],
        color=tangent_color,
        linewidth=4,
        label=r"$\{\ell_{F_1}(x,x^{(t)})\leq 0\}$",
    )

    # Points
    plt.scatter(
        [x0],
        [y0],
        c="k",
        s=scatter_marker_size,
        marker=marker_start,
        label=r"$\,(x^t,F_1(x^t))$",
    )
    plt.scatter(
        [x1],
        [0],
        c="black",
        s=scatter_marker_size,
        marker="x",
        label=r"$x^{t+1}=$" + rf"${x1:.2f}$",
    )

    # Faint guides at x = ±0.95
    for xg in [target_x1, (-1) * target_x1]:
        plt.plot([xg, xg], [0, F1(xg)], color=eth_gray, linestyle=":", linewidth=1)

    # Axes and labels
    plt.xlabel(r"$x$")
    plt.ylabel(r"$F_1(x)$")
    plt.title(r"Linearized Objective $\ell_{F_1}(x, x^{(t)})$", pad=14)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.axhline(0, color="black", linewidth=0.8, linestyle="--")
    plt.axvline(0, color="black", linewidth=0.8, linestyle="--")
    plt.xticks(x_ticks)
    plt.grid(True, alpha=grid_alpha)
    plt.gca().set_aspect(0.5)  # 2x stretched x-axis

    plt.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, -0.75),
        ncol=ncol_legend,
        framealpha=alpha_transparency_legend,
    )

    if path is not None:
        for ext in [".pdf", ".png"]:
            plt.savefig(path.with_suffix(ext), bbox_inches="tight")
    plt.show()


# --------------------------------------------
# Figure 2: Shifted tangent by 0.75
# --------------------------------------------
def figure_shifted(path: Path | None):
    plt.figure(figsize=(10, 5))

    # Function restricted
    plt.plot(x_domain, F1(x_domain), color=func_color, label=r"$F_1(x)$")

    # Original tangent point (not shifted)
    plt.scatter(
        [x0],
        [y0],
        c="k",
        s=scatter_marker_size,
        marker=marker_start,
        label=r"$\,(x^t,F_1(x^t))$",
    )

    # New intersection with y=0
    x1_shifted = x0 - (y0 - shift) / slope
    plt.scatter(
        [x1_shifted],
        [0],
        c="black",
        s=scatter_marker_size,
        marker="x",
        label=r"$x^{t+1}=$" + rf"${x1_shifted:.2f}$",
        # label=r"$x^{t+1}$",
    )

    # Original tangent (faded)
    plt.plot(
        x_plot,
        tangent_vals(x_plot),
        linestyle="--",
        color=tangent_color,
        alpha=0.5,
        label=r"$\ell_{F_1}(x,x^{(t)})$",
    )

    # Shifted tangent
    def tangent_shifted(x):
        return tangent_vals(x) - shift

    plt.plot(
        x_plot,
        tangent_shifted(x_plot),
        linestyle="--",
        color=tangent_color_shifted,
        label=r"$\ell_{F_1}(x,x^{(t)}) - (1-\alpha)F_1(x^{(t)}) - \tau$",
    )

    # Shifted highlighted orange segment to new x1
    plt.plot(
        [x_min, x1_shifted],
        [0, 0],
        color=tangent_color_shifted,
        linewidth=4,
        label=r"$\{\ell_{F_1}(x,x^{(t)}) - (1-\alpha)F_1(x^{(t)}) - \tau \leq 0\}$",
    )

    # Faint guides
    for xg in (-0.95, 0.95):
        plt.plot([xg, xg], [0, F1(xg)], color=eth_gray, linestyle=":", linewidth=1)

    # Axes and labels
    plt.xlabel(r"$x$")
    plt.ylabel(r"$F_1(x)$")
    plt.title(r"Shifted Linearized Objective $\ell_{F_1}(x, x^{(t)})$", pad=14)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(x_ticks)
    plt.axhline(0, color="black", linewidth=0.8, linestyle="--")
    plt.axvline(0, color="black", linewidth=0.8, linestyle="--")
    plt.grid(True, alpha=grid_alpha)
    # plt.gca().set_aspect(0.25)
    # plt.gca().set_aspect(0.5)

    plt.legend(
        loc="lower center",
        # bbox_to_anchor=(0.5, -1),
        bbox_to_anchor=(0.5, -0.75),
        ncol=ncol_legend,
        framealpha=alpha_transparency_legend,
    )
    if path is not None:
        for ext in [".pdf", ".png"]:
            plt.savefig(path.with_suffix(ext), bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    # out_dir = Path(__file__).parent
    figure_original(path / "PMM_objective_illustration_smooth_problem")
    figure_shifted(path / "PMM_objective_illustration_smooth_problem_SHIFTED")
