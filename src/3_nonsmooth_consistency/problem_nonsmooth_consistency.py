import numpy as np
from src.base import ProblemAbstract


def c_inv(u):
    u1, u2 = u
    x1 = u1 + 1
    x2 = 2 * np.abs(x1) - u2 - 1
    return np.array([x1, x2])


def c1_func(x):
    x1, _ = x
    return x1 - 1


def c2_func(x):
    x1, x2 = x
    return 2 * abs(x1) - x2 - 1  # non-smooth


class ProblemNonSmoothConsistency(ProblemAbstract):
    def __init__(self, epsilon: float):
        self.smooth = False

        self.optimum_XX = np.array([0.5, 1.4])
        # optimum_XX = c_inv(np.array([-0.15, -0.15]))
        self.optimum_global_XX = np.array([0, 2])
        self.optimum_UU = np.array([-0.5, 0.2])

        self.b1 = np.array([0, 2])
        self.b2 = np.array([-0.5, -0.6])

        self.rho = 2.0
        self.rho_hat = 2 * self.rho
        self.mu = self.rho_hat - self.rho
        self.G_subgradients = 2.0

        self.mu_c = 4.0
        self.D_U = np.sqrt(2) * 2.25

        # Slater's condition
        self.LAMBDA = 0.5
        self.beta_PPPM = 2 * self.LAMBDA + 2
        # Equation (11) in Huang et al. with our formula of theta for the stronger Slater
        self.nu = np.sqrt(4 * self.G_subgradients**2 / self.LAMBDA)
        # self.nu = np.sqrt(4 / 3) * (self.mu_c / self.D_U) * epsilon

        # options for plotting
        num_levels_colorbar = 200
        self.vmin, self.vmax = 0, 4.5
        self.ticks_colorbar = [0.5 * i for i in range(0, 10)]
        self.levels_colorbar = np.linspace(0, 4.5, num_levels_colorbar)

        # additional points of interest
        self.poi = [np.array([-0.3, 0.2])]
        self.poi_label_XX = [r"$x_\mathrm{local}$"]
        self.poi_label_UU = [r"$u_\mathrm{local}$"]
        self.poi_marker = ["."]

    def c_func(self, x: np.ndarray) -> np.ndarray:
        return np.array([c1_func(x), c2_func(x)])

    # get the (generalized) Jacobian of c
    def partial_c_func(self, x: np.ndarray) -> np.ndarray:
        return np.array([[1, 0], [2 * np.sign(x[0]), -1]])

    def F1(self, x: np.ndarray) -> float:
        return np.linalg.norm(x - self.b1, ord=2)  # X-space

    def partial_F1(self, x: np.ndarray) -> np.ndarray:
        # not used
        pass

    def F2(self, x: np.ndarray) -> float:
        # return np.linalg.norm(x - center, ord=1) - 0.8 # U-space
        return np.linalg.norm(self.c_func(x) - self.b2, ord=1) - 0.8  # X-space

    def partial_F2(self, x: np.ndarray) -> np.ndarray:
        # not used
        pass

    def F1_UU(self, u: np.ndarray) -> float:
        return np.linalg.norm(c_inv(u) - self.b1, ord=2)  # U-space

    def F2_UU(self, u: np.ndarray) -> float:
        return np.linalg.norm(u - self.b2, ord=1) - 0.8  # U-space
