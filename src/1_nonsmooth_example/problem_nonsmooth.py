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


class ProblemNonSmooth(ProblemAbstract):
    def __init__(self, epsilon: float):
        self.smooth = False

        self.optimum_XX = np.array([-0.15 + 1, 2 * abs(-0.15 + 1) + 0.15 - 1])
        self.optimum_global_XX = np.array([1, 1])

        self.b1 = np.array([0, 0])
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

    def c_func(self, x: np.ndarray) -> np.ndarray:
        return np.array([c1_func(x), c2_func(x)])

    # get the (generalized) Jacobian of c
    def partial_c_func(self, x: np.ndarray) -> np.ndarray:
        return np.array([[1, 0], [2 * np.sign(x[0]), -1]])

    def F1(self, x: np.ndarray) -> float:
        return np.linalg.norm(self.c_func(x) - self.b1, ord=np.inf)  # X-space

    def partial_F1(self, x: np.ndarray) -> np.ndarray:
        c_val = self.c_func(x)
        # max_abs_c = np.max(np.abs(c_val))
        F1_value = self.F1(x)
        active_indices = np.where(np.abs(c_val - self.b1) == F1_value)[0]
        J_c = self.partial_c_func(x)
        sub_grad = np.zeros_like(x)
        for i in active_indices:
            sub_grad += np.sign(c_val[i] - self.b1[i]) * J_c[i, :]
        sub_grad /= len(active_indices)
        return sub_grad

    def F2(self, x: np.ndarray) -> float:
        return np.linalg.norm(self.c_func(x) - self.b2, ord=1) - 0.8  # X-space

    def partial_F2(self, x: np.ndarray) -> np.ndarray:
        return np.transpose(self.partial_c_func(x)) @ np.sign(self.c_func(x) - self.b2)

    def F1_UU(self, u: np.ndarray) -> float:
        return np.linalg.norm(u - self.b1, ord=np.inf)  # U-space

    def F2_UU(self, u: np.ndarray) -> float:
        return np.linalg.norm(u - self.b2, ord=1) - 0.8  # U-space
