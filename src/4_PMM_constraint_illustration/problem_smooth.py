import numpy as np
from src.base import ProblemAbstract


class ProblemSmooth(ProblemAbstract):

    def __init__(self, epsilon: float):
        self.smooth = True
        # global and feasible optima
        self.optimum_global_UU = np.array([4 * np.log(2) / 3, -2 * np.log(2) / 3])
        self.optimum_global_XX = self.c_inv(self.optimum_global_UU)
        self.optimum_UU = self.optimum_global_UU - np.log(2) / 3
        self.optimum_XX = self.c_inv(self.optimum_UU)

        # problem constants
        self.rho = 1
        self.rho_hat = 2 * self.rho
        self.mu = self.rho_hat - self.rho
        self.mu_c = 1 / 3
        self.G_subgradients = (
            25.286  # for X = [0.4, 3]^2 while we have 412.189 for x=[0.1, 3]^2
        )
        self.D_X = np.sqrt(2) * 2.6
        self.D_U = 2
        self.LAMBDA = 85  # 20
        self.lambdaOpt = 1
        self.beta_PPPM = 2 * (self.lambdaOpt + 1) ** 2 / epsilon  # 10
        nu = np.sqrt(4 * self.G_subgradients**2 / self.LAMBDA)

        # min and max of F1
        self.vmin, self.vmax = 4.5, 13
        num_levels_colorbar = 200
        self.ticks_colorbar = [
            self.vmin + (self.vmax - self.vmin) / 5 * i for i in range(0, 6)
        ]
        self.levels_colorbar = np.linspace(self.vmin, self.vmax, num_levels_colorbar)

        # ACGD constants
        self.sqrt_kappa = np.inf
        self.L = self.G_subgradients + self.rho_hat * self.D_X**2
        self.slater_value = 0.25
        self.beta = 0.001
        self.shift = epsilon + self.beta * self.slater_value / 3
        self.L_Lambda_r = self.L * (
            1
            + (self.vmax - self.vmin + self.slater_value)
            / (self.beta * self.slater_value / 6)
        )
        self.epsilon = epsilon

    def F1(self, x: np.ndarray) -> float:
        return x[0] * x[1] + 4 / x[0] + 1 / x[1]

    def F2(self, x: np.ndarray) -> float:
        return x[0] * x[1] - 1

    # in the U-space
    def F1_UU(self, u: np.ndarray) -> float:
        return np.exp(u[0] + u[1]) + 4 * np.exp(-u[0]) + 1 * np.exp(-u[1])

    def F2_UU(self, u: np.ndarray) -> float:
        return u[0] + u[1]

    # gradients
    def partial_F1(self, x: np.ndarray) -> np.ndarray:
        return np.array([-4 / (x[0] ** 2) + x[1], -1 / (x[1] ** 2) + x[0]])

    def partial_F2(self, x: np.ndarray) -> np.ndarray:
        return np.array([x[1], x[0]])

    # transformation function
    def c_func(self, x):
        return np.log(x)

    def c_inv(self, u):
        return np.exp(u)
