import numpy as np
from dataclasses import dataclass
from typing import Callable, Dict, Tuple


def proj_box(x: np.ndarray, l: float, u: float) -> np.ndarray:
    return np.clip(x, l, u)


def posynomial_eval(b: np.ndarray, A: np.ndarray, x: np.ndarray) -> float:
    logx = np.log(x)
    return float(np.sum(np.exp(np.log(b) + A @ logx)))


def posynomial_grad(b: np.ndarray, A: np.ndarray, x: np.ndarray) -> np.ndarray:
    logx = np.log(x)
    term = np.exp(np.log(b) + A @ logx)
    return (term[:, None] * A).sum(axis=0) / x


@dataclass
class CGPProblem:
    F0: Callable[[np.ndarray], float]
    dF0: Callable[[np.ndarray], np.ndarray]
    F1: Callable[[np.ndarray], float]
    dF1: Callable[[np.ndarray], np.ndarray]
    box: Tuple[float, float]
    x0: np.ndarray
    tau: float
    rho_hat: float

    def projection(self, x: np.ndarray) -> np.ndarray:
        l, u = self.box
        return proj_box(x, l, u)


def build_near_tight_instance(
    d: int = 100,
    K0: int = 10,
    K1: int = 8,
    seed: int = 21,
    box: Tuple[float, float] = (0.5, 2.0),
    tau: float = 1e-3,
    rho_hat: float = 1.0,
) -> Tuple[CGPProblem, Dict[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    A0 = rng.uniform(-0.5, 0.5, size=(K0, d))
    A1 = rng.uniform(-0.5, 0.5, size=(K1, d))
    b0 = np.exp(rng.normal(0.0, 0.4, size=K0))
    b1 = np.exp(rng.normal(0.0, 0.4, size=K1))
    b1 *= 1 / b1.sum()

    F0 = lambda x: posynomial_eval(b0, A0, x)
    dF0 = lambda x: posynomial_grad(b0, A0, x)
    F1 = lambda x: posynomial_eval(b1, A1, x)
    dF1 = lambda x: posynomial_grad(b1, A1, x)

    problem = CGPProblem(
        F0=F0,
        dF0=dF0,
        F1=F1,
        dF1=dF1,
        box=box,
        x0=np.ones(d),
        tau=tau,
        rho_hat=rho_hat,
    )

    return problem, {"A0": A0, "b0": b0, "A1": A1, "b1": b1}
