from abc import ABC, abstractmethod
import numpy as np
from dataclasses import dataclass, field


# abstract problem class
class ProblemAbstract(ABC):
    poi: list = []

    # transformation
    @abstractmethod
    def c_func(self, x: np.ndarray) -> np.ndarray:
        pass

    # methods in X-space
    @abstractmethod
    def F1(self, x: np.ndarray) -> float:
        pass

    @abstractmethod
    def partial_F1(self, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def F2(self, x: np.ndarray) -> float:
        pass

    @abstractmethod
    def partial_F2(self, x: np.ndarray) -> np.ndarray:
        pass

    # methods in the U-space
    @abstractmethod
    def F1_UU(self, x: np.ndarray) -> float:
        pass

    @abstractmethod
    def F2_UU(self, x: np.ndarray) -> float:
        pass


class OptimizationResult:
    def __init__(
        self,
        x_final: np.ndarray,
        iter_history: np.ndarray,
        history: np.ndarray,
        label: str,
        marker: str,
        linestyle: str,
        color: str | tuple,
        _skip_iterates=1,
    ):
        self.x_final = x_final
        self.iter_history = iter_history
        self._history = history
        self.label = label
        self.marker = marker
        self.linestyle = linestyle
        self.color = color
        self._skip_iterates = _skip_iterates

    @property
    def history(self):
        return self._history

    @history.setter
    def history(self, value):
        self._history = value

    @property
    def skipped_history(self):
        return self._history[:: self._skip_iterates]

    @property
    def skipped_iter_history(self):
        return self.iter_history[:: self._skip_iterates]
