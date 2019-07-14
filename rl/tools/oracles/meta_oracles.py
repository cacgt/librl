from abc import abstractmethod
import numpy as np
import copy
from rl.tools.utils.mvavg import ExpMvAvg
from rl.tools.oracles.oracle import Oracle


class MetaOracle(Oracle):
    """These Oracles are built on other Oracle objects."""

    @abstractmethod
    def __init__(self, base_oracle, *args, **kwargs):
        """It should have attribute base_oracle or base_oracles."""


class DummyOracle(MetaOracle):

    def __init__(self, base_oracle, *args, **kwargs):
        self._base_oracle = copy.deepcopy(base_oracle)
        self._g = 0.

    def fun(self, x):
        return 0.

    def grad(self, x):
        return self._g

    def update(self, g=None, *args, **kwargs):
        assert g is not None
        self._base_oracle.update(*args, **kwargs)
        self._g = np.copy(g)


class LazyOracle(MetaOracle):
    """Function-based oracle based on moving average."""

    def __init__(self, base_oracle, beta=0.):
        self._base_oracle = copy.deepcopy(base_oracle)
        self._beta = beta
        self._f = ExpMvAvg(None, beta)
        self._g = ExpMvAvg(None, beta)

    def update(self, x, *args, **kwargs):
        self._base_oracle.update(*args, **kwargs)
        self._f.update(self._base_oracle.fun(x))
        self._g.update(self._base_oracle.grad(x))

    def fun(self, x):
        return self._f.val

    def grad(self, x):
        return self._g.val


class AdversarialOracle(LazyOracle):
    """For debugging purpose."""

    def __init__(self, base_oracle, beta):
        super().__init__(base_oracle, beta)
        self._max = None

    def grad(self, x):
        g = super().grad(x)
        if self._max is None:
            self._max = np.linalg.norm(g)
        else:
            self._max = max(self._max, np.linalg.norm(g))
        return -g / max(np.linalg.norm(g), 1e-5) * self._max
