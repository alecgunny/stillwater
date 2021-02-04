import time
import typing

import numpy as np

from stillwater.data_generator import DataGenerator
from stillwater.utils import Package


class DummyDataGenerator(DataGenerator):
    def __init__(
        self,
        shape: typing.Tuple[int, ...],
        name: str,
        sample_rate: typing.Optional[float],
    ) -> None:
        self.shape = shape

        _wait_time = None
        if sample_rate is not None:
            _wait_time = sample_rate * shape[-1]
        self._last_time = time.time()

        def _generator_fn(idx):
            if (
                _wait_time is not None
                and time.time() - self._last_time < _wait_time
            ):
                return
            x = np.random.randn(*shape)
            package = Package(x=x, t0=time.time())
            self._last_time = package.t0
            return package

        super().__init__(_generator_fn, 1, name)
