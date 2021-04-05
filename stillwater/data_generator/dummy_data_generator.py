import time
import typing

import numpy as np

from stillwater.data_generator import DataGenerator
from stillwater.utils import Package, gps_time


class DummyDataGenerator(DataGenerator):
    def __init__(
        self,
        shape: typing.Tuple[int, ...],
        name: str,
        generation_rate: typing.Optional[float] = None,
    ) -> None:
        super().__init__(name=name)
        self.shape = shape
        if generation_rate is not None:
            self._sleep_time = 1.0 / generation_rate - 2e-4
        else:
            self._sleep_time = None
        self._last_time = gps_time()

    def __next__(self):
        if self._sleep_time is not None:
            while (time.time() - self._last_time) < self._sleep_time:
                time.sleep(1e-6)
        x = np.random.randn(*self.shape).astype("float32")
        package = Package(x=x, t0=gps_time())
        self._last_time = package.t0
        return package
