import typing

import attr
import numpy as np

from stillwater.data_generator import DataGenerator
from stillwater.utils import Package, gps_time


@attr.s(auto_attribs=True)
class DummyDataGeneratorFn:
    shape: typing.Tuple[int, ...]
    generation_rate: typing.Optional[float]

    def __attrs_post_init__(self):
        self._last_time = gps_time()

    def __call__(self, idx):
        if self.generation_rate is not None:
            if (gps_time() - self._last_time) < (1 / self.generation_rate):
                return

        x = np.random.randn(*self.shape).astype("float32")
        package = Package(x=x, t0=gps_time())
        self._last_time = package.t0
        return package


class DummyDataGenerator(DataGenerator):
    def __init__(
        self,
        shape: typing.Tuple[int, ...],
        name: str,
        generation_rate: typing.Optional[float] = None,
    ) -> None:
        generator_fn = DummyDataGeneratorFn(shape, generation_rate)
        super().__init__(generator_fn, 1, name)
