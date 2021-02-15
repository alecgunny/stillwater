import typing

import attr
import numpy as np

from stillwater.data_generator import DataGenerator
from stillwater.utils import gps_time, Package


@attr.s(auto_attribs=True)
class DummyDataGeneratorFn:
    shape: typing.Tuple[int, ...]
    sample_rate: typing.Optional[float]

    def __attrs_post_init__(self):
        if self.sample_rate is not None:
            self._wait_time = self.shape[-1] / self.sample_rate
        else:
            self._wait_time = None
        self._last_time = gps_time()

    def __call__(self, idx):
        if (
            self._wait_time is not None
            and gps_time() - self._last_time < self._wait_time
        ):
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
        sample_rate: typing.Optional[float],
    ) -> None:
        generator_fn = DummyDataGeneratorFn(shape, sample_rate)
        super().__init__(generator_fn, 1, name)
