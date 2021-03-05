import os
import re
import time
import typing

import attr
import numpy as np
from gwpy.timeseries import TimeSeriesDict

from stillwater.data_generator import DataGenerator
from stillwater.utils import Package


@attr.s(auto_attribs=True)
class LowLatencyFrameGeneratorFn:
    path_pattern: str
    t0: int
    kernel_stride: float
    sample_rate: float
    channels: typing.List[str]

    def __attrs_post_init__(self):
        self.data = None
        self._latency_t0 = None
        self._update_size = int(self.sample_rate * self.kernel_stride)

    def __call__(self, idx):
        start = idx * self._update_size
        stop = (idx + 1) * self._update_size

        if self.data is None or stop > self.data.shape[1]:
            # try to load in the next second's worth of data
            # if it takes more than a second to get created,
            # then assume the worst and raise an error
            start_time = time.time()
            while time.time() - start_time < 3:
                try:
                    path = self.path_pattern.format(self.t0)
                    data = TimeSeriesDict.read(path, self.channels)
                    break
                except FileNotFoundError:
                    continue
            else:
                raise ValueError(f"Couldn't find next timestep file {path}")
            self._latency_t0 = os.stat(path).st_ctime - 1

            # resample the data and turn it into a numpy array
            data.resample(self.sample_rate)
            data = np.stack(
                [data[channel].value for channel in self.channels]
            ).astype("float32")

            if self.data is not None and start < self.data.shape[1]:
                leftover = self.data[:, start:]
                data = np.concatenate([leftover, data], axis=1)
            self.data = data
            self.t0 += 1

            # raising an index error will get the DataGenerator's
            # _get_data fn to reset the index
            raise IndexError

        # return the next piece of data
        x = self.data[:, start:stop]

        # offset the frame's initial time by the time
        # corresponding to the first sample of stream
        t0 = self._latency_t0 + idx * self.kernel_stride
        return Package(x=x, t0=t0)


class LowLatencyFrameGenerator(DataGenerator):
    def __init__(
        self,
        data_dir: str,
        channels: typing.List[str],
        sample_rate: float,
        kernel_stride: float,
        t0: typing.Optional[int] = None,
        file_pattern: typing.Optional[str] = None,
        name: str = None,
    ) -> None:
        t0, path_pattern = self._get_t0_and_path(data_dir, t0, file_pattern)

        generator_fn = LowLatencyFrameGeneratorFn(
            path_pattern,
            t0 + 0,
            kernel_stride,
            sample_rate,
            channels,
        )
        idx_range = int(1 / kernel_stride) + 1
        super().__init__(generator_fn, idx_range, name)

    def reset(self):
        super().reset()

        t0, _ = self._get_t0_and_path(
            os.path.dirname(self._generator_fn.path_pattern),
            os.path.basename(self._generator_fn.path_pattern),
            None,
        )
        self._generator_fn.data = None
        self._generator_fn.t0 = t0

    def _get_t0_and_path(self, data_dir, file_pattern, t0):
        if file_pattern is None and t0 is None:
            raise ValueError(
                "Must specify either a file pattern or initial timestamp"
            )
        elif file_pattern is None:
            # we didn't specify a file pattern, so use the initial
            # timestamp with the data directory to look for a pattern
            # that we can use
            file_pattern = re.compile(fr".*-{t0}-.*\.gwf")
            files = list(filter(file_pattern.full_match, os.listdir(data_dir)))
            if len(files) == 0:
                raise ValueError(
                    "Couldn't find any files matching timestamp {} "
                    "in directory {}".format(t0, data_dir)
                )
            elif len(files) > 1:
                raise ValueError(
                    "Found more than 1 file matching timestamp {} "
                    "in directory {}: {}".format(t0, data_dir, ", ".join(files))
                )
            file_pattern = files[0].replace(str(t0), "{}")
        elif t0 is None:
            prefix, postfix = file_pattern.split("{}")
            regex = re.compile(
                "(?<={})[0-9]{}(?={})".format(prefix, "{10}", postfix)
            )
            timestamps = map(regex.search, os.listdir(data_dir))
            if not any(timestamps):
                raise ValueError(
                    "Couldn't find any timestamps matching the "
                    "pattern {}".format(file_pattern)
                )
            timestamps = [int(t.group(0)) for t in timestamps if t is not None]
            t0 = max(timestamps)
        return t0, os.path.join(data_dir, file_pattern)
