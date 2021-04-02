import time
import typing
from multiprocessing import Event, Process, Queue
from queue import Empty

import numpy as np

from stillwater.data_generator import DataGenerator
from stillwater.utils import ExceptionWrapper, Package, gps_time


class Stopped(Exception):
    pass


def generate(q, stop_event, shape, generation_rate):
    if generation_rate is not None:
        _sleep_time = 1 / generation_rate - 1e-4
    else:
        _sleep_time = None

    try:
        last_time = time.time()
        while not stop_event.is_set():
            if _sleep_time is not None:
                while (time.time() - last_time) < _sleep_time:
                    time.sleep(1e-6)

            while q.full():
                if stop_event.is_set():
                    raise Stopped
                time.sleep(1e-6)
            x = np.random.randn(*shape).astype("float32")
            package = Package(x=x, t0=gps_time())
            last_time = package.t0
            q.put(package)

    except Stopped:
        pass
    except Exception as e:
        q.put(ExceptionWrapper(e))


class DummyDataGenerator(DataGenerator):
    def __init__(
        self,
        shape: typing.Tuple[int, ...],
        name: str,
        generation_rate: typing.Optional[float] = None,
    ) -> None:
        super().__init__(name=name)
        self._q = Queue(maxsize=100)
        self._stop_event = Event()
        self.shape = shape
        self.generation_rate = generation_rate

    def __iter__(self):
        self._p = Process(
            target=generate,
            args=(self._q, self._stop_event, self.shape, self.generation_rate)
        )
        self._p.start()
        return self

    def __next__(self):
        try:
            package = self._q.get(timeout=1)
        except Empty:
            raise RuntimeError("Nothing generating")

        if isinstance(package, ExceptionWrapper):
            package.reraise()
        return package

    def stop(self):
        self._stop_event.set()
        self._p.join(0.5)
        try:
            self._p.close()
        except ValueError:
            self._p.terminate()

