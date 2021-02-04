import typing

from stillwater import StreamingInferenceProcess


class DataGenerator(StreamingInferenceProcess):
    def __init__(
        self, generator_fn: typing.Callable, idx_range: int, name: str
    ) -> None:
        super().__init__(name)
        self._generator_fn = generator_fn
        self.idx = 0
        self.idx_range = idx_range
        super().__init__(name)

    def _get_data(self):
        try:
            return self._generator_fn(self.idx)
        except IndexError:
            self.idx = 0
            return self._generator_fn(self.idx)

    def _do_stuff_with_data(self, objs):
        for child in self._children.values():
            child.send(objs)

        self.idx += 1
        if self.idx == self.idx_range:
            self.idx = 0
