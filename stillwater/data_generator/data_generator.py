import typing


class DataGenerator:
    def __init__(
        self, generator_fn: typing.Callable, idx_range: int, name: str
    ) -> None:
        self._generator_fn = generator_fn
        self.idx = 0
        self.idx_range = idx_range
        self.name = name

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return self._generator_fn(self.idx)
        except IndexError:
            self.idx = 0
            return self._generator_fn(self.idx)

    def stop(self):
        return


class MultiSourceGenerator(DataGenerator):
    def __init__(self, data_generators, name=None):
        def generator_fn(idx):
            packages = {}
            for gen in data_generators:
                packages[gen.name] = gen._generator_fn(idx)
            return packages

        super().__init__(generator_fn, data_generators[0].idx_range, name)
