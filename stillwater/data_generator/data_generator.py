import typing


class DataGenerator:
    def __init__(self, name=None):
        self.name = name

    def __iter__(self):
        return self

    def __next__(self):
        raise NotImplementedError

    def stop(self):
        return


class MultiSourceGenerator(DataGenerator):
    def __init__(self, data_generators, name=None):
        self._data_generators = [iter(d) for d in data_generators]
        super().__init__(name)

    def __next__(self):
        packages = {}
        for gen in self._data_generators:
            packages[gen.name] = next(gen)
        return packages
