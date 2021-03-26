import re
import time
import typing
from collections import defaultdict
from contextlib import contextmanager
from itertools import product
from queue import Empty
from multiprocessing import Event, Process, Queue

import requests
from tritonclient import grpc as triton

from stillwater.utils import ExceptionWrapper

if typing.TYPE_CHECKING:
    from stillwater.client import ThreadedMultiStreamInferenceClient


class ServerInferenceMetric:
    def __init__(self):
        self.value = 0
        self.count = 0

    def update(self, count, value):
        update = count - self.count
        if update == 0:
            return
        average = (value - self.value) / update

        self.count = count
        self.value = value
        return average, update


class MonitoredMetricViolationException(Exception):
    pass


# TODO: can we just do this in the init of the exception?
# will that screw things up with ExceptionWrapper?
def _raise_exception(metric, limit, value):
    raise MonitoredMetricViolationException(
        f"Metric {metric} violated limit {limit} with value {value}"
    )


class ThreadedStatWriter(Process):
    def __init__(
        self,
        columns: typing.List[str],
        output_file: typing.Optional[str] = None,
        quantities: typing.Optional[typing.Dict[str, typing.Callable]] = None,
        monitors: typing.Optional[typing.Dict[str, float]] = None,
    ) -> None:
        self.output_file = output_file
        self.columns = columns

        self.quantities = quantities or {}
        self.monitors = monitors or {}
        for metric in self.monitors:
            assert metric in self.quantities

        self.f = None
        self._stop_event = Event()
        self._error_q = Queue()
        super().__init__()

        # TODO: make configurable
        self._grace_period = 10000
        self._n = 0
        self._sustain = 500
        self._steps_since_violation_started = 0

    def monitor(self, values):
        for metric, value in values.items():
            try:
                limit = self.monitors[metric]
            except KeyError:
                continue

            if (
                self.quantities[metric](limit, value) == limit and
                self._n >= self._grace_period and
                self._steps_since_violation_started >= self._sustain
            ):
                _raise_exception(metric, limit, value)
            elif (
                self.quantities[metric](limit, value) == limit and
                self._n >= self._grace_period
            ):
                self._steps_since_violation_started += 1
            elif self.quantities[metric](limit, value) == value:
                self._steps_since_violation_started = 0

    @property
    def stopped(self) -> bool:
        return self._stop_event.is_set()

    def stop(self) -> None:
        self._stop_event.set()

    @property
    def error(self) -> Exception:
        try:
            return self._error_q.get_nowait()
        except Empty:
            return None

    @contextmanager
    def open(self):
        if self.output_file is not None:
            f = open(self.output_file, "w")
            try:
                yield f
            finally:
                f.close()
        else:
            yield

    def write_row(self, f, values):
        values = list(map(str, values))
        if len(values) != len(self.columns):
            raise ValueError(
                "Can't write values {} with length {}, "
                "doesn't match number of columns {}".format(
                    ", ".join(values), len(values), len(self.columns)
                )
            )
        f.write("\n" + ",".join(values))

    def run(self):
        with self.open() as f:
            if f is not None:
                f.write(",".join(self.columns))

            while not self.stopped:
                try:
                    values = self._get_values()
                    if f is None or values is None:
                        if values is not None:
                            self._n += 1
                        continue
                    self._n += 1

                    if not isinstance(values[0], list):
                        values = [values]
                    for v in values:
                        self.write_row(f, v)

                except Exception as e:
                    self.stop()
                    self._error_q.put(ExceptionWrapper(e))

    def _get_values(self):
        raise NotImplementedError


class ServerStatsMonitor(ThreadedStatWriter):
    def __init__(
        self,
        client: "ThreadedMultiStreamInferenceClient",
        output_file: typing.Optional[str] = None,
        monitor: typing.Optional[typing.Dict[str, float]] = None,
    ):
        model_config = client.model_config
        self.models = [model_config.name]

        if len(model_config.ensemble_scheduling.step) > 0:
            self.models += [
                i.model_name for i in model_config.ensemble_scheduling.step
            ]

        # TODO: make this more general
        self.url = "http://" + client.url.replace("8001", "8002/metrics")
        self.stats = defaultdict(lambda: defaultdict(ServerInferenceMetric))

        processes = [
            "success",
            "queue",
            "compute_input",
            "compute_infer",
            "compute_output",
        ]

        quantities = {}
        for model, process in product(self.models, processes):
            quantities[f"{model}_{process}"] = min

        columns = processes + ["gpu_utilization", "model", "count", "interval"]
        self._last_time = time.time()

        super().__init__(
            output_file=output_file,
            columns=columns,
            quantities=quantities,
            monitors=monitor
        )

        # initialize metrics
        _ = self._get_values()

    def _get_gpu_id(self, row):
        id = re.search('(?<=gpu_uuid=").+(?="})', row)
        if id is not None:
            return id.group(0)
        return

    def _get_row_value(self, row):
        return float(row.split()[1])

    def _filter_rows_by_start(self, rows, start):
        return [i for i in rows if i.startswith(start)]

    def _filter_rows_by_content(self, rows, content):
        return [i for i in rows if content in i]

    def _get_values(self):
        response = requests.get(self.url)
        new_time = time.time()
        interval = new_time - self._last_time
        self._last_time = new_time

        content = response.content.decode("utf-8").split("\n")
        util_rows = self._filter_rows_by_start(content, "nv_gpu_utilization")
        gpu_util = sum(map(self._get_row_value, util_rows)) / len(util_rows)

        values = []
        for model in self.models:
            model_stats = self._filter_rows_by_content(
                content, f'model="{model}",'
            )
            if len(model_stats) == 0:
                raise ValueError("couldn't find model in content")

            counts = map(
                self._get_row_value,
                self._filter_rows_by_start(
                    model_stats, "nv_inference_request_success"
                ),
            )
            count = sum(counts)

            model_values = []
            for process in self.columns[:5]:
                field = process if process != "success" else "request"
                process_times = list(
                    map(
                        self._get_row_value,
                        self._filter_rows_by_start(
                            model_stats, f"nv_inference_{field}_duration_us"
                        ),
                    )
                )

                us = sum(process_times)
                data = self.stats[model][process].update(count, us)
                if data is None:
                    return

                t, diff = data
                model_values.append(t)
                self.monitor({f"{model}_{process}": t})

            model_values.extend([gpu_util, model, diff, interval])
            values.append(model_values)

        return values


class ClientStatsMonitor(ThreadedStatWriter):
    def __init__(
        self,
        client: "ThreadedMultiStreamInferenceClient",
        output_file: typing.Optional[str] = None,
        monitor: typing.Optional[typing.List[str]] = None,
    ):
        self.q = client._metric_q
        self.n = 0

        self._latency = 0

        self._throughput_q = Queue()
        self._request_rate_q = Queue()
        self._latency_q = Queue()

        super().__init__(
            output_file=output_file,
            columns=[
                "sequence_id",
                "message_start",
                "request_send",
                "request_return",
            ],
            quantities={"latency": min, "throughput": max},
            monitors=monitor,
        )

    def _get_values(self):
        try:
            measurements = self.q.get_nowait()
        except Empty:
            return

        if measurements[0] == "start_time":
            self.start_time = measurements[1]
            return

        measurements = list(measurements)
        sequence_id = measurements.pop(0)
        measurements = [i - self.start_time for i in measurements]
        measurements = [sequence_id] + measurements

        self.n += 1
        latency = measurements[-1] - measurements[1]
        self._latency += (latency - self._latency) / self.n
        throughput = self.n / measurements[-1]

        self._throughput_q.put(throughput)
        self._request_rate_q.put(self.n / measurements[1])
        self._latency_q.put(self._latency)

        self.monitor({"latency": self._latency, "throughput": throughput})
        return measurements

    @property
    def latency(self):
        value = 0
        for i in range(100):
            try:
                value = self._latency_q.get_nowait()
            except Empty:
                break
        return int(value * 10 ** 6)

    @property
    def throughput(self):
        value = 0
        for i in range(100):
            try:
                value = self._throughput_q.get_nowait()
            except Empty:
                break
        return value

    @property
    def request_rate(self):
        value = 0
        for i in range(100):
            try:
                value = self._request_rate_q.get_nowait()
            except Empty:
                break
        return value
