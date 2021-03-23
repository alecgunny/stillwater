import random
import time
import typing
import zlib
from collections import OrderedDict
from contextlib import contextmanager
from functools import partial
from threading import Event, Thread

import numpy as np
import tritonclient.grpc as triton

from stillwater.client import monitor as monitor_utils
from stillwater.streaming_inference_process import StreamingInferenceProcess
from stillwater.utils import ExceptionWrapper, Package, gps_time

if typing.TYPE_CHECKING:
    from multiprocessing import Process
    from multiprocessing.connection import Connection

    from stillwater.data_generator import DataGenerator


class _Callback:
    def __init__(self, metric_q):
        self._metric_q = metric_q

        self.source_map = {}
        self.message_start_times = {}
        self.request_start_times = {}
        self.sequence_id_map = {}

    def clock_start(self, sequence_id, request_id, t0):
        self.sequence_id_map[request_id] = sequence_id
        self.message_start_times[request_id] = t0
        self.request_start_times[request_id] = gps_time()

    def clock_stop(self, request_id):
        message_t0 = self.message_start_times.pop(request_id)
        send_t0 = self.request_start_times.pop(request_id)
        tf = gps_time()

        sequence_id = self.sequence_id_map.pop(request_id)
        self._metric_q.put((sequence_id, message_t0, send_t0, tf))

        return sequence_id, message_t0

    def __call__(self, result, error):
        # raise the error if anything went wrong
        if error is not None:
            if isinstance(error, triton.InferenceServerException):
                error = RuntimeError(str(error))
            exc = ExceptionWrapper(error)
            for conn in self.source_map.values():
                conn.send(exc)
            return

        request_id = int(result.get_response().id)
        sequence_id, message_t0 = self.clock_stop(request_id)


        np_output = {}
        for output in result._result.outputs:
            np_output[output.name] = Package(
                x=result.as_numpy(output.name), t0=message_t0
            )
        self.source_map[sequence_id].send(np_output)


def _client_stream(
    data_source,
    inputs,
    sequence_id,
    client,
    model_name,
    model_version,
    stop_event,
    callback,
    states,
):
    try:
        sequence_start = True
        data_source = iter(data_source)

        # if we passed a value to states, we should only
        # have one input: the streaming input tensor.
        # Otherwise, just set it equal to the inputs
        if states is not None:
            assert isinstance(inputs, triton.InferInput)
        else:
            states = inputs

        while not stop_event.is_set():
            request_id = random.randint(0, 1e9)

            try:
                packages = next(data_source)
            except StopIteration as e:
                callback(None, e)
                break

            if not isinstance(packages, dict):
                packages = {name: x for name, x in zip(states, [packages])}
            assert len(packages) == len(states)

            x, t0 = [], 0

            # if we're using streams, the order matters since
            # we need to concatenate them. Otherwise we'll just
            # grab the appropriate input and set its value
            for name, input in states.items():
                package = packages[name]

                if isinstance(input, triton.InferInput):
                    # states weren't provided, so update the
                    # appropriate input
                    input.set_data_from_numpy(package.x)
                else:
                    # we have streaming states, deal with these
                    # later. Add a dummy batch dimension
                    x.append(package.x[None])

                # use the average of package creation times as
                # the value for latency measurement. Shouldn't
                # make a difference for most practical use cases
                # since these should be the same (in fact it's
                # probably worth checking to ensure that)
                t0 += package.t0

            # concatenate streaming states if we have them and
            # set the stream input
            if len(x) > 0:
                if len(x) > 1:
                    x = np.concatenate(x, axis=1)
                else:
                    x = x[0]
                inputs.set_data_from_numpy(x)
                infer_inputs = [inputs]
            else:
                infer_inputs = list(states.values())

            t0 /= len(packages)
            callback.clock_start(sequence_id, request_id, t0)

            client.async_stream_infer(
                model_name,
                model_version=str(model_version),
                inputs=infer_inputs,
                request_id=str(request_id),
                sequence_start=sequence_start,
                sequence_id=sequence_id,
                timeout=60,
            )
            sequence_start = False

    except Exception as e:
        data_source.stop()
        callback(None, e)


class ThreadedMultiStreamInferenceClient(StreamingInferenceProcess):
    def __init__(
        self,
        url: str,
        model_name: str,
        model_version: int,
        name: typing.Optional[str] = None,
    ) -> None:
        # do a few checks on the server to make sure we'll be good to go
        try:
            client = triton.InferenceServerClient(url)

            if not client.is_server_live():
                raise RuntimeError(f"Server at url {url} isn't live")
            if not client.is_model_ready(model_name):
                raise RuntimeError(
                    f"Model {model_name} isn't ready at server url {url}"
                )
        except triton.InferenceServerException:
            raise RuntimeError(f"Couldn't connect to server at {url}")

        super().__init__(name)
        self.model_metadata = client.get_model_metadata(model_name)
        self.model_config = client.get_model_config(model_name).config
        self.url = url
        self.client = client

        states = self.model_config.parameters.get("states", None)
        if states is not None:
            update_size = self.model_metadata.inputs[0].shape[-1]

            _states = OrderedDict()
            for state in states.string_value.split(","):
                model, input = state.split("/")
                step_config = client.get_model_config(model).config

                input = [i for i in step_config.input if i.name == input]
                if len(input) == 0:
                    raise ValueError(
                        f"Couldn't find model input for state {state}"
                    )
                _states[state] = (input[0].dims[1], update_size)
            states = _states
        self.states = states

        self._source_stop_event = Event()
        self._callback = _Callback(self._metric_q)
        self._target_fn = partial(
            _client_stream,
            client=self.client,
            model_name=model_name,
            model_version=model_version,
            stop_event=self._stop_event,
            callback=self._callback,
            states=states,
        )
        self._streams = []

    def add_data_source(
        self,
        source: "DataGenerator",
        child: typing.Union[str, "Process"],
        sequence_id: typing.Optional[int] = None,
        conn: typing.Optional["Connection"] = None,
    ):
        if self.is_alive() or self.exitcode is not None:
            raise ValueError(
                "Cannot add data source to already started client process"
            )

        inputs = {}
        for input in self.model_metadata.inputs:
            input = triton.InferInput(input.name, input.shape, input.datatype)
            if self.states is None:
                inputs[input.name()] = input
            else:
                inputs = input

        if sequence_id is None:
            sequence_id = zlib.adler32(source.name.encode("utf-8"))

        stream = Thread(
            target=self._target_fn, args=(source, inputs, sequence_id)
        )
        self._streams.append(stream)

        conn = self.add_child(child, conn)

        if not isinstance(child, str):
            child = child.name
        self._callback.source_map[sequence_id] = getattr(self._children, child)

        return conn

    def stop(self) -> None:
        self._source_stop_event.set()
        super().stop()

    def _main_loop(self):
        self._metric_q.put(("start_time", gps_time()))

        with self.client as client:
            client.start_stream(callback=self._callback, stream_timeout=60)
            for stream in self._streams:
                stream.start()

            # since the streams also wait for our stop
            # signal, we can just use them to block
            for stream in self._streams:
                stream.join()

    def monitor(
        self,
        pipes: typing.Optional[typing.Dict[str, "Connection"]] = None,
        server_monitor: typing.Optional[typing.Dict[str, float]] = None,
        client_monitor: typing.Optional[typing.Dict[str, float]] = None,
        timeout: float = 10.0,
    ):
        pipes = pipes or {}

        if server_monitor is not None:
            try:
                output_file = server_monitor.pop("output_file")
            except KeyError:
                output_file = None

            server_monitor = monitor_utils.ServerStatsMonitor(
                self, monitor=server_monitor, output_file=output_file
            )
            server_monitor.start()

        if client_monitor is not None:
            try:
                output_file = client_monitor.pop("output_file")
            except KeyError:
                output_file = None

            client_monitor = monitor_utils.ClientStatsMonitor(
                self, monitor=client_monitor, output_file=output_file
            )
            client_monitor.start()

        self.start()
        try:
            last_package_time = time.time()
            latency, throughput, request_rate, max_msg_length = 0, 0, 0, 0
            while True:
                # first go through our monitors and see if
                # any errors got raised
                for monitor in [client_monitor, server_monitor]:
                    if monitor is None:
                        continue

                    error = monitor.error
                    if error is not None:
                        error.reraise()

                # now iterate through our pipes and try to return
                # packages if we have any, along with the sequence
                # to which the package belongs
                for name, pipe in pipes.items():
                    if pipe.poll():
                        packages = pipe.recv()
                    else:
                        continue

                    if isinstance(packages, ExceptionWrapper):
                        packages.reraise()
                    elif packages is None:
                        continue

                    last_package_time = time.time()
                    yield name, packages

                # if we've gone over `timeout` seconds without a
                # package, something has gone wrong, so exit
                if time.time() - last_package_time > timeout:
                    raise RuntimeError(
                        "Timed out after no packages arrived "
                        f"for {timeout} seconds"
                    )

                # TODO: this should be moved to the script calling
                # I think, but I don't have a good way of getting
                # this info there
                if client_monitor is None:
                    continue
                latency = client_monitor.latency or latency
                throughput = client_monitor.throughput or throughput
                request_rate = client_monitor.request_rate or request_rate

                msg = f"Average latency: {latency} us, "
                msg += f"Average throughput: {throughput:0.1f} frames / s, "
                msg += f"Average request rate: {request_rate:0.1f} frames / s"

                max_msg_length = max(max_msg_length, len(msg))
                msg += " " * (max_msg_length - len(msg))
                print(msg, end="\r", flush=True)
        finally:
            print("\n")

            # shut everything down
            for p in [self, server_monitor, client_monitor]:
                if p is None:
                    continue

                p.stop()
                p.join(0.5)
                try:
                    p.close()
                except AttributeError:
                    continue
                except ValueError:
                    p.terminate()
                    time.sleep(0.1)
                    p.close()
