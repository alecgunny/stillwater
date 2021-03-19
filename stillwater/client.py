import random
import typing
import zlib
from threading import Event, Thread

import numpy as np
import tritonclient.grpc as triton

from stillwater.streaming_inference_process import StreamingInferenceProcess
from stillwater.utils import ExceptionWrapper, Package, gps_time

if typing.TYPE_CHECKING:
    from multiprocessing import Process
    from multiprocessing.connection import Connection


def _get_seq_id(source):
    return zlib.adler32(source.name.encode("utf-8"))


class Callback:
    def __init__(self, data_sources, children, metric_q):
        self.source_map = {
            _get_seq_id(source): source.name for source in data_sources
        }
        self.children = children
        self.metric_q = metric_q

        self.message_start_times = {}
        self.request_start_times = {}
        self.sequence_id_map = {}

    def __call__(self, result, error):
        # raise the error if anything went wrong
        if error is not None:
            exc = ExceptionWrapper(error)
            for conn in self.children:
                conn.send(exc)
            return

        request_id = int(result.get_response().id)

        message_t0 = self.message_start_times.pop(request_id)
        send_t0 = self.request_send_times.pop(request_id)
        tf = gps_time()
        self.metric_q.put((message_t0, send_t0, tf))

        sequence_id = self.sequence_id_map.pop(request_id)
        source_name = self.source_map[sequence_id]

        result = {}
        for output_name in result._result.outputs:
            result[output_name] = Package(
                x=result.as_numpy(output_name), t0=message_t0
            )
        self.children[source_name].send(result)


def _client_stream(
    client,
    data_source,
    model_name,
    model_version,
    stop_event,
    inputs,
    input_map,
    callback,
    streams,
):
    try:
        sequence_id = zlib.adler32(data_source.name)
        sequence_start = True

        data_source = iter(data_source)
        while not stop_event.is_set():
            request_id = random.randint(0, 1e9)

            packages = next(data_source)
            if not isinstance(packages, dict):
                packages = {data_source.name: packages}
            assert len(packages) == len(input_map)

            x, t0 = [], 0

            # if we're using streams, the order matters since
            # we need to concatenate them. Otherwise we'll just
            # grab the appropriate input and set its value
            streams = streams or packages.keys()
            for stream in streams:
                package = packages[input_map[stream]]

                if streams is None:
                    # assuming that if you're not streaming,
                    # you take care of batching on your own
                    inputs[stream].set_data_from_numpy(package.x)
                else:
                    # add a dummy batch dimension for streams
                    x.append(package.x[None])

                # use the average of package creation times as
                # the value for latency measurement. Shouldn't
                # make a difference for most practical use cases
                # since these should be the same (in fact it's
                # probably worth checking to ensure that)
                t0 += package.t0

            # concatenate streams if we have them and
            # set the stream input
            if len(x) > 0:
                if len(x) > 1:
                    x = np.concatenate(x, axis=1)
                else:
                    x = x[0]
                inputs["stream"].set_data_from_numpy(x)

            t0 /= len(packages)

            callback.message_start_times[request_id] = t0
            callback.request_send_times[request_id] = gps_time()
            callback.sequence_id_map[request_id] = sequence_id

            client.async_stream_infer(
                model_name,
                model_version=str(model_version),
                inputs=list(inputs.values()),
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
        data_sources: typing.List[typing.Callable],
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
        self.client = client

        model_config = client.get_model_config(model_name).config
        streams = model_config.parameters.get("streams", None)
        if streams is not None:
            streams = streams.string_value.split(",")

        self._source_stop_event = Event()
        self._callback = Callback(data_sources, self.children, self._metric_q)

        self._streams = []
        for data_source in data_sources:
            inputs = {}
            for input in client.get_model_metadata(model_name).inputs:
                inputs[input.name] = triton.InferInput(
                    input.name, tuple(input.shape), input.datatype
                )

            args = (
                self.client,
                data_source,
                model_name,
                model_version,
                self._source_stop_event,
                inputs,
                self._callback,
                streams,
            )
            stream = Thread(target=_client_stream, args=args)
            self._streams.append(stream)

        # expose some properties about the inputs to
        # the outside world
        self.inputs = {name: tuple(x.shape) for name, x in inputs.items()}
        self.streams = streams

    @property
    def sources(self):
        return list(self._callback.source_map.values())

    def add_child(
        self,
        child: typing.Union[str, "Process"],
        conn: typing.Optional["Connection"] = None,
    ):
        try:
            child_name = child.name
        except AttributeError:
            child_name = child

        if child_name not in self.sources:
            raise ValueError(
                "Tried to add output named {} "
                "to inference client expecting "
                "outputs {}".format(child_name, ", ".join(self.inputs.keys()))
            )
        return super().add_child(child, conn)

    def stop(self) -> None:
        self._source_stop_event.set()
        super().stop()

    def _main_loop(self):
        output_names = set([x.name() for x in self.outputs])
        missing_outputs = output_names - set(self.output_map)
        if not len(missing_outputs) == 0:
            raise RuntimeError(
                "Couldn't start inference client process, "
                "missing outputs {}".format(", ".join(missing_outputs))
            )

        with self.client as client:
            client.start_stream(callback=self._callback, stream_timeout=60)
            for stream in self._streams:
                stream.start()

            # since the streams also wait for our stop
            # signal, we can just use them to block
            for stream in self._streams:
                stream.join()
