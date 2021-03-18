import random
import string
import time
import typing
import zlib

import numpy as np
import tritonclient.grpc as triton

from stillwater.streaming_inference_process import StreamingInferenceProcess
from stillwater.utils import ExceptionWrapper, Package, gps_time

if typing.TYPE_CHECKING:
    from multiprocessing import Process
    from multiprocessing.connection import Connection


class StreamingInferenceClient(StreamingInferenceProcess):
    def __init__(
        self,
        url: str,
        model_name: str,
        model_version: int,
        name: str,
        sequence_id: typing.Optional[typing.Union[int, str, bytes]] = None,
        qps_limit: typing.Optional[int] = None,
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

        self.url = url
        self.model_name = model_name
        self.model_version = model_version

        # add a throttle
        if qps_limit is not None:
            self._wait_time = 1 / qps_limit - 1e-5
        else:
            self._wait_time = None

        # try to map sequence_id types to an int
        if not isinstance(sequence_id, int):
            if sequence_id is None:
                random_string = "".join(random.choices(string.printable, k=16))
                sequence_id = zlib.adler32(bytes(random_string))
            elif isinstance(sequence_id, (str, bytes)):
                sequence_id = zlib.adler32(bytes(sequence_id))
            else:
                raise ValueError(f"Invalid sequence id {sequence_id}")
        self.sequence_id = sequence_id

        # use the server to tell us what inputs and
        # outputs we need to expose for the given model
        # self._inputs represents the internal real input
        # that gets exposed to the model, while self.inputs
        # represents the named inputs that get exposed to
        # upstream data generators (for streaming input with
        # multiple streams)
        model_metadata = client.get_model_metadata(model_name)
        self._inputs = {
            x.name: triton.InferInput(x.name, tuple(x.shape), x.datatype)
            for x in model_metadata.inputs
        }

        model_config = client.get_model_config(model_name).config
        try:
            self.streams = model_config.parameters[
                "stream_channels"
            ].string_value.split(",")
        except KeyError:
            # no stream channels were specified, so our
            # inputs are as reported by model metadata
            self.inputs = self._inputs
            self.streams = None
        else:
            # we have some stream channels, so figure out
            # how big each one is supposed to be from the
            # corresponding model info and expose them
            # as inputs to the client
            self.inputs = {}
            for stream in self.streams:
                input_model_name, input_name = stream.split("/")
                md = client.get_model_metadata(input_model_name)
                input = [x for x in md.inputs if x.name == input_name][0]

                shape = list(input.shape)
                shape[-1] = model_config.input[0].dims[-1]
                self.inputs[stream] = triton.InferInput(
                    input.name, shape, input.datatype
                )

        # TODO: will need to implement similar logic once
        # output snapshotting is made functional
        self.outputs = [
            triton.InferRequestedOutput(output.name)
            for output in model_metadata.outputs
        ]

        self.input_map, self.output_map = {}, {}

    def add_parent(
        self,
        parent: typing.Union[str, "Process"],
        conn: typing.Optional["Connection"] = None,
        input_name: typing.Optional[str] = None,
    ):
        """
        Override these methods in order to match process
        names with the inputs and outputs the model is
        expecting
        """
        # add the key, then make sure it was valid to
        # add, deleting it an erroring if it wasn't
        try:
            parent_name = parent.name
        except AttributeError:
            parent_name = parent

        input_name = input_name or parent_name
        if input_name not in self.inputs:
            raise ValueError(
                "Tried to add data source named {} "
                "to inference client expecting "
                "sources {}".format(parent_name, ", ".join(self.inputs.keys()))
            )

        self.input_map[input_name] = parent_name
        return super().add_parent(parent, conn)

    def add_child(
        self,
        child: typing.Union[str, "Process"],
        conn: typing.Optional["Connection"] = None,
        output_name: typing.Optional[str] = None,
    ):
        try:
            child_name = child.name
        except AttributeError:
            child_name = child

        output_name = output_name or child_name
        output_names = [x.name() for x in self.outputs]
        if child_name not in output_names:
            raise ValueError(
                "Tried to add output named {} "
                "to inference client expecting "
                "outputs {}".format(child_name, ", ".join(self.inputs.keys()))
            )

        self.output_map[output_name] = child_name
        return super().add_child(child, conn)

    def _initialize_run(self):
        self._metric_q.put(("start_time", gps_time()))
        self._last_request_time = time.time()
        self._request_id = 0
        self._start_times = {}
        self._recv_times = {}
        self._send_times = {}

    def _callback(self, result, error):
        # raise the error if anything went wrong
        if error is not None:
            for conn in self._children:
                exc = ExceptionWrapper(
                    RuntimeError("Server returned error: " + str(error))
                )
                conn.send(exc)
            self.stop()
            return

        id = int(result.get_response().id)
        try:
            message_t0 = self._start_times.pop(id)
        except KeyError:
            # this is here because of issues with pausing,
            # but is a terrible solution and TODO: needs to
            # be sorted out
            return

        send_t0 = self._send_times.pop(id)
        recv_t0 = self._recv_times.pop(id)
        tf = gps_time()
        self._metric_q.put((message_t0, recv_t0, send_t0, tf))

        for output_name, child_name in self.output_map.items():
            x = result.as_numpy(output_name)
            conn = getattr(self._children, child_name)
            conn.send(Package(x, message_t0))

    def _main_loop(self):
        # first make sure we've plugged in all the necessary
        # input data streams, and have places to send all
        # the necessary outputs
        missing_sources = set(self.inputs) - set(self.input_map)
        if not len(missing_sources) == 0:
            raise RuntimeError(
                "Couldn't start inference client process, "
                "missing sources {}".format(", ".join(missing_sources))
            )

        output_names = set([x.name() for x in self.outputs])
        missing_outputs = output_names - set(self.output_map)
        if not len(missing_outputs) == 0:
            raise RuntimeError(
                "Couldn't start inference client process, "
                "missing outputs {}".format(", ".join(missing_outputs))
            )

        # call the main loop within a client context to make
        # sure the client stream closes if anything goes awry
        with triton.InferenceServerClient(url=self.url) as self.client:
            self.client.start_stream(callback=self._callback, stream_timeout=60)
            self._initialize_run()
            super()._main_loop()

    def _do_stuff_with_data(self, objs):
        assert len(objs) == len(self.inputs)

        x, t0 = [], 0

        # if we're using streams, the order matters since
        # we need to concatenate them. Otherwise we'll just
        # grab the appropriate input and set its value
        streams = self.streams or objs.keys()
        for stream in streams:
            package = objs[self.input_map[stream]]
            if self.streams is None:
                self._inputs[stream].set_data_from_numpy(package.x[None])
            else:
                x.append(package.x[None])

            # use the average of package creation times as
            # the value for latency measurement. Shouldn't
            # make a different for most practical use cases
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
            self._inputs["stream"].set_data_from_numpy(x)

        t0 /= len(objs)

        if self._wait_time is not None:
            while (time.time() - self._last_request_time) < self._wait_time:
                time.sleep(1e-6)

        self._request_id += 1
        self._send_times[self._request_id + 0] = gps_time()
        self._start_times[self._request_id + 0] = t0

        self.client.async_stream_infer(
            self.model_name,
            inputs=list(self._inputs.values()),
            outputs=self.outputs,
            request_id=str(self._request_id),
            sequence_start=self._request_id == 1,
            sequence_id=self.sequence_id,
            timeout=60,
        )

        self._last_request_time = time.time()

    def _get_data(self):
        stuff = super()._get_data()
        self._recv_times[self._request_id + 1] = gps_time()
        return stuff

    def reset(self):
        # wait for all in-flight requests to return
        while self.client._stream._request_queue.qsize() > 0:
            time.sleep(0.001)

        # reinitialize metric trackers
        self._initialize_run()

        # clear the input qs
        super().reset()
