import ctypes
import random
import string
import time
import typing

import numpy as np
import tritonclient.grpc as triton

from stillwater.streaming_inference_process import StreamingInferenceProcess
from stillwater.utils import ExceptionWrapper, gps_time, Package

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
        sequence_id: typing.Optional[typing.Union[int, str]] = None,
    ) -> None:
        # do a few checks on the server to make sure
        # we'll be good to go
        try:
            client = triton.InferenceServerClient(url)
        except triton.InferenceServerException:
            raise RuntimeError(
                "Couldn't connect to server at specified " "url {}".format(url)
            )
        if not client.is_server_live():
            raise RuntimeError("Server at url {} isn't live".format(url))
        if not client.is_model_ready(model_name):
            raise RuntimeError(
                "Model {} isn't ready at server url {}".format(model_name, url)
            )
        super().__init__(name)

        self.url = url
        self.model_name = model_name
        self.model_version = model_version
        self._start_times = {}

        if not isinstance(sequence_id, int):
            if sequence_id is None:
                random_string = "".join(random.choices(string.printable, k=16))
                sequence_id = hash(random_string)
            elif isinstance(sequence_id, str):
                # TODO: won't be repeatable between different
                # runs of the same script: is that a problem?
                sequence_id = hash(sequence_id)
            else:
                raise ValueError(f"Invalid sequence id {sequence_id}")
            # map to positive int
            sequence_id = ctypes.c_size_t(sequence_id).value
        self.sequence_id = sequence_id

        # use the server to tell us what inputs and
        # outputs we need to expose for the given
        # model
        model_metadata = client.get_model_metadata(model_name)
        self.inputs = {}
        inputs = [
            ("witness_h", (1, 21, 4000)),
            ("witness_l", (1, 21, 4000)),
            ("strain", (1, 2, 4000))
        ]
        print(model_metadata.inputs)
        for name, shape in inputs:
            self.inputs[name] = triton.InferInput(
                name, tuple(shape), model_metadata.inputs[0].datatype
            )
        self._inputs = triton.InferInput(
            "stream", tuple(model_metadata.inputs[0].shape), model_metadata.inputs[0].datatype
        )
        self.outputs = [
            triton.InferRequestedOutput(output.name)
            for output in model_metadata.outputs
        ]
        self._send_times = {}
        self._av_send_time = 0.

    def add_parent(
        self,
        parent: typing.Union[str, "Process"],
        conn: typing.Optional["Connection"] = None,
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
        if parent_name not in self.inputs:
            raise ValueError(
                "Tried to add data source named {} "
                "to inference client expecting "
                "sources {}".format(parent_name, ", ".join(self.inputs.keys()))
            )
        return super().add_parent(parent, conn)

    def add_child(
        self,
        child: typing.Union[str, "Process"],
        conn: typing.Optional["Connection"] = None,
    ):
        try:
            child_name = child.name
        except AttributeError:
            child_name = child
        if child_name not in [x.name() for x in self.outputs]:
            raise ValueError(
                "Tried to add output named {} "
                "to inference client expecting "
                "outputs {}".format(child_name, ", ".join(self.inputs.keys()))
            )
        return super().add_child(child, conn)

    def _callback(self, result, error):
        # raise the error if anything went wrong
        if error is not None:
            print(error)
            for conn in self._children:
                exc = ExceptionWrapper(RuntimeError(error))
                conn.send(exc)
            self.stop()
            return

        id = int(result.get_response().id)
        t0 = self._start_times.pop(id)
        send_t0 = self._send_times.pop(id)

        end_time = gps_time()
        self._av_send_time += (end_time - send_t0 - self._av_send_time) / id

        latency = end_time - t0
        throughput = id / (end_time - self._start_time)
        for name in self._children._fields:
            x = result.as_numpy(name)
            conn = getattr(self._children, name)
            conn.send(Package(x, (latency, throughput)))

    def _main_loop(self):
        # first make sure we've plugged in all the necessary
        # input data streams, and have places to send all
        # the necessary outputs
        missing_sources = set(self.inputs) - set(self._parents._fields)
        if not len(missing_sources) == 0:
            raise RuntimeError(
                "Couldn't start inference client process, "
                "missing sources {}".format(", ".join(missing_sources))
            )

        output_names = set([x.name() for x in self.outputs])
        missing_outputs = output_names - set(self._children._fields)
        if not len(missing_outputs) == 0:
            raise RuntimeError(
                "Couldn't start inference client process, "
                "missing outputs {}".format(", ".join(missing_outputs))
            )

        # call the main loop within a client context
        # to make sure the client stream closes if
        # anything goes awry
        with triton.InferenceServerClient(url=self.url) as self.client:
            self.client.start_stream(callback=self._callback, stream_timeout=60)

            # measure the stream start time as a
            # point of reference for profiling purposes
            self._start_time = gps_time()
            self._request_id = 0
            super()._main_loop()

    def _do_stuff_with_data(self, objs):
        t0 = 0
        assert len(objs) == len(self.inputs)
        x = []
        for name, package in objs.items():
            x.append(package.x[None])
            t0 += package.t0
        x = np.concatenate(x, axis=1)
        self._inputs.set_data_from_numpy(x)
        t0 /= len(objs)

        self._request_id += 1
        self._send_times[self._request_id + 0] = time.time()
        self._start_times[self._request_id + 0] = t0

        self.client.async_stream_infer(
            self.model_name,
            inputs=[self._inputs],
            outputs=self.outputs,
            request_id=str(self._request_id),
            sequence_start=self._request_id == 1,
            sequence_id=self.sequence_id,
        )

    def get_inference_stats(self):
        with triton.InferenceServerClient(self.url) as client:
            return client.get_inference_statistics().model_stats
