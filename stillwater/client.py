import ctypes
import random
import string
import time
import typing

import tritonclient.grpc as triton

from stillwater.streaming_inference_process import StreamingInferenceProcess
from stillwater.utils import ExceptionWrapper, Package, Relative


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
        for input in model_metadata.inputs:
            self.inputs[input.name] = triton.InferInput(
                input.name, tuple(input.shape), input.datatype
            )
        self.outputs = [
            triton.InferRequestedOutput(output.name)
            for output in model_metadata.outputs
        ]

    def add_parent(self, parent: Relative):
        """
        Override these methods in order to match process
        names with the inputs and outputs the model is
        expecting
        """
        # add the key, then make sure it was valid to
        # add, deleting it an erroring if it wasn't
        current_keys = set(self._parents)
        super().add_parent(parent)

        new_key = (set(self._parents) - current_keys).pop()
        if new_key not in self.inputs:
            self._parents.pop(new_key)
            raise ValueError(
                "Tried to add data source named {} "
                "to inference client expecting "
                "sources {}".format(new_key, ", ".join(self.inputs.keys()))
            )

    def add_child(self, child: Relative):
        current_keys = set(self._children)
        super().add_child(child)

        new_key = (set(self._children) - current_keys).pop()
        if new_key not in [x.name() for x in self.outputs]:
            raise ValueError(
                "Tried to add output named {} "
                "to inference client expecting "
                "outputs {}".format(new_key, ", ".join(self.inputs.keys()))
            )

    def _callback(self, result, error):
        # raise the error if anything went wrong
        if error is not None:
            for name, conn in self._children.items():
                exc = ExceptionWrapper(RuntimeError(error))
                conn.send(exc)
            self.stop()
            return

        id = int(result.get_response().id)
        t0 = self._start_times.pop(id)

        # TODO: use GPS time
        end_time = time.time()
        latency = end_time - t0
        throughput = id / (end_time - self._start_time)
        for name, conn in self._children.items():
            x = result.as_numpy(name)
            conn.send(Package(x, (latency, throughput)))

    def _main_loop(self):
        # first make sure we've plugged in all the necessary
        # input data streams, and have places to send all
        # the necessary outputs
        missing_sources = set(self.inputs) - set(self._parents)
        if not len(missing_sources) == 0:
            raise RuntimeError(
                "Couldn't start inference client process, "
                "missing sources {}".format(", ".join(missing_sources))
            )

        output_names = set([x.name() for x in self.outputs])
        missing_outputs = output_names - set(self._children)
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
            self._start_time = time.time()
            self._request_id = 0
            super()._main_loop()

    def _do_stuff_with_data(self, objs):
        t0 = 0
        assert len(objs) == len(self.inputs)
        for name, package in objs.items():
            self.inputs[name].set_data_from_numpy(package.x[None])
            t0 += package.t0
        t0 /= len(objs)

        self._request_id += 1
        self._start_times[self._request_id + 0] = t0

        self.client.async_stream_infer(
            self.model_name,
            inputs=list(self.inputs.values()),
            outputs=self.outputs,
            request_id=str(self._request_id),
            sequence_start=self._request_id == 1,
            sequence_id=self.sequence_id,
        )
