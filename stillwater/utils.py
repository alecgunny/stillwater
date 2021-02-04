import sys
import typing
from multiprocessing import Pipe

import attr
import numpy as np
from tblib import pickling_support

if typing.TYPE_CHECKING:
    from multiprocessing.connection import Connection

    from stillwater.streaming_inference_process import (
        StreamingInferenceProcess,
    )


@attr.s(auto_attribs=True)
class Relative:
    process: "StreamingInferenceProcess"
    conn: "Connection"


@attr.s(auto_attribs=True)
class Package:
    x: np.ndarray
    t0: float


@pickling_support.install
class ExceptionWrapper(Exception):
    def __init__(self, exc: Exception) -> None:
        self.exc = exc
        _, __, self.tb = sys.exc_info()

    def reraise(self) -> None:
        raise self.exc.with_traceback(self.tb)


def pipe(
    parent: typing.Optional["StreamingInferenceProcess"],
    child: typing.Optional["StreamingInferenceProcess"],
) -> None:
    # TODO: use none checks to do piping to
    # main process
    parent_conn, child_conn = Pipe()
    parent.add_child(Relative(child, child_conn))
    child.add_parent(Relative(parent, parent_conn))
