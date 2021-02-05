import sys
import typing
from collections import namedtuple
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
    parent_conn, child_conn = Pipe()

    str_relative = namedtuple("Relative", ["process", "conn"])
    if isinstance(parent, str) and isinstance(child, str):
        raise ValueError("Must provide at least one process to pipe between")

    parent_cls = str_relative if isinstance(parent, str) else Relative
    child_cls = str_relative if isinstance(parent, str) else Relative

    parent = parent_cls(parent, parent_conn)
    child = child_cls(child, child_conn)
    if not isinstance(parent, str_relative):
        parent.add_child(child)
    else:
        conn = child.add_parent(parent)

    if not isinstance(child, str_relative):
        child.add_parent(parent)
    else:
        conn = parent.add_child(child)
    return conn
