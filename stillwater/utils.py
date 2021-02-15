import datetime
import sys
import time
import typing
from collections import namedtuple
from itertools import starmap
from multiprocessing import Pipe

import attr
import numpy as np
from tblib import pickling_support

if typing.TYPE_CHECKING:
    from multiprocessing.connection import Connection

    from stillwater.streaming_inference_process import (
        StreamingInferenceProcess,
    )


_GPS_EPOCH = datetime.datetime(
    1980, 1, 6, 0, 0, 0, # tzinfo=datetime.timezone.utc
).timestamp()
def gps_time():
    """
    TODO: why are these off?
    """
    return time.time() # datetime.datetime.utcnow().timestamp() - _GPS_EPOCH + 7


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

    str_process = namedtuple("Process", ["name"])
    str_relative = namedtuple("Relative", ["process", "conn"])
    if isinstance(parent, str) and isinstance(child, str):
        raise ValueError("Must provide at least one process to pipe between")

    conn = None
    if isinstance(parent, str):
        parent = str_relative(str_process(parent), child_conn)
        _ = child.add_parent(parent)
        conn = parent_conn
    else:
        if isinstance(child, str):
            child = str_relative(str_process(child), parent_conn)
            _ = parent.add_child(child)
            conn = child_conn
        else:
            parent.add_child(Relative(child, parent_conn))
            child.add_parent(Relative(parent, child_conn))
    return conn


def sync_recv(
    pipes: typing.Dict[str, "Connection"],
    timeout: float = 1.0
) -> typing.Optional["Connection"]:
    """
    Do a synchronized reading from multiple connections
    to inference processes. Works by iterating through
    the connections, polling to check for available
    objects, and receiving them if they're available.
    If all connections retrun `conn.poll() = False`,
    `None` will be returned. Otherwise, if *at least*
    one process returns an object, this process will
    block for at most `timeout` seconds to allow the
    other processes the opportunity to return an object.
    If they don't, a `RuntimeError` will be raised.
    Otherwise, a dictionary mapping the keys in `pipes`
    to the corresponding objects will be returned.

    :param pipes: Dictionary mapping from process names
        to `multiprocessing.connection.Connection`
        objects from which data will be attempted to
        be received.
    :param timeout: Maximum amount of time to block
        waiting for processes to return objects if
        any other processes have returned something.
    """
    ready_objs = {}

    def _recv_and_update(name, conn):
        if not conn.poll():
            return
        obj = conn.recv()

        # if we got passed an exception from a
        # parent process, it's time to exit so
        # raise it and the try/except in self.run
        # will catch it
        if isinstance(obj, ExceptionWrapper):
            obj.reraise()
        elif isinstance(obj, Exception):
            # TODO: should this be an error? I think
            # any exceptions should be caught and
            # wrapped first right?
            raise obj
        ready_objs[name] = obj

    for name, conn in pipes.items():
        _recv_and_update(name, conn)
    if not ready_objs:
        # Nothing ever got returned, so bail
        return

    if len(ready_objs) < len(pipes):
        # at least one parent returned an item, and
        # at least one did not. Try to wait around for
        # `timeout` seconds and synchronize them
        start_time = time.time()
        while (time.time() - start_time) < timeout:
            # loop through those parents that didn't
            # return anything on the first go-around
            for name in set(pipes) - set(ready_objs):
                _recv_and_update(name, pipes[name])

            if len(ready_objs) == len(pipes):
                # every process has now returned something,
                # so break out of the loop
                break
        else:
            # the loop never broke, which means at least one
            # process continued not to return anything.
            # Raise an error and report what was being difficult
            unfinished = set(pipes) - set(ready_objs)
            raise RuntimeError(
                "Couldn't sync processes {}, processes {} "
                "unable to provide data before timeout".format(
                    ", ".join(pipes), ", ".join(unfinished)
                )
            )
    return ready_objs
