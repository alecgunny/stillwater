import sys
import time
import typing
from collections import namedtuple

import attr
import numpy as np
from tblib import pickling_support

if typing.TYPE_CHECKING:
    from multiprocessing.connection import Connection


class ObjectMappingTuple:

    """
    Utility class for creating namedtuples capable
    of adding or subtracting members. Useful for
    building variable length, dictionary-like objects
    that can autocomplete their members and are immutable
    TODO: maybe something like a mappingproxy makes more
    sense? This is bad because we can do isinstance checking
    etc.
    """

    def __new__(
        cls,
        fields: typing.Optional[typing.List[str]] = None,
        values: typing.Optional[typing.List[None]] = None,
    ):
        class ObjectMapping(namedtuple(cls._name, fields or [])):
            def add(self, field: str, value: cls._value_type):
                fields = self._fields + (field,)
                values = [getattr(self, f) for f in self._fields] + [value]
                return cls.__new__(cls, fields, values)

            def remove(self, field: str):
                if field not in self._fields:
                    raise ValueError(f"No field named {field}")
                fields = [f for f in self._fields if f != field]
                values = [getattr(self, f) for f in fields]
                return cls.__new__(cls, fields, values)

        values = values or []
        return ObjectMapping(*values)


class Relatives(ObjectMappingTuple):
    _name = "Relative"
    _value_type = Connection


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


def sync_recv(
    pipes: typing.Dict[str, "Connection"], timeout: float = 1.0
) -> typing.Optional[typing.Dict[str, typing.Union[Package, ExceptionWrapper]]]:
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
