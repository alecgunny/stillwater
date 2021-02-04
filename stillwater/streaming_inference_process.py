import sys
import time
from multiprocessing import Event, Process

from stillwater.utils import ExceptionWrapper, Relative


class StreamingInferenceProcess(Process):
    def __init__(self, name: str) -> None:
        self._parents = {}
        self._children = {}

        self._pause_event = Event()
        self._stop_event = Event()
        super().__init__(name=name)

    def add_parent(self, parent: Relative):
        if parent.process is None:
            name = None
        else:
            name = parent.process.name
        self._parents[name] = parent.conn

    def add_child(self, child: Relative):
        if child.process is None:
            name = None
        else:
            name = child.process.name
        self._children[name] = child.conn

    @property
    def stopped(self):
        return self._stop_event.is_set()

    def stop(self) -> None:
        self._stop_event.set()

    @property
    def paused(self):
        return self._pause_event.is_set()

    def pause(self) -> None:
        self._pause_event.set()

    def unpause(self) -> None:
        self._pause_event.clear()

    def _break_glass(self, exception: Exception) -> None:
        """
        Our way out in case any exceptions get
        raised by our code or get passed into
        us from parent processes. Passes the
        exception to children if there are any.
        If we don't have any, you'll have to look
        at the exitcode to know that this didn't
        end pleasantly
        """
        if not isinstance(exception, ExceptionWrapper):
            exception = ExceptionWrapper(exception)

        self.stop()
        for child in self._children.values():
            child.send(exception)

    def run(self) -> None:
        """
        Wrap the code that does all the work in a try/except
        here to save some tabs and also allow _main_loop
        to be overwritten by child classes without having
        to recreate the try/except
        """
        try:
            self._main_loop()
        except Exception as e:
            self._break_glass(e)
            sys.exit(1)

    def _read_parent(self, parent_name, timeout=None):
        try:
            conn = self._parents[parent_name]
        except KeyError:
            raise ValueError(f"No parent named {parent_name}")

        # try to get an object, otherwise timeout
        if conn.poll(timeout):
            obj = conn.recv()
        else:
            return

        # if we got passed an exception from a
        # child process, it's time to exit so
        # raise it and the try/except in self.run
        # will catch it
        if isinstance(obj, Exception):
            raise obj
        return obj

    def _get_data(self):
        """
        This method can be overwritten by child classes
        that have some custom method of generating data
        to do their work on, e.g. data generators that
        don't have any children. By default, the idea
        is to read data from all the parent processes
        in a synchronized fashion
        """
        # start by looping through all parents
        # and trying to read one item from them
        ready_objs = {}
        for parent_name in self._parents:
            obj = self._read_parent(parent_name)
            if obj is not None:
                ready_objs[parent_name] = obj

        if ready_objs and len(ready_objs) < len(self._parents):
            # at least one parent returned an item, and
            # at least one did it. Try to wait around for
            # 1 second and synchronize thme
            start_time = time.time()
            _TIMEOUT = 1
            while (time.time() - start_time) < _TIMEOUT:
                # loop through those parents that didn't
                # return anything on the first go-around
                for parent_name in set(self._parents) - set(ready_objs):
                    obj = self._get(parent_name)
                    if obj is not None:
                        ready_objs[parent_name] = obj

                # if everything has returned now, we're good so
                # break out of the loop
                if len(ready_objs) == len(self._parents):
                    break
            else:
                # the loop never broke, which means at least one
                # process continued not to return anything.
                # Raise an error and report what was being difficult
                unfinished = set(self._parents) - set(ready_objs)
                raise RuntimeError(
                    "Parent processes {} stopped providing data".format(
                        ", ".join(unfinished)
                    )
                )
        elif not ready_objs:
            # if none of the parents returned anything, then
            # just keep on truckin
            return
        return ready_objs

    def _main_loop(self):
        while not self.stopped:
            ready_objs = self._get_data()
            if ready_objs is None:
                if self.paused:
                    # TODO: introduce code for updating parameters
                    # when paused
                    self.unpause()
                continue
            self._do_stuff_with_data(ready_objs)

    def _do_stuff_with_data(self, objs):
        """
        The obligatory process to implement: presumably
        you're creating this class to do something
        interesting. Here is where you do it
        """
        raise NotImplementedError
