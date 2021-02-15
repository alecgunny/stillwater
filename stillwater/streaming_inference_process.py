import sys
import time
import typing
from multiprocessing import Event, Process

from stillwater.utils import ExceptionWrapper, Relative, sync_recv

if typing.TYPE_CHECKING:
    from multiprocessing.connection import Connection


class StreamingInferenceProcess(Process):
    def __init__(self, name: str) -> None:
        self._parents = {}
        self._children = {}

        self._pause_event = Event()
        self._stop_event = Event()
        super().__init__(name=name)

    def add_parent(self, parent: Relative) -> "Connection":
        if parent.process is None:
            name = None
        else:
            name = parent.process.name
        if name in self._parents and self._parents[name] is not parent.conn:
            raise ValueError(f"Parent {name} already has associated connection.")
        elif name not in self._parents:
            self._parents[name] = parent.conn
        return parent.conn

    def add_child(self, child: Relative) -> "Connection":
        if child.process is None:
            name = None
        else:
            name = child.process.name
        if name in self._children and self._children[name] is not child.conn:
            raise ValueError(f"Child {name} already has associated connection.")
        elif name not in self._children:
            self._children[name] = child.conn
        return child.conn

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
        print(exception)
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

    def _get_data(self):
        """
        This method can be overwritten by child classes
        that have some custom method of generating data
        to do their work on, e.g. data generators that
        don't have any children. By default, the idea
        is to read data from all the parent processes
        in a synchronized fashion
        """
        return sync_recv(self._parents)

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
