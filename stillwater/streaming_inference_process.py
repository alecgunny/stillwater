import sys
import typing
from multiprocessing import Event, Pipe, Process

from stillwater.utils import ExceptionWrapper, Relatives, sync_recv

if typing.TYPE_CHECKING:
    from multiprocessing.connection import Connection


class StreamingInferenceProcess(Process):
    def __init__(self, name: str) -> None:
        self._parents = Relatives()
        self._children = Relatives()

        self._pause_event = Event()
        self._stop_event = Event()
        super().__init__(name=name)

    def add_parent(
        self,
        parent: typing.Union[Process, str],
        conn: typing.Optional["Connection"] = None,
    ) -> typing.Optional["Connection"]:
        # make sure we're not routing us to ourselves
        if parent is self:
            raise ValueError("Cannot pipe process to itself!")

        # try to get name
        try:
            parent_name = parent.name
        except AttributeError:
            parent_name = parent

        # check if we already have an incoming connection
        # from a process by this name
        try:
            existing_conn = getattr(self._parents, parent_name)
        except AttributeError:
            pass
        else:
            # if we do and we didn't pass a connection, or
            # the one we passed isn't the one we already have,
            # raise an exception
            if conn is None or conn is not existing_conn:
                raise ValueError(
                    f"Process {parent.name} already a parent to "
                    f"process {self.name}"
                )

        # if we don't make a connection here,
        # don't pass anything back to the main
        # process
        parent_conn = None
        if conn is None:
            # we didn't pass a connection, so make one here
            parent_conn, conn = Pipe()
            if not isinstance(parent, str):
                # add ourselves as a child to the parent
                # process
                parent.add_child(self, parent_conn)

        # maybe return a connection if we had to
        # create one here
        _return_conn = None
        if not isinstance(parent, str):
            # parent is a process, so we don't
            # need to return anything. Grab the
            # name for adding the relative
            parent = parent.name
        else:
            # parent is a string, so we want to
            # return the connection to the main
            # process if we had to create it
            _return_conn = parent_conn

        # add relative
        self._parents = self._parents.add(parent, conn)
        return _return_conn

    def add_child(
        self,
        child: typing.Union[Process, str],
        conn: typing.Optional["Connection"] = None,
    ) -> typing.Optional["Connection"]:
        # make sure we're not routing us to ourselves
        if child is self:
            raise ValueError("Cannot pipe process to itself!")

        # try to get name
        try:
            child_name = child.name
        except AttributeError:
            child_name = child

        # check if we already have outgoing connection for
        # a process by this name
        try:
            existing_conn = getattr(self._children, child_name)
        except AttributeError:
            pass
        else:
            # if we do and we didn't pass a connection, or
            # the one we passed isn't the one we already have,
            # raise an exception
            if conn is None or conn is not existing_conn:
                raise ValueError(
                    f"Process {child.name} already a parent to "
                    f"process {self.name}"
                )

        # see comments from parent section for
        # explanation as to what's happening here
        child_conn = None
        if conn is None:
            conn, child_conn = Pipe()
            if not isinstance(child, str):
                child.add_parent(self, child_conn)

        _return_conn = None
        if not isinstance(child, str):
            child = child.name
        else:
            _return_conn = child_conn

        self._children = self._children.add(child, conn)
        return _return_conn

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
        pipes = {f: getattr(self._parents, f) for f in self._parents._fields}
        return sync_recv(pipes)

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
