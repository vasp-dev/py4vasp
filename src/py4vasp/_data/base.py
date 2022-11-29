# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import contextlib
import dataclasses
import functools
import pathlib

from py4vasp import exception, raw
from py4vasp._util import check, convert


def data_access(func):
    """Use this decorator for all public methods of Refinery children. It creates the
    necessary wrappers to load the data from the VASP output and makes it available
    via the _raw_data property."""

    @functools.wraps(func)
    def func_with_access(self, *args, selection=None, **kwargs):
        self._set_selection(selection)
        with self._data_context:
            check.raise_error_if_not_callable(func, self, *args, **kwargs)
            return func(self, *args, **kwargs)

    return func_with_access


class Refinery:
    def __init__(self, data_context, **kwargs):
        self._data_context = data_context
        self._path = kwargs.get("path") or pathlib.Path.cwd()
        self._repr = kwargs.get("repr", f"({repr(data_context)})")
        self.__post_init__()

    def __post_init__(self):
        # overload this to do extra initialization
        pass

    @classmethod
    def from_data(cls, raw_data):
        """Create the instance directly from the raw data.

        Use this approach when the data is put into the correct format by other means
        than reading from the VASP output files. A typical use case is to read the
        data with `from_path` and then act on it with some postprocessing and pass
        the results to this method.

        Parameters
        ----------
        raw_data
            The raw data required to produce this Refinery.

        Returns
        -------
            A Refinery instance to handle the passed data.
        """
        return cls(_DataWrapper(raw_data), repr=f".from_data({repr(raw_data)})")

    @classmethod
    def from_path(cls, path=None):
        """Read the quantities from the given path.

        The VASP schema determines the particular files accessed. The files will only
        be accessed when the data is required for a particular postprocessing call.

        Parameters
        ----------
        path : str or pathlib.Path
            Path to the directory with the outputs of the VASP calculation. If not
            set explicitly the current directory will be used.

        Returns
        -------
        Refinery
            The returned instance handles opening and closing the files for every
            function called on it.
        """
        repr_ = f".from_path({repr(path)})" if path is not None else ".from_path()"
        return cls(_DataAccess(_quantity(cls), path=path), repr=repr_, path=path)

    @classmethod
    def from_file(cls, file):
        """Read the quantities from the given file.

        You want to use this method if you want to avoid using the Calculation
        wrapper, for example because you renamed the output of the VASP calculation.

        Parameters
        ----------
        file : str or io.BufferedReader
            Filename from which the data is extracted. Alternatively, you can open the
            file yourself and pass the Reader object. In that case, you need to take
            care the file is properly closed again and be aware the generated instance
            of this class becomes unusable after the file is closed.

        Returns
        -------
        Refinery
            The returned instance handles opening and closing the file for every
            function called on it, unless a Reader object is passed in which case this
            is left to the user.

        Notes
        -----
        VASP produces multiple output files whereas this routine will only link to the
        single specified file. Prefer `from_path` routine over this method unless you
        renamed the VASP output files, because `from_path` can collate results from
        multiple files.
        """
        repr_ = f".from_file({repr(file)})"
        path = _get_path_to_file(file)
        return cls(_DataAccess(_quantity(cls), file=file), repr=repr_, path=path)

    @property
    def path(self):
        "Returns the path from which the output is obtained."
        return self._path

    def _set_selection(self, selection):
        if not selection:
            return
        try:
            self._data_context.selection = selection.strip().lower()
        except dataclasses.FrozenInstanceError as error:
            message = f"Creating {self.__class__.__name__}.from_data does not allow to select a source."
            raise exception.IncorrectUsage(message) from error

    @property
    def _raw_data(self):
        return self._data_context.data

    @data_access
    def print(self):
        "Print a string representation of this instance."
        print(str(self))

    def read(self, *args, **kwargs):
        "Convenient wrapper around to_dict. Check that function for examples and optional arguments."
        return self.to_dict(*args, **kwargs)

    def _repr_pretty_(self, p, cycle):
        p.text(str(self))

    def __repr__(self):
        return f"{self.__class__.__name__}{self._repr}"


def _quantity(cls):
    return convert.to_snakecase(cls.__name__)


def _get_path_to_file(file):
    try:
        file = pathlib.Path(file)
    except TypeError:
        file = pathlib.Path(file.name)
    return file.parent


def _do_nothing(*args, **kwargs):
    pass


@dataclasses.dataclass(frozen=True)
class _DataWrapper(contextlib.AbstractContextManager):
    data: raw.VaspData
    source: str = None
    __enter__ = _do_nothing
    __exit__ = _do_nothing


class _DataAccess(contextlib.AbstractContextManager):
    def __init__(self, *args, **kwargs):
        self.selection = None
        self._args = args
        self._kwargs = kwargs
        self._counter = 0
        self._stack = contextlib.ExitStack()

    def __enter__(self):
        if self._counter == 0:
            context = raw.access(*self._args, selection=self.selection, **self._kwargs)
            self.data = self._stack.enter_context(context)
        self._counter += 1
        return self.data

    def __exit__(self, *_):
        self._counter -= 1
        if self._counter == 0:
            self.selection = None
            self._stack.close()
