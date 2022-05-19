# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import contextlib
import dataclasses
from py4vasp import raw


class Refinery:
    def __init__(self, data_context):
        self._data_context = data_context
        self.__post_init__()

    def __post_init__(self):
        # overload this to do extra initialization
        pass

    @classmethod
    def from_data(cls, raw_data):
        return cls(_DataWrapper(raw_data))

    @classmethod
    def from_path(cls, path=None):
        return cls(_DataAccess(cls.__name__.lower(), path=path))

    @classmethod
    def from_file(cls, file):
        return cls(_DataAccess(cls.__name__.lower(), file=file))

    def access(func):
        def func_with_access(self, *args, source=None, **kwargs):
            self._set_source(source)
            with self._data_context:
                return func(self, *args, **kwargs)

        return func_with_access

    def _set_source(self, source):
        if not source:
            return
        try:
            self._data_context.source = source
        except dataclasses.FrozenInstanceError as error:
            message = f"Creating {self.__class__.__name__}.from_data does not allow to specify a source."
            raise exception.IncorrectUsage(message) from error

    @property
    def _raw_data(self):
        return self._data_context.data

    def print(self):
        print(str(self))


def do_nothing(*args, **kwargs):
    pass


@dataclasses.dataclass(frozen=True)
class _DataWrapper(contextlib.AbstractContextManager):
    data: raw.VaspData
    source: str = None
    __enter__ = do_nothing
    __exit__ = do_nothing


class _DataAccess(contextlib.AbstractContextManager):
    def __init__(self, *args, **kwargs):
        self.source = None
        self._args = args
        self._kwargs = kwargs
        self._counter = 0
        self._stack = contextlib.ExitStack()

    def __enter__(self):
        if self._counter == 0:
            context = raw.access(*self._args, source=self.source, **self._kwargs)
            self.data = self._stack.enter_context(context)
        self._counter += 1
        return self.data

    def __exit__(self, *_):
        self._counter -= 1
        if self._counter == 0:
            self.source = None
            self._stack.close()


# ---------------------------------------------------------------------------------------
import contextlib
import functools
import pathlib
import py4vasp.raw as raw
import py4vasp.exceptions as exception
from py4vasp._util.version import minimal_vasp_version, current_vasp_version
import py4vasp._util.convert as _convert


class DataBase:
    _missing_data_message = "The raw data is None, please check your setup."

    def __init__(self, raw_data):
        data_dict = raw.DataDict({"default": raw_data}, current_vasp_version)
        self._from_context_generator(lambda: contextlib.nullcontext(data_dict))
        self._repr = f"({repr(raw_data)})"
        self._path = pathlib.Path.cwd()
        self._initialize()

    @classmethod
    def from_dict(cls, dict_):
        """Initialize refinement class from data dictionary

        Parameters
        ----------
        data_dict : dict
            Data dictionary that contains one or more different raw data sources.
        """
        obj = cls.__new__(cls)
        data_dict = raw.DataDict(dict_, current_vasp_version)
        obj._from_context_generator(lambda: contextlib.nullcontext(data_dict))
        obj._repr = f".from_dict({repr(data_dict)})"
        obj._path = pathlib.Path.cwd()
        return obj._initialize()

    @classmethod
    def from_file(cls, file=None):
        """Read the data dictionary from the given file.

        You want to use this method if you want to avoid using the Calculation
        wrapper, for example because you renamed the output of the VASP calculation.

        Parameters
        ----------
        file : str or Path or raw.File
            Filename from which the data is extracted. If not present the default
            filename is used. Alternatively, you can open the file yourself and pass the
            `File` object. In that case, you need to take care the file is properly
            closed again and be aware the generated instance of this class becomes
            unusable after the file is closed.

        Returns
        -------
        DataBase
            The returned instance handles opening and closing the file for every
            function called on it, unless a `File` object in which case this is left to
            the user.
        """
        obj = cls.__new__(cls)
        name = _convert.to_snakecase(cls.__name__)
        obj._from_context_generator(lambda: _from_file(file, name))
        obj._repr = f".from_file({repr(file)})"
        obj._path = _get_absolute_path(file)
        return obj._initialize()

    def _from_context_generator(self, context_generator):
        self._data_dict_from_context = context_generator

    def _initialize(self):
        # overload this to do extra initialization
        return self

    def __repr__(self):
        return f"{self.__class__.__name__}{self._repr}"

    def _repr_pretty_(self, p, cycle):
        p.text(str(self))

    def print(self):
        "Print a string representation of this class to standard output."
        print(self)

    def _set_data_or_raise_error_if_data_is_missing(self, raw_data):
        if raw_data is not None:
            self._raw_data = raw_data
        else:
            raise exception.NoData(self._missing_data_message)


@contextlib.contextmanager
def _from_file(file, name):
    if file is None or isinstance(file, str) or isinstance(file, pathlib.Path):
        context = raw.File(file)
    else:
        context = contextlib.nullcontext(file)
    with context as file:
        yield getattr(file, name)


def _get_absolute_path(file):
    if file is None:
        return pathlib.Path.cwd()
    elif isinstance(file, str) or isinstance(file, pathlib.Path):
        absolute_path = pathlib.Path(file).resolve()
        return absolute_path if absolute_path.is_dir() else absolute_path.parent
    else:
        return file.path


class RefinementDescriptor:
    def __init__(self, name):
        self._name = name

    def __get__(self, instance, type_):
        function = getattr(type_, self._name)

        @functools.wraps(function)
        def wrapper(*args, source="default", **kwargs):
            with instance._data_dict_from_context() as data_dict:
                raise_error_if_version_is_outdated(data_dict.version)
                raw_data = read_raw_data_from_source(data_dict, source)
                instance._set_data_or_raise_error_if_data_is_missing(raw_data)
                return function(instance, *args, **kwargs)

        return wrapper


def read_raw_data_from_source(data_dict, source):
    source = source.strip().lower()
    try:
        return data_dict[source]
    except KeyError as err:
        message = f"""The source {source} could not be found in the data. Please check the spelling.
The following sources are available: {", ".join(data_dict.keys())}."""
        raise exception.IncorrectUsage(message) from err


def raise_error_if_version_is_outdated(version_data):
    if version_data < minimal_vasp_version:
        raise exception.OutdatedVaspVersion(
            "To use py4vasp features, you need at least Vasp version "
            f"{minimal_vasp_version.major}.{minimal_vasp_version.minor}."
            f"{minimal_vasp_version.patch}. The used version is "
            f"{version_data.major}.{version_data.minor}."
            f"{version_data.patch}. Please use a newer version of Vasp."
        )
