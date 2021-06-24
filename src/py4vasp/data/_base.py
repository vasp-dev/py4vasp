import contextlib
import functools
import importlib
import pathlib
import py4vasp.raw as raw
import py4vasp.exceptions as exception
from py4vasp.data._util import minimal_vasp_version


class DataBase:
    _missing_data_message = "The raw data is None, please check your setup."

    def __init__(self, raw_data):
        self._from_context_generator(lambda: contextlib.nullcontext(raw_data))
        self._repr = f"({repr(raw_data)})"

    @classmethod
    def from_file(cls, file=None):
        """Read the raw data from the given file.

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
        obj._from_context_generator(lambda: _from_file(file, cls.__name__.lower()))
        obj._repr = f".from_file({repr(file)})"
        return obj

    def _from_context_generator(self, context_generator):
        self._raw_data_from_context = context_generator

    def __repr__(self):
        return f"{self.__class__.__name__}{self._repr}"

    def _repr_pretty_(self, p, cycle):
        p.text(str(self))

    def print(self):
        print(self)


@contextlib.contextmanager
def _from_file(file, name):
    if file is None or isinstance(file, str) or isinstance(file, pathlib.Path):
        context = raw.File(file)
    else:
        context = contextlib.nullcontext(file)
    with context as file:
        yield getattr(file, name)()


class RefinementDescriptor:
    def __init__(self, name):
        self._name = name

    def __get__(self, instance, type_):
        module = importlib.import_module(type_.__module__)
        function = getattr(module, self._name)

        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            with instance._raw_data_from_context() as raw_data:
                raise_error_if_data_is_missing(raw_data, instance._missing_data_message)
                raise_error_if_version_is_outdated(raw_data)
                return function(raw_data, *args, **kwargs)

        return wrapper


def raise_error_if_data_is_missing(raw_data, error_message):
    if raw_data is None:
        raise exception.NoData(error_message)


def raise_error_if_version_is_outdated(raw_data):
    if raw_data.version < minimal_vasp_version:
        raise exception.OutdatedVaspVersion(
            "To use py4vasp features, you need at least Vasp version "
            f"{minimal_vasp_version.major}.{minimal_vasp_version.minor}."
            f"{minimal_vasp_version.patch}. The used version is "
            f"{raw_data.version.major}.{raw_data.version.minor}."
            f"{raw_data.version.patch}. Please use a newer version of Vasp."
        )
