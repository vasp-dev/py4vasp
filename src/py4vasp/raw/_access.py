# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import contextlib
import dataclasses
import functools
import h5py
import numpy as np
import pathlib
import textwrap
import py4vasp.exceptions as exception
import py4vasp.raw as raw
from py4vasp.raw._definition import schema, DEFAULT_FILE, DEFAULT_SOURCE
from py4vasp.raw._schema import Link, Length


@contextlib.contextmanager
def _access(quantity, *, source=None, path=None, file=None):
    """Create access to a particular quantity from the VASP output.

    Parameters
    ----------
    quantity : str
        Select which particular quantity to access.
    source : str, optional
        Keyword-only argument to select different sources of the quantity.
    path : str, optional
        Keyword-only argument to set the path from which VASP output is read. Defaults
        to the current directory.
    file : str, optional
        Keyword-only argument to set the file from which VASP output is read. Defaults
        are set in the schema.

    Returns
    -------
    ContextManager
        Entering the context manager results in access to the desired quantity. Note
        that the access terminates at the end of the context to ensure all VASP files
        are properly closed.
    """
    state = _State(path, file)
    with state.exit_stack:
        yield state.access(quantity, source)


@functools.wraps(_access)
def access(*args, **kwargs):
    try:
        return _access(*args, **kwargs)
    except TypeError as error:
        message = (
            "The arguments to the function are incorrect. Please use keywords for all "
            "arguments except for the first."
        )
        raise exception.IncorrectUsage(message) from error


class _State:
    def __init__(self, path, file):
        self.exit_stack = contextlib.ExitStack()
        self._files = {}
        self._path = path or pathlib.Path(".")
        self._file = file

    def access(self, quantity, source):
        source = self._get_source(quantity, source)
        filename = self._file or source.file or DEFAULT_FILE
        path = self._path / pathlib.Path(filename)
        h5f = self._open_file(path)
        self._check_version(h5f, source.required, quantity)
        datasets = self._get_datasets(h5f, source.data)
        return dataclasses.replace(source.data, **datasets)

    def _get_source(self, quantity, source):
        source = source or DEFAULT_SOURCE
        try:
            return schema.sources[quantity][source]
        except KeyError as error:
            raise exception.FileAccessError(_error_message(quantity, source)) from error

    def _open_file(self, filename):
        if filename in self._files:
            return self._files[filename]
        else:
            file = self._create_and_enter_context(filename)
            self._files[filename] = file
            return file

    def _create_and_enter_context(self, filename):
        try:
            h5f = h5py.File(filename, "r")
        except FileNotFoundError as error:
            message = (
                f"{filename} could not be opened. Please make sure the file exists."
            )
            raise exception.FileAccessError(message) from error
        except OSError as error:
            message = (
                f"Error when reading from {filename}. Please check whether the file "
                "format is correct and you have the permissions to read it."
            )
            raise exception.FileAccessError(message)
        return self.exit_stack.enter_context(h5f)

    def _check_version(self, h5f, required, quantity):
        if not required:
            return
        version = raw.Version(
            major=h5f[schema.version.major][()],
            minor=h5f[schema.version.minor][()],
            patch=h5f[schema.version.patch][()],
        )
        if version < required:
            message = f"The {quantity} is not available in VASP {version}. It requires at least {required}."
            raise exception.OutdatedVaspVersion(message)

    def _get_datasets(self, h5f, data):
        return {
            field.name: self._get_dataset(h5f, getattr(data, field.name))
            for field in dataclasses.fields(data)
        }

    def _get_dataset(self, h5f, key):
        if key is None:
            return None
        if isinstance(key, Link):
            return self.access(key.quantity, source=key.source)
        if isinstance(key, Length):
            dataset = h5f.get(key.dataset)
            return len(dataset) if dataset else None
        return self._parse_dataset(h5f.get(key))

    def _parse_dataset(self, dataset):
        result = raw.VaspData(dataset)
        if _is_scalar(result):
            result = result[()]
        return result


def _is_scalar(data):
    return not data.is_none() and data.ndim == 0


def _error_message(quantity, source):
    if quantity in schema.sources:
        sources = schema.sources[quantity]
        first_part = f"""\
            py4vasp did not understand your input! The code executed requires to access
            the source="{source}" of the quantity "{quantity}". Perhaps there is a
            spelling mistake in the source? Please, compare the spelling of the source
            "{source}" with the sources py4vasp knows about "{'", "'.join(sources)}"."""
    else:
        first_part = f"""\
            py4vasp does not know how to access the quantity "{quantity}". Perhaps there
            is a spelling mistake? Please, compare the spelling of the quantity "{quantity}"
            with the quantities py4vasp knows about "{'", "'.join(schema.sources)}"."""
    second_part = """\
        It is also possible that this feature was only added in a later version of
        py4vasp, so please check that you use the most recent version."""
    message = textwrap.dedent(first_part) + " " + textwrap.dedent(second_part)
    return "\n" + "\n".join(textwrap.wrap(message, width=80))
