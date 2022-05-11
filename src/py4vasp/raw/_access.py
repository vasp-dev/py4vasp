import contextlib
import dataclasses
import h5py
import pathlib
import py4vasp
import py4vasp.exceptions as exception
import py4vasp.raw as raw
from py4vasp.raw._definition import schema, DEFAULT_FILE
from py4vasp.raw._schema import Link, Length


@contextlib.contextmanager
def access(quantity, source="default", path=None, file=None):
    state = _State(path, file)
    with state.exit_stack:
        yield state.access(quantity, source)


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
        try:
            return schema.sources[quantity][source]
        except KeyError as error:
            message = (
                f"{quantity}/{source} is not available in the HDF5 file. Please check "
                + "the spelling of the arguments. Perhaps the version of py4vasp "
                + f"({py4vasp.__version__}) is not up to date with the documentation."
            )
            raise exception.FileAccessError(message)

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
        return self.exit_stack.enter_context(h5f)

    def _check_version(self, h5f, required, quantity):
        if not required:
            return
        version = raw.Version(
            major=h5f[schema.version.major],
            minor=h5f[schema.version.minor],
            patch=h5f[schema.version.patch],
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
        if dataset is None:
            return None
        if dataset.ndim > 0:
            return raw.VaspData(dataset)
        return self._parse_scalar(dataset[()])

    def _parse_scalar(self, scalar):
        if isinstance(scalar, bytes):
            return scalar.decode()
        else:
            return scalar
