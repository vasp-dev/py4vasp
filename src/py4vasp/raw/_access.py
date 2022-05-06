import contextlib
import dataclasses
import h5py
import pathlib
import py4vasp.exceptions as exception
import py4vasp.raw as raw
from py4vasp.raw._schema import Link

DEFAULT_FILE = "vaspout.h5"
schema = "TODO"
from py4vasp.raw._schema import Link, Length


@contextlib.contextmanager
def access(quantity, source="default", path=None):
    state = _State(path)
    with state.exit_stack:
        yield state.access(quantity, source)


class _State:
    def __init__(self, path):
        self.exit_stack = contextlib.ExitStack()
        self._files = {}
        self._path = path or pathlib.Path(".")

    def access(self, quantity, source):
        source = schema.sources[quantity][source]
        filename = source.file or DEFAULT_FILE
        path = self._path / pathlib.Path(filename)
        h5f = self._open_file(path)
        self._check_version(h5f, source.required, quantity)
        datasets = self._get_datasets(h5f, source.data)
        return dataclasses.replace(source.data, **datasets)

    def _open_file(self, filename):
        if filename in self._files:
            return self._files[filename]
        else:
            file = self.exit_stack.enter_context(h5py.File(filename, "r"))
            self._files[filename] = file
            return file

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
