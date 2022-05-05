import contextlib
import dataclasses
import h5py
import pathlib
from py4vasp.raw._schema import Link

DEFAULT_FILE = "vaspout.h5"
schema = "TODO"


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
        datasets = self._get_datasets(h5f, source.data)
        return dataclasses.replace(source.data, **datasets)

    def _open_file(self, filename):
        if filename in self._files:
            return self._files[filename]
        else:
            file = self.exit_stack.enter_context(h5py.File(filename, "r"))
            self._files[filename] = file
            return file

    def _get_datasets(self, h5f, data):
        return {
            field.name: self._get_dataset(h5f, getattr(data, field.name))
            for field in dataclasses.fields(data)
        }

    def _get_dataset(self, h5f, key):
        if key is None:
            return None
        elif isinstance(key, Link):
            return self.access(key.quantity, source=key.source)
        else:
            dataset = h5f.get(key)
            if dataset.ndim == 0:
                return dataset[()]
            else:
                return raw.VaspData(dataset)
