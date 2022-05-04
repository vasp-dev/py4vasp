import contextlib
from dataclasses import fields, replace
import h5py

DEFAULT_FILE = "vaspout.h5"
schema = "TODO"


@contextlib.contextmanager
def access(quantity, source="default"):
    source = schema.sources[quantity][source]
    filename = source.file or DEFAULT_FILE
    with h5py.File(filename, "r") as h5f:
        datasets = _get_datasets(h5f, source.data)
        yield replace(source.data, **datasets)


def _get_datasets(h5f, data):
    return {
        field.name: _get_dataset(h5f, getattr(data, field.name))
        for field in fields(data)
    }


def _get_dataset(h5f, key):
    if key is None:
        return None
    else:
        return h5f.get(key)
