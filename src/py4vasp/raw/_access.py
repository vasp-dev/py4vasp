import contextlib
from dataclasses import fields, replace
import h5py

DEFAULT_FILE = "vaspout.h5"
schema = "TODO"


@contextlib.contextmanager
def access(quantity):
    source = schema.sources[quantity]["default"]
    with h5py.File(DEFAULT_FILE, "r") as h5f:
        datasets = _get_datasets(h5f, source.data)
        yield replace(source.data, **datasets)


def _get_datasets(h5f, data):
    return {field.name: h5f.get(getattr(data, field.name)) for field in fields(data)}
