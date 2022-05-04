import contextlib
from dataclasses import fields, replace
import h5py
from py4vasp.raw._schema import Link

DEFAULT_FILE = "vaspout.h5"
schema = "TODO"


@contextlib.contextmanager
def access(quantity, source="default", exit_stack=None):
    source = schema.sources[quantity][source]
    filename = source.file or DEFAULT_FILE
    with _get_exit_stack(exit_stack) as stack:
        h5f = stack.enter_context(h5py.File(filename, "r"))
        datasets = _get_datasets(h5f, source.data, stack)
        yield replace(source.data, **datasets)


def _get_exit_stack(stack):
    if stack is None:
        return contextlib.ExitStack()
    else:
        return contextlib.nullcontext(stack)


def _get_datasets(h5f, data, stack):
    return {
        field.name: _get_dataset(h5f, getattr(data, field.name), stack)
        for field in fields(data)
    }


def _get_dataset(h5f, key, stack):
    if key is None:
        return None
    elif isinstance(key, Link):
        context = access(key.quantity, source=key.source, exit_stack=stack)
        return stack.enter_context(context)
    else:
        return h5f.get(key)
