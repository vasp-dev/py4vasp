from contextlib import contextmanager, nullcontext
import py4vasp.raw as raw


@contextmanager
def from_file(cls, file, attr):
    if file is None or isinstance(file, str):
        context = raw.File(file)
    else:
        context = nullcontext(file)
    with context as file:
        yield cls(getattr(file, attr)())
