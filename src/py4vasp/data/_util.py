from contextlib import contextmanager
import py4vasp.raw as raw


@contextmanager
def from_file(cls, file, attr):
    if file is None or isinstance(file, str):
        with raw.File(file) as local_file:
            yield cls(getattr(local_file, attr)())
    else:
        yield cls(getattr(file, attr)())
