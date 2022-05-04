import contextlib
import h5py

DEFAULT_FILE = "vaspout.h5"
schema = "TODO"


@contextlib.contextmanager
def access(quantity):
    with h5py.File(DEFAULT_FILE, "r") as h5f:
        yield
