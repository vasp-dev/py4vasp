from contextlib import contextmanager, nullcontext
from typing import NamedTuple, Iterable
import py4vasp.raw as raw
import functools


def from_file_doc(doc):
    return """Read the {} from the given file.

        Parameters
        ----------
        file : str or raw.File
            Filename from which the data is extracted, using {} if not present.
            Alternatively, you can open the file yourself and pass the `File`
            object. In that case, you need to take care the file is properly
            closed again and be aware the generated instance of this class
            becomes unusable after the file is closed.

        Yields
        ------
        contextmanager
            The returned context manager handles opening and closing the file.
            If a `File` object is passed a `nullcontext` is returned.
        """.format(
        doc, raw.File.default_filename
    )


@contextmanager
def from_file(cls, file, attr):
    if file is None or isinstance(file, str):
        context = raw.File(file)
    else:
        context = nullcontext(file)
    with context as file:
        yield cls(getattr(file, attr)())


def add_doc(doc):
    def add_doc_to_func(func):
        func.__doc__ = doc
        return func

    return add_doc_to_func


def decode_if_possible(string):
    try:
        return string.decode()
    except (UnicodeDecodeError, AttributeError):
        return string


default_selection = "*"


class Selection(NamedTuple):
    indices: Iterable[int]
    label: str = ""


def add_specific_wrappers(specific_wrappers={}):
    default_wrappers = {"read": "to_dict", "plot": "to_plotly"}
    actual_wrappers = {**default_wrappers, **specific_wrappers}

    def add_wrappers_decorator(cls):
        for wrapping, wrapped in actual_wrappers.items():
            if hasattr(cls, wrapped):
                setattr(cls, wrapping, _make_wrapper(cls, wrapped))
        return cls

    return add_wrappers_decorator


def _make_wrapper(cls, wrap_this_func):
    @functools.wraps(getattr(cls, wrap_this_func))
    def wrapper(self, *args, **kwargs):
        this_func = getattr(self, wrap_this_func)
        return this_func(*args, **kwargs)

    return wrapper


add_wrappers = add_specific_wrappers()
