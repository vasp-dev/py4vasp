from contextlib import contextmanager, nullcontext
from typing import NamedTuple, Iterable
import py4vasp.raw as raw


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
