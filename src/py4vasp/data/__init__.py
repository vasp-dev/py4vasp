""" Refine the raw data produced by Vasp for plotting or analysis.

Usually one is not directly interested in the raw data that is produced, but
wants to produce either a figure for a publication or some post processing of
the data. This module contains multiple classes that enable these kind of
workflows by extracting the relevant data from the HDF5 file and transforming
them into an accessible format. The classes also provide plotting functionality
to get a quick insight about the data, which can then be refined either within
python or a different tool to obtain publication quality figures.

Generally, all classes provide a `read` function that extracts the data from the
HDF5 file and puts it into a python dictionary. Where it makes sense in addition
a `plot` function is available that converts the data into a figure for Jupyter
notebooks. In addition, data conversion routines `to_X` may be available
transforming the data into another format or file, which may be useful to
generate plots with tools other than python. For the specifics, please refer to
the documentation of the individual classes.
"""

from .band import Band
from .dos import Dos
from .energy import Energy
from .kpoints import Kpoints
from .magnetism import Magnetism
from .projectors import Projectors
from .topology import Topology
from .viewer3d import Viewer3d
from .structure import Structure
from .density import Density
from .trajectory import Trajectory
from py4vasp.config import config

import py4vasp.exceptions as exception
import plotly.io as pio
import cufflinks as cf
import inspect
import itertools
import textwrap
import sys

pio.templates.default = "ggplot2"
cf.go_offline()
cf.set_config_file(theme="ggplot")

_this_mod = sys.modules[__name__]
_class_names = [name for name, _ in inspect.getmembers(_this_mod, inspect.isclass)]
_classes = [value for _, value in inspect.getmembers(_this_mod, inspect.isclass)]
_functions = set(("read", "plot"))
for c in _classes:
    for name, _ in inspect.getmembers(c, inspect.isfunction):
        if "to_" in name:
            _functions.add(name)
__all__ = _class_names + list(_functions)


def _generate_documentation(classes, function):
    class_list = ""
    example = ""
    for c in classes:
        if hasattr(c, function):
            class_list += "\n* :class:`{}`".format(c.__name__)
            example = example or c.__name__
    return """This function wraps the {0} method of the data classes.

{0} is a function available for the following classes:
{1}

This wrapper deals with opening the Vasp output file, reading the relevant data
and closing the file after reading. As such it is particularly suited for simple
access. More advanced users in particular when using the data in a python script
may want to control the opening of the file explicitly for better efficiency and
more flexibility.

Parameters
----------
cls : Class
    Choice of quantity of which the {0} function is evaluated.

Notes
-----
The remaining arguments passed to this routine are directly passed on to the {0}
function of the selected class. Please check the help of the individual classes
to know which arguments are available and what exactly is returned, e.g.

>>> help({2}.{0})
""".format(
        function, class_list, example
    )


def _wrapper_factory(module, name, documentation):
    @_util.add_doc(documentation)
    def wrapper(cls, *args, **kwargs):
        try:
            with cls.from_file() as obj:
                function = _get_function_if_possible(obj, name)
                return _evaluate_if_possible(function, *args, **kwargs)
        except Exception as err:
            _handle_exception(err)

    setattr(module, name, wrapper)


def _get_function_if_possible(obj, name):
    try:
        return getattr(obj, name)
    except AttributeError as err:
        class_ = obj.__class__.__name__
        msg = "For the {} no {} function is implemented.".format(class_, name)
        raise exception.NotImplemented(msg) from err


def _evaluate_if_possible(function, *args, **kwargs):
    try:
        return function(*args, **kwargs)
    except TypeError as err:
        msg = (
            f"Error in function call: {err.args[0]}. Please check the function "
            f"documentation help({function.__qualname__}) to check for possible errors."
        )
        raise exception.IncorrectUsage(msg) from err


def _handle_exception(err):
    if config().catch_exceptions_in_wrappers:
        raw_message = f"{err.__class__.__name__}: {err.args[0]}"
        if isinstance(err, exception.Py4VaspError):
            msg = _format(raw_message, _header_error, _footer_error)
        else:
            msg = _format(raw_message, _header_bug, _footer_bug)
        sys.stderr.write(msg)
        raise exception.StopExecution
    else:
        raise err


_header_error = "\n" + (
    "EEEEEEE  RRRRRR   RRRRRR   OOOOOOO  RRRRRR      ###     ###     ###\n"
    "E        R     R  R     R  O     O  R     R     ###     ###     ###\n"
    "E        R     R  R     R  O     O  R     R     ###     ###     ###\n"
    "EEEEE    RRRRRR   RRRRRR   O     O  RRRRRR       #       #       # \n"
    "E        R   R    R   R    O     O  R   R                          \n"
    "E        R    R   R    R   O     O  R    R      ###     ###     ###\n"
    "EEEEEEE  R     R  R     R  OOOOOOO  R     R     ###     ###     ###\n\n"
)
_footer_error = (
    "\n\n  ---->  I REFUSE TO CONTINUE WITH THIS SICK JOB ... BYE!!! <----\n"
)

_header_bug = (
    "                _     ____    _    _    _____     _ \n"
    "               | |   |  _ \\  | |  | |  / ____|   | |\n"
    "               | |   | |_) | | |  | | | |  __    | |\n"
    "               |_|   |  _ <  | |  | | | | |_ |   |_|\n"
    "                _    | |_) | | |__| | | |__| |    _ \n"
    "               (_)   |____/   \\____/   \\_____|   (_)\n\n"
)
_footer_bug = (
    "\n\nIf you are not a developer, you should not encounter this problem.\n"
    "Please submit a bug report.\n"
)


def _format(message, header, footer):
    width = 67
    padding = " " * 5
    horizontal_line = " " + "-" * (width + 2 * len(padding))
    lines = (header + message + footer).split("\n")
    wrap_line = lambda line: textwrap.wrap(line, width) if line.strip() else [""]
    wrap_lines = itertools.chain.from_iterable(wrap_line(line) for line in lines)
    pad_lines = (f"|{padding}{line:67}{padding}|" for line in wrap_lines)
    all_lines = (horizontal_line, *pad_lines, horizontal_line)
    return "\n".join(all_lines)


for function in _functions:
    documentation = _generate_documentation(_classes, function)
    _wrapper_factory(_this_mod, function, documentation)
