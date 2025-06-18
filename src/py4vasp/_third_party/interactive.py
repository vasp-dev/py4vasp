# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import pathlib
import traceback

import py4vasp
from py4vasp import exception
from py4vasp._util import import_

IPython = import_.optional("IPython")
_ERROR_VERBOSITY = "not set"
_ALLOWED_VERBOSITIES = ["Inherit", "Plain", "Minimal"]


def set_error_handling(verbosity):
    """Set the error handling verbosity for IPython.

    Parameters
    ----------
    verbosity : str
        The verbosity level for error handling. Allowed values are:
        - "Inherit": Inherit the error handling from IPython.
        - "Plain": Use plain error messages (traceback for user code).
        - "Minimal": Use minimal error messages (no traceback).
    """
    if verbosity not in _ALLOWED_VERBOSITIES:
        raise exception.NotImplemented(
            f"Error handling mode '{verbosity}' is not supported. "
            f"Allowed modes are: {_ALLOWED_VERBOSITIES}."
        )
    global _ERROR_VERBOSITY
    _ERROR_VERBOSITY = verbosity
    ipython = _get_ipython()
    if ipython is not None:
        custom_exceptions = () if verbosity == "Inherit" else (exception.Py4VaspError,)
        ipython.set_custom_exc(custom_exceptions, _handle_exception)


def error_handling():
    return _ERROR_VERBOSITY


def _get_ipython():
    if import_.is_imported(IPython):
        return IPython.get_ipython()
    else:
        return None


def handle_exception(exception):
    _handle_exception(
        _get_ipython(), type(exception), exception, exception.__traceback__
    )


def _handle_exception(shell, etype, evalue, tb, tb_offset=0):
    if shell is not None:
        tb_offset = tb_offset or shell.InteractiveTB.tb_offset
    traceback_formatter = IPython.core.ultratb.FormattedTB(
        _ERROR_VERBOSITY, tb_offset=tb_offset
    )
    frames = traceback.extract_tb(tb, limit=None)
    frames_outside_py4vasp = list(_keep_frames_outside_py4vasp(frames))
    traceback_formatter(etype, evalue, frames_outside_py4vasp)


def _keep_frames_outside_py4vasp(frames):
    module_root = pathlib.Path(py4vasp.__file__).parent.resolve()
    for frame in frames:
        if pathlib.Path(frame.filename).resolve().is_relative_to(module_root):
            return
        yield frame
