# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
"""Deals with the possible exceptions in py4vasp.

The design goal is that all foreseeable exceptions in py4vasp issue an exception of the Py4VaspException class. Any other kind of exception would indicate a bug in the code. If possible the part standard users interact with should not raise any exception, but should give advice on how to overcome the issue."""


class Py4VaspError(Exception):
    """Base class for all exceptions raised by py4vasp"""


class RefinementError(Py4VaspError):
    """When refining the raw dataclass into the class handling e.g. reading and
    plotting of the data an error occurred"""


class IncorrectUsage(Py4VaspError):
    """The user provided input is not suitable for processing"""


class NotImplemented(Py4VaspError):
    """Exception raised when a function is called that is not implemented."""


class NoData(Py4VaspError):
    """Exception raised when certain data is not present, because the corresponding
    INCAR flags have not been set."""


class FileAccessError(Py4VaspError):
    """Exception raised when error occurs during accessing the HDF5 file."""


class OutdatedVaspVersion(Py4VaspError):
    """Exception raised when the py4vasp features used are not available in the
    used version of Vasp."""


class MissingAttribute(Py4VaspError, AttributeError):
    """Exception raised when py4vasp attribute of Calculation, Batch, ... is used
    that does not exist"""


class ModuleNotInstalled(Py4VaspError):
    """Exception raised when a functionality is used that relies on an optional
    dependency of py4vasp but that dependency is not installed."""


class StopExecution(Py4VaspError):
    """Exception raised when an error occurred in the user interface. This prevents
    further cells from being executed."""

    def _render_traceback_(self):
        "This exception is silent and does not produce any traceback."
        pass


class ParserError(Py4VaspError):
    """Exception raised when the parser encounters an error."""


class _Py4VaspInternalError(Exception):
    """This error should not propagate to the user. It should be raised when a local
    routine encounters unexpected behavior that the routine cannot deal with. Then
    the calling routine should resolve the error or reraise it as a Py4VaspError."""
