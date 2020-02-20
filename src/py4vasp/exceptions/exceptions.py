class Py4VaspException(Exception):
    """Base class for all exceptions raised by py4vasp"""


class RefinementException(Py4VaspException):
    """When refining the raw dataclass into the class handling e.g. reading and
    plotting of the data an error occured"""
