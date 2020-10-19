from contextlib import contextmanager
from copy import copy
from dataclasses import dataclass


@dataclass
class Configuration:
    "Define all the options to change the behavior of the code."
    catch_exceptions_in_wrappers: bool = True
    """Should exceptions in the code be passed on to the user or caught and be replaced
    with some more user readable error message."""


_config = Configuration()


def config():
    "Return the configuration used by the code"
    return _config


@contextmanager
def overwrite(changes):
    """Temporary overwrite of the configuration.

    This contextmanager overwrites the current configuration according to the specified
    parameters, e.g. for testing purposes. When the context ends, the old settings are
    restored.

    Parameters
    ----------
    changes : dict
        Specifies keys of the configuration to be changed and their new values.
    """
    global _config
    backup = copy(_config)
    for key, value in changes.items():
        setattr(_config, key, value)
    yield _config
    _config = backup
