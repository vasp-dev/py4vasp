# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
""" Deals with the possible exceptions in py4vasp.

The design goal is that all forseeable exceptions in py4vasp issue an exception
of the Py4VaspException class. Any other kind of exception would indicate a bug
in the code. If possible the part standard users interact with should not raise
any exception, but should give advice on how to overcome the issue.
"""

from .exceptions import *
import inspect
import sys

_this_mod = sys.modules[__name__]
__all__ = [name for name, _ in inspect.getmembers(_this_mod, inspect.isclass)]
