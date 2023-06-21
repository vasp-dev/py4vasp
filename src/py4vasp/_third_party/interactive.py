# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import contextlib
import os

from py4vasp._util import import_

IPython = import_.optional("IPython")
_ERROR_VERBOSITY = "Minimal"


def set_error_handling(verbosity):
    global _ERROR_VERBOSITY
    ipython = _get_ipython()
    if ipython is None:
        _ERROR_VERBOSITY = verbosity
    else:
        with open(os.devnull, "w") as ignore, contextlib.redirect_stdout(ignore):
            ipython.magic(f"xmode {verbosity}")


def error_handling():
    ipython = _get_ipython()
    if ipython is None:
        return _ERROR_VERBOSITY
    else:
        return ipython.xmode


def _get_ipython():
    if import_.is_imported(IPython):
        return IPython.get_ipython()
    else:
        return None
