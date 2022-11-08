# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import contextlib
import os

import IPython


def set_error_handling(verbosity):
    ipython = IPython.get_ipython()
    if ipython is not None:
        with open(os.devnull, "w") as ignore, contextlib.redirect_stdout(ignore):
            ipython.magic(f"xmode {verbosity}")
