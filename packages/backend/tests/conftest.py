# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
"""Skip these tests when vasp-backend is not part of the environment.

`testpaths` in the root pyproject.toml covers `packages`, so a bare `pytest` also
collects this directory. In a py4vasp-core-only environment vasp-backend is absent and
importing the test modules would abort collection instead of skipping.
"""

import importlib.util


def _is_installed():
    try:
        # find_spec imports the parent package, which does not exist either.
        return importlib.util.find_spec("vasp.backend") is not None
    except ModuleNotFoundError:
        return False


collect_ignore_glob = [] if _is_installed() else ["test_*.py"]
