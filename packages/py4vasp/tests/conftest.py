# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
"""Skip these tests when the py4vasp wrapper distribution is not installed.

`testpaths` in the root pyproject.toml covers `packages`, so a bare `pytest` also
collects this directory. A py4vasp-core-only environment provides `import py4vasp` but
not the `py4vasp` distribution whose metadata these tests inspect.
"""

import importlib.metadata

try:
    importlib.metadata.distribution("py4vasp")
    collect_ignore_glob = []
except importlib.metadata.PackageNotFoundError:
    collect_ignore_glob = ["test_*.py"]
