# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
"""Run the command line interface with ``python -m py4vasp``.

The ``py4vasp`` console script is declared by the *py4vasp* distribution because
:mod:`py4vasp.cli` imports click unconditionally. This module gives the same access to
anyone who installed ``py4vasp-core[cli]`` instead.
"""

from py4vasp.cli import cli

if __name__ == "__main__":
    cli()
