# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
from unittest.mock import patch

import py4vasp._third_party.interactive as interactive


def test_no_error_handling_outside_ipython():
    assert interactive.error_handling() == "Minimal"  # default
    interactive.set_error_handling("Plain")
    assert interactive.error_handling() == "Plain"


def test_set_error_handling(not_core):
    import IPython

    shell = IPython.terminal.interactiveshell.TerminalInteractiveShell()
    with patch("IPython.get_ipython", return_value=shell) as mock:
        assert shell.InteractiveTB.mode != "Minimal"
        interactive.set_error_handling("Minimal")
        assert shell.InteractiveTB.mode == "Minimal"
