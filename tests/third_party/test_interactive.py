# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import re
from unittest.mock import patch

import pytest

import py4vasp._third_party.interactive as interactive
from py4vasp import exception
from py4vasp._calculation.dos import Dos


@pytest.fixture
def ipython(not_core):
    """Mock IPython for testing purposes."""
    import IPython

    shell = IPython.terminal.interactiveshell.TerminalInteractiveShell()
    with patch("IPython.get_ipython", return_value=shell):
        yield shell


def test_no_error_handling_outside_ipython():
    assert interactive.error_handling() == "Plain"  # default
    interactive.set_error_handling("Minimal")
    assert interactive.error_handling() == "Minimal"


def test_py4vasp_does_not_affect_ipython_error_handling(ipython):
    assert ipython.InteractiveTB.mode != "Minimal"
    old_mode = ipython.InteractiveTB.mode
    interactive.set_error_handling("Minimal")
    # Check that py4vasp does not change the global error handling
    assert ipython.InteractiveTB.mode == old_mode


@pytest.mark.parametrize("mode", ["Plain", "Minimal"])
def test_py4vasp_defines_custom_error_handling(ipython, mode):
    interactive.set_error_handling(mode)
    assert interactive.error_handling() == mode
    assert ipython.custom_exceptions == (exception.Py4VaspError,)


def test_py4vasp_inherits_error_handling(ipython):
    interactive.set_error_handling("Inherit")
    assert interactive.error_handling() == "Inherit"
    assert ipython.custom_exceptions == ()


def test_py4vasp_error_handling(raw_data, capsys):
    interactive.set_error_handling("Plain")
    raw_dos = raw_data.dos("Sr2TiO4 with_projectors")
    try:
        Dos.from_data(raw_dos).read("SR")  # This should raise an error
    except exception.Py4VaspError as error:
        interactive.handle_exception(error)
    standard_output, _ = capsys.readouterr()
    assert re.search(r"src[\\/]py4vasp", standard_output) is None


@pytest.mark.parametrize("mode", ["Context", "Verbose"])
def test_verbose_modes_not_supported(mode):
    interactive.set_error_handling("Plain")
    with pytest.raises(exception.NotImplemented):
        interactive.set_error_handling(mode)
    assert interactive.error_handling() == "Plain"  # default remains unchanged
