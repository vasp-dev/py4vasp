# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import subprocess
import sys

from py4vasp import broadening
from py4vasp._third_party import numeric


def run_in_a_fresh_interpreter(code):
    # a subprocess, because both properties below are about what importing py4vasp does;
    # inside this process py4vasp is long imported and scipy may be too
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)


def test_public_names():
    assert broadening.__all__ == ["broaden", "Gaussian", "Lorentzian"]
    for name in broadening.__all__:
        assert getattr(broadening, name) is getattr(numeric, name)


def test_broadening_is_reachable_after_importing_py4vasp():
    # the documentation lists broadening beside plot and exception, so it has to resolve
    # the same way those do; every doctest imports from py4vasp.broadening directly and
    # would not notice
    result = run_in_a_fresh_interpreter(
        "import py4vasp; py4vasp.broadening.Gaussian(fwhm=1.0)"
    )
    assert result.returncode == 0, result.stderr


def test_broadening_does_not_import_scipy():
    # broadening must work on the core installation; it shares a module with the
    # interpolation routines that do need scipy, and those are imported lazily
    result = run_in_a_fresh_interpreter(
        "import sys\n"
        "from py4vasp.broadening import Gaussian, broaden\n"
        "broaden([0.0, 1.0, 2.0], [1.0], shape=Gaussian(fwhm=1.0))\n"
        "imported = [name for name in sys.modules if name.startswith('scipy')]\n"
        "assert not imported, imported\n"
    )
    assert result.returncode == 0, result.stderr
