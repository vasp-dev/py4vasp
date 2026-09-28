# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import numpy as np

from py4vasp import broadening
from py4vasp._third_party import numeric
from py4vasp._util import import_


def test_public_names():
    assert broadening.__all__ == ["broaden", "Gaussian", "Lorentzian"]
    for name in broadening.__all__:
        assert getattr(broadening, name) is getattr(numeric, name)


def test_broadening_works_without_scipy(monkeypatch, Assert):
    # the core installation has no scipy, and broadening shares its module with the
    # interpolation routines that do need it; touching those must not be required
    missing = import_.optional("_this_package_is_not_installed_")
    monkeypatch.setattr(numeric, "interpolate", missing)
    monkeypatch.setattr(numeric, "optimize", missing)
    mesh = np.linspace(-4, 4, 801)
    spectrum = broadening.broaden(
        mesh, [-1.0, 1.0], shape=broadening.Gaussian(fwhm=0.5)
    )
    Assert.allclose(np.trapezoid(spectrum, mesh), 2.0)
