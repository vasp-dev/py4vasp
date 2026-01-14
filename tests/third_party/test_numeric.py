# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import numpy as np

from py4vasp._third_party.numeric import analytic_continuation


def test_analytic_continuation_for_lorentzian(Assert):
    z_in = 1j * np.array([0.1, 1.0, 10.0])
    f_in = lorentzian(z_in)
    z_out = np.linspace(0.0, 2.5, 6)
    f_out = analytic_continuation(z_in, f_in, z_out)
    f_expected = lorentzian(z_out)
    Assert.allclose(f_out, f_expected)


def lorentzian(z):
    z0 = 1.0
    gamma = 0.5
    return 1 / (z - z0 + 1j * gamma)
