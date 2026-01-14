# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
from scipy.interpolate import AAA


def analytic_continuation(z_in, f_in, z_out):
    aaa = AAA(z_in, f_in)
    return aaa(z_out)
