# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import numpy as np
from scipy.interpolate import AAA


def analytic_continuation(z_in, f_in, z_out):
    shape = f_in.shape
    data_sets = f_in.reshape((shape[0], -1)).T
    f_out = np.array(
        [_analytic_continuation_single(z_in, data_set, z_out) for data_set in data_sets]
    )
    return f_out.T.reshape((len(z_out),) + shape[1:])


def _analytic_continuation_single(z_in, f_in, z_out):
    aaa = AAA(z_in, f_in)
    return aaa(z_out)
