# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import numpy as np
from scipy.interpolate import AAA
from scipy.optimize import curve_fit


def analytic_continuation(z_in, f_in, z_out):
    shape = f_in.shape
    data_sets = f_in.reshape((-1, shape[-1]))
    f_out = np.array(
        [_analytic_continuation_single(z_in, data_set, z_out) for data_set in data_sets]
    )
    return f_out.reshape(shape[:-1] + (len(z_out),))


def _analytic_continuation_single(z_in, f_in, z_out):
    aaa = AAA(z_in, f_in)
    return aaa(z_out)


def interpolate_with_function(function, x_in, y_in, x_out):
    shape = y_in.shape
    data_sets = y_in.reshape((-1, shape[-1]))
    y_out = np.array(
        [
            _interpolate_with_function_single(function, x_in, data_set, x_out)
            for data_set in data_sets
        ]
    )
    return y_out.reshape(shape[:-1] + (len(x_out),))


def _interpolate_with_function_single(function, x_in, y_in, x_out):
    parameters, _ = curve_fit(function, x_in, y_in)
    return function(x_out, *parameters)
