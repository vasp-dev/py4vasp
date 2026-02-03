# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
from dataclasses import dataclass
from typing import Optional

import numpy as np

from py4vasp._util import import_

interpolate = import_.optional("scipy.interpolate")
optimize = import_.optional("scipy.optimize")


@dataclass(kw_only=True)
class AAAConfig:
    rtol: Optional[float] = None
    max_terms: int = 100
    clean_up: bool = True
    clean_up_tol: float = 1e-13


def analytic_continuation(z_in, f_in, z_out, *, config: AAAConfig = AAAConfig()):
    shape = f_in.shape
    data_sets = f_in.reshape((-1, shape[-1]))
    f_out = [
        _analytic_continuation_single(z_in, data_set, z_out, config)
        for data_set in data_sets
    ]
    return np.reshape(f_out, shape[:-1] + (len(z_out),))


def _analytic_continuation_single(z_in, f_in, z_out, config):
    aaa = interpolate.AAA(
        z_in,
        f_in,
        rtol=config.rtol,
        max_terms=config.max_terms,
        clean_up=config.clean_up,
        clean_up_tol=config.clean_up_tol,
    )
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
    parameters, _ = optimize.curve_fit(function, x_in, y_in)
    return function(x_out, *parameters)
