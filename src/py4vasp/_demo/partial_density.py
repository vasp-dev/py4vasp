# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import random

import numpy as np

from py4vasp import _demo, raw
from py4vasp._util import import_

stats = import_.optional("scipy.stats")


def partial_density(selection):
    grid_dim = _demo.GRID_DIMENSIONS
    if "CaAs3_110" in selection:
        structure = _demo.structure.CaAs3_110()
        grid_dim = (240, 40, 32)
    elif "Sr2TiO4" in selection:
        structure = _demo.structure.Sr2TiO4()
    elif "Ca3AsBr3" in selection:
        structure = _demo.structure.Ca3AsBr3()
    elif "Ni100" in selection:
        structure = _demo.structure.Ni100()
    else:
        structure = _demo.structure.Graphite()
        grid_dim = (216, 24, 24)
    if "split_bands" in selection:
        bands = raw.VaspData(random.sample(range(1, 51), 3))
    else:
        bands = raw.VaspData(np.asarray([0]))
    if "split_kpoints" in selection:
        kpoints = raw.VaspData((random.sample(range(1, 26), 5)))
    else:
        kpoints = raw.VaspData(np.asarray([0]))
    if "spin_polarized" in selection:
        spin_dimension = 2
    else:
        spin_dimension = 1
    grid = raw.VaspData(tuple(reversed(grid_dim)))
    charge_shape = (len(kpoints), len(bands), spin_dimension, *grid_dim)
    gaussian_charge = np.zeros(charge_shape)
    cov = grid_dim[0] / 10  # standard deviation
    z = np.arange(grid_dim[0])  # z range
    for gy in range(grid_dim[1]):
        for gx in range(grid_dim[2]):
            m = int(grid_dim[0] / 2) + gy / 10 + gx / 10
            val = stats.multivariate_normal(mean=m, cov=cov).pdf(z)
            # Fill the gaussian_charge array
            gaussian_charge[:, :, :, :, gy, gx] = val
    gaussian_charge = raw.VaspData(gaussian_charge)
    return raw.PartialDensity(
        structure=structure,
        bands=bands,
        kpoints=kpoints,
        partial_charge=gaussian_charge,
        grid=grid,
    )
