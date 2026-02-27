# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import numpy as np

from py4vasp import _demo, raw


def dielectric_tensor(method, with_ion):
    shape = (3, _demo.AXES, _demo.AXES)
    data = np.arange(np.prod(shape)).reshape(shape)
    ion = raw.VaspData(data[1] if with_ion else None)
    independent_particle = raw.VaspData(
        data[2] if method in ("dft", "dft_slab", "rpa") else None
    )
    cell = None if method != "dft_slab" else _demo.cell.Graphite()
    return raw.DielectricTensor(
        electron=raw.VaspData(data[0]),
        ion=ion,
        independent_particle=independent_particle,
        method=method.encode(),
        cell=cell,
    )
