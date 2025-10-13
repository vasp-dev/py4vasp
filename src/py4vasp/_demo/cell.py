# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import numpy as np

from py4vasp import _demo, raw


def Sr2TiO4():
    scale = raw.VaspData(6.9229)
    lattice_vectors = [
        [1.0, 0.0, 0.0],
        [0.678112209738693, 0.734958387251008, 0.0],
        [-0.839055341042049, -0.367478859090843, 0.401180037874301],
    ]
    return raw.Cell(
        lattice_vectors=np.array(_demo.NUMBER_STEPS * [lattice_vectors]), scale=scale
    )
