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


def Fe3O4():
    lattice_vectors = [
        [5.1427, 0.0, 0.0],
        [0.0, 3.0588, 0.0],
        [-1.3633791448, 0.0, 5.0446102592],
    ]
    scaling = np.linspace(0.98, 1.01, _demo.NUMBER_STEPS)
    lattice_vectors = np.multiply.outer(scaling, lattice_vectors)
    return raw.Cell(lattice_vectors, scale=raw.VaspData(None))


def Ba2PbO4():
    lattice_vectors = [
        [4.34, 0.0, 0.0],
        [0.0, 4.34, 0.0],
        [-2.17, -2.17, 6.682],
    ]
    return raw.Cell(lattice_vectors=np.array(lattice_vectors), scale=1.0)
