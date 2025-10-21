# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import numpy as np

from py4vasp import _demo, raw


def Ba2PbO4():
    lattice_vectors = [
        [4.34, 0.0, 0.0],
        [0.0, 4.34, 0.0],
        [-2.17, -2.17, 6.682],
    ]
    return raw.Cell(lattice_vectors=np.array(lattice_vectors), scale=1.0)


def Ca3AsBr3():
    return raw.Cell(
        scale=raw.VaspData(5.93),
        lattice_vectors=_demo.wrap_data(np.eye(3)),
    )


def CaAs3_110():
    lattice_vectors = [
        [5.65019183, 0.00000000, 1.90320681],
        [0.85575829, 7.16802977, 0.65250675],
        [0.00000000, 0.00000000, 44.41010402],
    ]
    return raw.Cell(np.asarray(lattice_vectors), scale=raw.VaspData(1.0))


def Fe3O4():
    lattice_vectors = [
        [5.1427, 0.0, 0.0],
        [0.0, 3.0588, 0.0],
        [-1.3633791448, 0.0, 5.0446102592],
    ]
    scaling = np.linspace(0.98, 1.01, _demo.NUMBER_STEPS)
    lattice_vectors = np.multiply.outer(scaling, lattice_vectors)
    return raw.Cell(lattice_vectors, scale=raw.VaspData(None))


def Graphite():
    lattice_vectors = [
        [2.44104624, 0.00000000, 0.00000000],
        [-1.22052312, 2.11400806, 0.00000000],
        [0.00000000, 0.00000000, 22.0000000],
    ]
    return raw.Cell(np.asarray(lattice_vectors), scale=raw.VaspData(1.0))


def Ni100():
    lattice_vectors = [
        [2.496086836, 0.00000000, 0.00000000],
        [-1.22052312, 35.2999992371, 0.00000000],
        [0.00000000, 0.00000000, 2.4960868359],
    ]
    return raw.Cell(np.asarray(lattice_vectors), scale=raw.VaspData(1.0))


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
