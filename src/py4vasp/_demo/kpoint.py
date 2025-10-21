# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import itertools

import numpy as np

from py4vasp import _demo, raw


def grid(mode, labels):
    nkpx, nkpy, nkpz = (4, 3, 4)
    x = np.linspace(0, 1, nkpx, endpoint=False)
    y = np.linspace(0, 1, nkpy, endpoint=False)
    z = np.linspace(0, 1, nkpz, endpoint=False) + 1 / 8
    coordinates = np.array(list(itertools.product(x, y, z)))
    number_kpoints = len(coordinates) if mode[0] in ["e", b"e"[0]] else 0
    kpoints = raw.Kpoint(
        mode=mode,
        number=number_kpoints,
        number_x=nkpx,
        number_y=nkpy,
        number_z=nkpz,
        coordinates=coordinates,
        weights=np.arange(len(coordinates)),
        cell=_demo.cell.Sr2TiO4(),
    )
    if labels == "with_labels":
        kpoints.labels = _demo.wrap_data(["foo", b"bar", "baz"])
        kpoints.label_indices = _demo.wrap_data([9, 25, 40])
    return kpoints


def line_mode(mode, labels):
    line_length = 5
    GM = [0, 0, 0]
    Y = [0.5, 0.5, 0.0]
    A = [0, 0, 0.5]
    M = [0.5, 0.5, 0.5]
    coordinates = (
        np.linspace(GM, A, line_length),
        np.linspace(A, M, line_length),
        np.linspace(GM, Y, line_length),
        np.linspace(Y, M, line_length),
    )
    kpoints = raw.Kpoint(
        mode=mode,
        number=line_length,
        coordinates=np.concatenate(coordinates),
        weights=np.ones(len(coordinates)),
        cell=_demo.cell.Sr2TiO4(),
    )
    if labels == "with_labels":
        kpoints.labels = _demo.wrap_data([r"$\Gamma$", " M ", r"$\Gamma$", "Y", "M"])
        kpoints.label_indices = _demo.wrap_data([1, 4, 5, 7, 8])
    return kpoints


def slice_(selection):
    if selection == "x~y":
        number_x, number_y, number_z = 4, 3, 1
    elif selection == "x~z":
        number_x, number_y, number_z = 4, 1, 3
    else:  # selection == "y~z"
        number_x, number_y, number_z = 1, 4, 3
    x = np.linspace(0, 1, number_x, endpoint=False)
    y = np.linspace(0, 1, number_y, endpoint=False)
    z = np.linspace(0, 1, number_z, endpoint=False)
    coordinates = np.array(list(itertools.product(x, y, z)))
    kpoints = raw.Kpoint(
        mode="explicit",
        number=len(coordinates),
        number_x=number_x,
        number_y=number_y,
        number_z=number_z,
        coordinates=coordinates,
        weights=np.arange(len(coordinates)),
        cell=_demo.cell.Ba2PbO4(),
    )
    return kpoints


def qpoints():
    qpoints = line_mode("line", "with_labels")
    qpoints.cell.lattice_vectors = qpoints.cell.lattice_vectors[-1]
    return qpoints
