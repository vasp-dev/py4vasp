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


def slice_(mode):
    nkpx, nkpy, nkpz = (4, 3, 1)
    x = np.linspace(0, 1, nkpx, endpoint=False)
    y = np.linspace(0, 1, nkpy, endpoint=False)
    z = np.linspace(0, 1, nkpz, endpoint=False) + 1 / 8
    coordinates = np.array(list(itertools.product(x, y, z)))
    number_kpoints = len(coordinates) if mode[0] in ["e", b"e"[0]] else 0
    number_kpx = nkpx if mode[0] in ["e", b"e"[0]] else 0
    number_kpy = nkpy if mode[0] in ["e", b"e"[0]] else 0
    number_kpz = nkpz if mode[0] in ["e", b"e"[0]] else 0
    kpoints = raw.Kpoint(
        mode=mode,
        number=number_kpoints,
        number_x=number_kpx,
        number_y=number_kpy,
        number_z=number_kpz,
        coordinates=coordinates,
        weights=np.arange(len(coordinates)),
        cell=_demo.cell.Ba2PbO4(),
    )
    return kpoints
