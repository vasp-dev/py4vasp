# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import numpy as np

from py4vasp import _demo, raw


def single_band():
    kpoints = _demo.kpoint.grid("explicit", "no_labels")
    eigenvalues = np.array([np.linspace([0], [1], len(kpoints.coordinates))])
    return raw.Dispersion(kpoints, eigenvalues)


def multiple_bands():
    kpoints = _demo.kpoint.grid("explicit", "no_labels")
    shape = (_demo.NONPOLARIZED, len(kpoints.coordinates), _demo.NUMBER_BANDS)
    eigenvalues = np.arange(np.prod(shape)).reshape(shape)
    return raw.Dispersion(kpoints, eigenvalues)


def line_mode(labels):
    kpoints = _demo.kpoint.line_mode("line", labels)
    shape = (_demo.NONPOLARIZED, len(kpoints.coordinates), _demo.NUMBER_BANDS)
    eigenvalues = np.arange(np.prod(shape)).reshape(shape)
    return raw.Dispersion(kpoints, eigenvalues)


def spin_polarized_bands():
    kpoints = _demo.kpoint.grid("explicit", "no_labels")
    kpoints.cell = _demo.cell.Fe3O4()
    shape = (_demo.COLLINEAR, len(kpoints.coordinates), _demo.NUMBER_BANDS)
    eigenvalues = np.arange(np.prod(shape)).reshape(shape)
    return raw.Dispersion(kpoints, eigenvalues)


def noncollinear_bands():
    kpoints = _demo.kpoint.line_mode("explicit", "no_labels")
    kpoints.cell = _demo.cell.Ba2PbO4()
    shape = (_demo.NONCOLLINEAR, len(kpoints.coordinates), _demo.NUMBER_BANDS)
    return raw.Dispersion(kpoints, eigenvalues=_demo.wrap_random_data(shape))


def spin_texture():
    kpoints = _demo.kpoint.slice_("explicit")
    kpoints.cell = _demo.cell.Ba2PbO4()
    shape = (_demo.NONCOLLINEAR, len(kpoints.coordinates), _demo.NUMBER_BANDS)
    eigenvalues = np.arange(np.prod(shape)).reshape(shape)
    return raw.Dispersion(kpoints, eigenvalues)
