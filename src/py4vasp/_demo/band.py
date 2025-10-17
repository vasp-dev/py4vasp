# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import numpy as np

from py4vasp import _demo, raw


def single_band():
    dispersion = _demo.dispersion.single_band()
    return raw.Band(
        dispersion=dispersion,
        fermi_energy=0.0,
        occupations=np.array([np.linspace([1], [0], dispersion.eigenvalues.size)]),
        projectors=_demo.projector.Sr2TiO4(use_orbitals=False),
    )


def multiple_bands(projectors):
    dispersion = _demo.dispersion.multiple_bands()
    shape = dispersion.eigenvalues.shape
    use_orbitals = projectors == "with_projectors"
    raw_band = raw.Band(
        dispersion=dispersion,
        fermi_energy=0.5,
        occupations=np.arange(np.prod(shape)).reshape(shape),
        projectors=_demo.projector.Sr2TiO4(use_orbitals),
    )
    if use_orbitals:
        number_orbitals = len(raw_band.projectors.orbital_types)
        shape = (_demo.NONPOLARIZED, _demo.NUMBER_ATOMS, number_orbitals, *shape[1:])
        raw_band.projections = np.random.random(shape)
    return raw_band


def line_mode(labels):
    dispersion = _demo.dispersion.line_mode(labels)
    shape = dispersion.eigenvalues.shape
    return raw.Band(
        dispersion=dispersion,
        fermi_energy=0.5,
        occupations=np.arange(np.prod(shape)).reshape(shape),
        projectors=_demo.projector.Sr2TiO4(use_orbitals=False),
    )


def spin_polarized_bands(projectors):
    dispersion = _demo.dispersion.spin_polarized_bands()
    shape = dispersion.eigenvalues.shape
    use_orbitals = projectors in ["with_projectors", "excess_orbitals"]
    raw_band = raw.Band(
        dispersion=dispersion,
        fermi_energy=0.0,
        occupations=np.arange(np.prod(shape)).reshape(shape),
        projectors=_demo.projector.Fe3O4(use_orbitals),
    )
    if use_orbitals:
        number_orbitals = len(raw_band.projectors.orbital_types)
        shape = (_demo.COLLINEAR, _demo.NUMBER_ATOMS, number_orbitals, *shape[1:])
        raw_band.projections = np.random.random(shape)
    if projectors == "excess_orbitals":
        orbital_types = _demo.wrap_orbital_types(use_orbitals, "s p d f g h i")
        raw_band.projectors.orbital_types = orbital_types
    return raw_band


def noncollinear_bands(projectors):
    dispersion = _demo.dispersion.noncollinear_bands()
    shape = dispersion.eigenvalues.shape
    use_orbitals = projectors == "with_projectors"
    raw_band = raw.Band(
        dispersion=dispersion,
        fermi_energy=0.0,
        occupations=_demo.wrap_random_data(shape),
        projectors=_demo.projector.Ba2PbO4(use_orbitals),
    )
    if use_orbitals:
        number_orbitals = len(raw_band.projectors.orbital_types)
        shape = (_demo.NONCOLLINEAR, _demo.NUMBER_ATOMS, number_orbitals, *shape[1:])
        raw_band.projections = _demo.wrap_random_data(shape)
    return raw_band


def spin_texture(selection):
    dispersion = _demo.dispersion.spin_texture()
    projectors = _demo.projector.Ba2PbO4(use_orbitals=True)
    number_orbitals = len(projectors.orbital_types)
    shape_occ = dispersion.eigenvalues.shape
    shape = (_demo.NONCOLLINEAR, _demo.NUMBER_ATOMS, number_orbitals, *shape_occ[1:])
    return raw.Band(
        dispersion=dispersion,
        fermi_energy=0.0,
        occupations=_demo.wrap_random_data(shape_occ),
        projectors=projectors,
        projections=_demo.wrap_random_data(shape),
    )
