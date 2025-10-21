# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import numpy as np

from py4vasp import _demo, raw


def Sr2TiO4(projectors):
    energies = np.linspace(-1, 3, _demo.NUMBER_POINTS)
    use_orbitals = projectors == "with_projectors"
    raw_dos = raw.Dos(
        fermi_energy=1.372,
        energies=energies,
        dos=np.array([energies**2]),
        projectors=_demo.projector.Sr2TiO4(use_orbitals),
    )
    if use_orbitals:
        number_orbitals = len(raw_dos.projectors.orbital_types)
        shape = _shape_projections(_demo.NONPOLARIZED, number_orbitals)
        raw_dos.projections = _demo.wrap_random_data(shape)
    return raw_dos


def Fe3O4(projectors):
    energies = np.linspace(-2, 2, _demo.NUMBER_POINTS)
    use_orbitals = projectors in ["with_projectors", "excess_orbitals"]
    raw_dos = raw.Dos(
        fermi_energy=-0.137,
        energies=energies,
        dos=np.array(((energies + 0.5) ** 2, (energies - 0.5) ** 2)),
        projectors=_demo.projector.Fe3O4(use_orbitals),
    )
    if use_orbitals:
        number_orbitals = len(raw_dos.projectors.orbital_types)
        shape = _shape_projections(_demo.COLLINEAR, number_orbitals)
        raw_dos.projections = _demo.wrap_random_data(shape)
    if projectors == "excess_orbitals":
        orbital_types = _demo.wrap_orbital_types(use_orbitals, "s p d f g h i")
        raw_dos.projectors.orbital_types = orbital_types
    return raw_dos


def Ba2PbO4(projectors):
    assert projectors == "noncollinear"
    energies = np.linspace(-4, 1, _demo.NUMBER_POINTS)
    raw_dos = raw.Dos(
        fermi_energy=-1.3,
        energies=energies,
        dos=_demo.wrap_random_data((_demo.NONCOLLINEAR, _demo.NUMBER_POINTS)),
        projectors=_demo.projector.Ba2PbO4(use_orbitals=True),
    )
    number_orbitals = len(raw_dos.projectors.orbital_types)
    shape = _shape_projections(_demo.NONCOLLINEAR, number_orbitals)
    raw_dos.projections = _demo.wrap_random_data(shape)
    return raw_dos


def _shape_projections(components, num_orbitals):
    return (components, _demo.NUMBER_ATOMS, num_orbitals, _demo.NUMBER_POINTS)
