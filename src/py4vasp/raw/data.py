# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
from __future__ import annotations
import dataclasses
from py4vasp.raw.data_wrapper import VaspData


@dataclasses.dataclass(order=True, frozen=True)
class Version:
    "The version number of VASP."
    major: int
    "The major version number."
    minor: int = 0
    "The minor version number."
    patch: int = 0
    "Indicates number of bugfixes since last minor release."
    __str__ = lambda self: f"version {self.major}.{self.minor}.{self.patch}"


@dataclasses.dataclass
class Band:
    "Electronic band structure"
    fermi_energy: float
    "Fermi energy obtained by VASP."
    kpoints: Kpoint
    "**k** points at which the bands are calculated."
    eigenvalues: VaspData
    "Calculated eigenvalues at the **k** points."
    occupations: VaspData
    "The occupations of the different bands."
    projections: VaspData = None
    "If present, orbital projections of the bands."
    projectors: Projector = None
    "If present, projector information (element, angular momentum, spin)."


@dataclasses.dataclass
class BornEffectiveCharge:
    "The Born effective charges resulting form a linear response calculation."
    structure: Structure
    "Structural information about the system to identify specific atoms."
    charge_tensors: VaspData
    "The  data of the Born effective charges."


@dataclasses.dataclass
class Cell:
    "Unit cell of the crystal or simulation cell for molecules."
    lattice_vectors: VaspData
    "Lattice vectors defining the unit cell."
    scale: float
    "Global scaling factor applied to all lattice vectors."


@dataclasses.dataclass
class Density:
    "The electronic charge and magnetization density."
    structure: Structure
    "The atomic structure to represent the densities."
    charge: VaspData
    "The  data of electronic charge and magnetization density."


@dataclasses.dataclass
class DielectricFunction:
    "The electronic or ionic dielectric function."
    energies: VaspData
    "The energies at which the dielectric function is evaluated."
    density_density: VaspData
    "The values of the electronic dielectric function using the density-density response."
    current_current: VaspData
    "The values of the electronic dielectric function using the current-current response."
    ion: VaspData
    "The values of the ionic dielectrion function."


@dataclasses.dataclass
class DielectricTensor:
    "The dielectric tensor resulting from ionic and electronic contributions."
    electron: VaspData
    "The electronic contribution to the dielectric tensor."
    ion: VaspData
    "The ionic contribution to the dielectric tensor."
    independent_particle: VaspData
    "The dielectric tensor in the independent particle approximation."
    method: str
    "The method used to generate the dielectric tensor."


@dataclasses.dataclass
class Dos:
    "Electronic density of states."
    fermi_energy: float
    "Fermi energy obtained by VASP."
    energies: VaspData
    "Energy E at which the Dos is evaluated."
    dos: VaspData
    "Dos at the energies D(E)."
    projections: VaspData = None
    "If present, orbital projections of the Dos."
    projectors: Projector = None
    "If present, projector information (element, angular momentum, spin)."


@dataclasses.dataclass
class ElasticModulus:
    "The elastic module calculated in a linear response calculation."
    clamped_ion: VaspData
    "Elastic modulus when the ions are clamped into their positions."
    relaxed_ion: VaspData
    "Elastic modulus when the position of the ions is relaxed."


@dataclasses.dataclass
class Energy:
    "Various energies during ionic relaxation or MD simulation."
    labels: VaspData
    "Label identifying which energy is contained."
    values: VaspData
    "Energy specified by labels for all iteration steps."


@dataclasses.dataclass
class Force:
    "The forces acting on the atoms at all steps."
    structure: Structure
    "Structural information about the system to inform about the forces."
    forces: VaspData
    "The values of the forces at the atoms."


@dataclasses.dataclass
class ForceConstant:
    "The force constants of the material."
    structure: Structure
    "Structural information about the system to inform about the atoms the force constants relate to."
    force_constants: VaspData
    "The values of the force constants."


@dataclasses.dataclass
class InternalStrain:
    "The internal strain calculated in a linear response calculation."
    structure: Structure
    "Structural information about the system to inform about the unit cell."
    internal_strain: VaspData
    "The  data of the internal strain."


@dataclasses.dataclass
class Kpoint:
    "**k** points at which wave functions are calculated."
    mode: str
    "Mode used to generate the **k**-point list."
    number: int
    "Number of **k** points specified in the generation."
    coordinates: VaspData
    "Coordinates of the **k** points as fraction of the reciprocal lattice vectors."
    weights: VaspData
    "Weight of the **k** points used for integration."
    cell: Cell
    "Unit cell of the crystal."
    labels: VaspData = None
    "High symmetry label for specific **k** points used in band structures."
    label_indices: VaspData = None
    "Indices of the labeled **k** points in the generation list."


@dataclasses.dataclass
class Magnetism:
    "Data about the magnetism in the system."
    structure: Structure
    "Structural information about the system."
    moments: VaspData
    "Contains the charge and magnetic moments atom and orbital resolved."


@dataclasses.dataclass
class PiezoelectricTensor:
    "The piezoelectric tensor calculated in a linear response calculation."
    electron: VaspData
    "The electronic contribution to the piezoelectric tensor"
    ion: VaspData
    "The ionic contribution to the piezoelectric tensor"


@dataclasses.dataclass
class Polarization:
    "The electronic and ionic dipole moments."
    electron: VaspData
    "The electronic dipole moment resulting from the charge."
    ion: VaspData
    "The ionic dipole moment resulting from the position of the atoms."


@dataclasses.dataclass
class Projector:
    "Projectors used for orbital projections."
    topology: Topology
    "The topology of the system used, i.e., which elements are contained."
    orbital_types: VaspData
    "Character indicating the orbital angular momentum."
    number_spins: int
    "Indicates whether the calculation is spin polarized or not."


@dataclasses.dataclass
class Stress:
    "The stress acting on the unit cell at all steps."
    structure: Structure
    "Structural information about the system to inform about the unit cell."
    stress: VaspData
    "The values of the stress on the cell."


@dataclasses.dataclass
class Structure:
    "Structural information of the system."
    topology: Topology
    "The topology of the system used, i.e., which elements are contained."
    cell: Cell
    "Unit cell of the crystal or simulation cell for molecules."
    positions: VaspData
    "Position of all atoms in the unit cell in units of the lattice vectors."


@dataclasses.dataclass
class System:
    "The name of the system set in the input."
    system: str


@dataclasses.dataclass
class Topology:
    "The topology of the system used, i.e., which elements are contained."
    number_ion_types: VaspData
    "Amount of ions of a particular type."
    ion_types: VaspData
    "Element of a particular type."
