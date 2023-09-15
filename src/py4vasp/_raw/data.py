# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
from __future__ import annotations

import dataclasses

from py4vasp._raw.data_wrapper import VaspData


def NONE():
    return dataclasses.field(default_factory=lambda: VaspData(None))


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
    """A band structure generated by VASP.

    Contains the eigenvalues at specifics **k** points in the Brillouin zone. Carefully
    consider before using the absolute eigenvalues for analysis. Typically only relative
    eigenvalues e.g. with respect to the Fermi energy are meaningful. Includes
    projections of the bands on orbitals and atoms when available."""

    dispersion: Dispersion
    "The **k** points and eigenvalues of the dispersion."
    fermi_energy: float
    "Fermi energy obtained by VASP."
    occupations: VaspData
    "The occupations of the different bands."
    projectors: Projector
    "Projector information (element, angular momentum, spin)."
    projections: VaspData = NONE()
    "If present, orbital projections of the bands."


@dataclasses.dataclass
class Bandgap:
    """The bandgap of the system.

    Contains the band extrema defining the fundamental and optical band gap and the
    k-point coordinates where the band gaps are for all steps of a relaxation/MD
    simulation."""

    labels: VaspData
    "These labels identify which band data is stored."
    values: VaspData
    "The data contained according to the labels."


@dataclasses.dataclass
class BornEffectiveCharge:
    """The Born effective charges resulting form a linear response calculation.

    The Born effective charges describe how the polarization of a system changes with
    the position of the ions. Equivalently, an electric field induces forces on the
    ions. In general, the Born effective charges are matrices, i.e., the polarization
    and the displacement of the ion are not necessarily parallel."""

    structure: Structure
    "Structural information about the system to identify specific atoms."
    charge_tensors: VaspData
    "The data of the Born effective charges."


@dataclasses.dataclass
class Cell:
    """Unit cell of the crystal or simulation cell for molecules.

    In MD simulations or relaxations, VASP exports the unit cell for every step,
    because it may change depending on the ISIF setting."""

    lattice_vectors: VaspData
    "Lattice vectors defining the unit cell."
    scale: float = 1.0
    "Global scaling factor applied to all lattice vectors."


@dataclasses.dataclass
class CONTCAR:
    """The data corresponding to the CONTCAR file.

    The CONTCAR file contains structural information (lattice, positions, topology),
    relaxation constraints, and data relevant for continuation calculations.
    """

    structure: Structure
    "The structure of the system at the end of the calculation."
    system: str
    "A comment line describing the system."
    selective_dynamics: VaspData = NONE()
    "Specifies in which directions the atoms may move."
    lattice_velocities: VaspData = NONE()
    "The current velocities of the lattice vectors."
    ion_velocities: VaspData = NONE()
    "The current velocities of the ions."
    _predictor_corrector: VaspData = NONE()
    "Internal algorithmic data relevant for restarting calculations."


@dataclasses.dataclass
class Density:
    "The electronic charge and magnetization density on the Fourier grid."
    structure: Structure
    "The atomic structure to represent the densities."
    charge: VaspData
    "The data of electronic charge and magnetization density."


@dataclasses.dataclass
class DielectricFunction:
    """The full frequency-dependent dielectric function.

    The total dielectric function is the sum of the ionic and electronic part. It
    provides insight into optical properties such as the reflectivity and absorption.
    Note that the dielectric function is a 3x3 matrix for every frequency. There are
    many different levels of theory with which you can evaluate the dielectric function
    in VASP. For the electronic dielectric function a current-current response may be
    provided as alternative."""

    energies: VaspData
    "The energies at which the dielectric function is evaluated."
    dielectric_function: VaspData
    "The values of the dielectric function (frequency-dependent 3x3 tensor)."
    current_current: VaspData = NONE()
    "Dielectric function obtained using the current-current response."


@dataclasses.dataclass
class DielectricTensor:
    """The dielectric tensor resulting from ionic and electronic contributions.

    The dielectric tensor should match with the zero frequency limit of the dielectric
    function. The electronic contribution is also provided in the independent particle
    approximation."""

    electron: VaspData
    "The electronic contribution to the dielectric tensor."
    ion: VaspData
    "The ionic contribution to the dielectric tensor."
    independent_particle: VaspData
    "The dielectric tensor in the independent particle approximation."
    method: str
    "The method used to generate the dielectric tensor."


@dataclasses.dataclass
class Dispersion:
    """A general class for dispersions (electron and phonon)."""

    kpoints: Kpoint
    "**k** points at which the bands are calculated."
    eigenvalues: VaspData
    "Calculated eigenvalues at the **k** points."


@dataclasses.dataclass
class Dos:
    """Contains the density of states (DOS) including its projections where available.

    Contains the energy mesh and the values of the DOS at the mesh points. When LORBIT
    is set in the INCAR file, VASP projects the DOS onto atoms and orbitals. Typically,
    the absolute value of the energy mesh is not important and shifting the energies to
    a reference e.g. the Fermi energy is desired."""

    energies: VaspData
    "Energy E at which the Dos is evaluated."
    dos: VaspData
    "Dos at the energies D(E)."
    fermi_energy: float
    "Fermi energy obtained by VASP."
    projectors: Projector
    "Projector information (element, angular momentum, spin)."
    projections: VaspData = NONE()
    "If present, orbital projections of the Dos."


@dataclasses.dataclass
class ElasticModulus:
    """The elastic modulus calculated in a linear response calculation.

    The elastic modulus is the second derivative of the total energy with respect to
    a strain in the system. When straining the system, one can enforce or relax the
    positions of the ions. Correspondingly, VASP evaluates the clamped-ion and
    relaxed-ion elastic modulus."""

    clamped_ion: VaspData
    "Elastic modulus when the ions are clamped into their positions."
    relaxed_ion: VaspData
    "Elastic modulus when the position of the ions is relaxed."


@dataclasses.dataclass
class Energy:
    """Various energies during ionic relaxation or MD simulation.

    At every step during a simulation, VASP stores the energy values. For MD simulations,
    VASP includes temperature information from the thermostat."""

    labels: VaspData
    "Label identifying which energy is contained."
    values: VaspData
    "Energy specified by labels for all iteration steps."


@dataclasses.dataclass
class Fatband:
    """Contains the BSE data required to produce a fatband plot."""

    dispersion: Dispersion
    "The **k** points and the eigenvalues of the band structure."
    fermi_energy: float
    "The Fermi energy of the system."
    bse_index: VaspData
    "The connection between spin, band and **k**-point indices to an index of the optical transitions."
    fatbands: VaspData
    "Component of the eigenvector, norm can be used for plotting fatbands."
    first_valence_band: int
    "Index of the first valence band."
    first_conduction_band: int
    "Index of the first conduction band."


@dataclasses.dataclass
class Force:
    """The forces acting on the atoms at all steps of a MD simulation or relaxation."""

    structure: Structure
    "Structural information about the system to inform about the forces."
    forces: VaspData
    "The values of the forces at the atoms."


@dataclasses.dataclass
class ForceConstant:
    """The force constants of the material.

    The force constant describes the second derivative of the total energy with respect
    to the displacement of ions. It is an important quantity for the phonon spectrum."""

    structure: Structure
    "Structural information about the system to inform about the atoms the force constants relate to."
    force_constants: VaspData
    "The values of the force constants."


@dataclasses.dataclass
class InternalStrain:
    """The internal strain calculated in a linear response calculation.

    The internal strain describes which forces act on the ions when the crystal is
    subject to a strain. Equivalently, this determines the stress on the crystal
    induced from ionic displacements."""

    structure: Structure
    "Structural information about the system to inform about the unit cell."
    internal_strain: VaspData
    "The  data of the internal strain."


@dataclasses.dataclass
class Kpoint:
    """A **k**-point mesh in the Brillouin zone.

    Describes how VASP generated the **k** points and contains their coordinates.
    Labels may be defined in the KPOINTS file and are then available for band structure
    plots. For integrations over the Brillouin zone, every **k** point exhibits an
    integration weight. Use the unit cell information to transform to Cartesian
    coordinates if desired."""

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
    labels: VaspData = NONE()
    "High symmetry label for specific **k** points used in band structures."
    label_indices: VaspData = NONE()
    "Indices of the labeled **k** points in the generation list."


@dataclasses.dataclass
class Magnetism:
    """The local charges and magnetic moments on the ions.

    The projection on orbitals and atoms (LORBIT) distributes all bands over all ions
    according to the overlap with local projectors. This information reveals which
    ions and orbitals contribute to the magnetism in the system. VASP stores the
    projection for every step of a simulation so that one can monitor changes along
    a relaxation or MD run."""

    structure: Structure
    "Structural information about the system."
    spin_moments: VaspData
    "Contains the charge and magnetic moments atom and orbital resolved."
    orbital_moments: VaspData = NONE()
    "Contains the orbital magnetization for all atoms"


@dataclasses.dataclass
class PairCorrelation:
    """The pair-correlation function calculated during a MD simulation.

    The pair-correlation function describes how other ions are distributed around a
    given ion. VASP evaluates the total pair-function and the element-resolved ones."""

    distances: VaspData
    "The distances at which the pair-correlation function is evaluated."
    function: VaspData
    "The total and the element-resolved pair-correlation functions."
    labels: VaspData
    "Describes which indices correspond to which element pair."


@dataclasses.dataclass
class PhononBand:
    """The band structure of the phonons.

    Contains the eigenvalues and eigenvectors at specifics **q** points in the Brillouin
    zone. Includes the topology to map atoms onto specific modes."""

    dispersion: Dispersion
    "The **q** points and the eigenvalues."
    topology: Topology
    "The atom types in the crystal."
    eigenvectors: VaspData
    "The eigenvectors of the phonon modes."


@dataclasses.dataclass
class PhononDos:
    """Contains the phonon density of states (DOS) including its projections where available.

    Contains the energy mesh and the values of the DOS at the mesh points. The DOS can be
    projected onto specific ions or ion types."""

    energies: VaspData
    "Energy E at which the Dos is evaluated."
    dos: VaspData
    "Dos at the energies D(E)."
    projections: VaspData
    "Projection of the DOS onto contribution of specific atoms."
    topology: Topology
    "The atom types in the crystal."


@dataclasses.dataclass
class PiezoelectricTensor:
    """The piezoelectric tensor calculated in a linear response calculation.

    The piezoelectric tensor describes how an electric field induces a stress on the
    crystal. Equivalently, straining the system can induce a polarization. VASP splits
    the piezoelectric tensor into an electronic and an ionic contribution."""

    electron: VaspData
    "The electronic contribution to the piezoelectric tensor"
    ion: VaspData
    "The ionic contribution to the piezoelectric tensor"


@dataclasses.dataclass
class Polarization:
    """The electronic and ionic dipole moments.

    The polarization of a system results from the charge and the position of the ions.
    It is also the derivative of the total energy with respect to the electric field.
    VASP reports the dipole moments for ions and electrons, separately. For periodic
    systems polarizations are not well defined. Make sure to evaluate changes of the
    polarization relative to a centrosymmetric reference structure."""

    electron: VaspData
    "The electronic dipole moment resulting from the charge."
    ion: VaspData
    "The ionic dipole moment resulting from the position of the atoms."


@dataclasses.dataclass
class Projector:
    """Projectors used for atom and orbital projections.

    Set LORBIT to project quantities such as the DOS or the band structure onto atoms
    and orbitals. This class reports the atoms and orbitals included in the projection.
    """

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
    """Structural information of the system.

    Reports what ions are in the system and the positions of all ions as well as the
    unit cell for all steps in a relaxation in a MD run."""

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
    "Contains the type of ions in the system and how many of each type exist."
    number_ion_types: VaspData
    "Amount of ions of a particular type."
    ion_types: VaspData
    "Element of a particular type."


@dataclasses.dataclass
class Velocity:
    "Contains the ion velocities along the trajectory."
    structure: Structure
    "Structural information to relate the velocities to."
    velocities: VaspData
    "Observed ion velocities."
