from dataclasses import dataclass
import numpy as np


class DataDict(dict):
    """Provides an extension to a dictionary storing also the version of the data.

    Parameters
    ----------
    dict_: dict
        A dictionary containing raw data and descriptive keys.
    version: RawVersion
        The version of Vasp with which the data was generated.
    """

    def __init__(self, dict_, version):
        super().__init__(dict_)
        self.version = version


def _dataclass_equal(lhs, rhs):
    lhs, rhs = vars(lhs), vars(rhs)
    compare = (_element_equal(lhs[key], rhs[key]) for key in lhs)
    return all(compare)


def _element_equal(lhs, rhs):
    if _only_one_None(lhs, rhs):
        return False
    lhs, rhs = np.array(lhs), np.array(rhs)
    return lhs.shape == rhs.shape and np.all(lhs == rhs)


def _only_one_None(lhs, rhs):
    return (lhs is None) != (rhs is None)


@dataclass(order=True, frozen=True)
class RawVersion:
    "The version number of Vasp."
    major: int
    "The major version number."
    minor: int = 0
    "The minor version number."
    patch: int = 0
    "Indicates number of bugfixes since last minor release."


@dataclass
class RawTopology:
    "The topology of the system used, i.e., which elements are contained."
    number_ion_types: np.ndarray
    "Amount of ions of a particular type."
    ion_types: np.ndarray
    "Element of a particular type."
    __eq__ = _dataclass_equal


@dataclass
class RawTrajectory:
    "Describes the evolution of unit cell and atoms within over ionic steps."
    topology: RawTopology
    "The topology of the system used, i.e., which elements are contained."
    lattice_vectors: np.ndarray
    "Lattice vectors defining the unit cell for every time step."
    positions: np.ndarray
    """Position of all atoms in the unit cell in units of the lattice vectors
    for every timestep."""
    __eq__ = _dataclass_equal


@dataclass
class RawProjectors:
    "Projectors used for orbital projections."
    topology: RawTopology
    "The topology of the system used, i.e., which elements are contained."
    orbital_types: np.ndarray
    "Character indicating the orbital angular momentum."
    number_spins: int
    "Indicates whether the calculation is spin polarized or not."
    __eq__ = _dataclass_equal


@dataclass
class RawCell:
    "Unit cell of the crystal or simulation cell for molecules."
    lattice_vectors: np.ndarray
    "Lattice vectors defining the unit cell."
    scale: float = 1.0
    "Global scaling factor applied to all lattice vectors."
    __eq__ = _dataclass_equal


@dataclass
class RawMagnetism:
    "Data about the magnetism in the system."
    moments: np.ndarray
    "Contains the charge and magnetic moments atom and orbital resolved."
    __eq__ = _dataclass_equal


@dataclass
class RawStructure:
    "Structural information of the system."
    topology: RawTopology
    "The topology of the system used, i.e., which elements are contained."
    cell: RawCell
    "Unit cell of the crystal or simulation cell for molecules."
    positions: np.ndarray
    "Position of all atoms in the unit cell in units of the lattice vectors."
    magnetism: RawMagnetism = None
    "Magnetization of every atom in the unit cell."
    __eq__ = _dataclass_equal


@dataclass
class RawKpoints:
    "**k** points at which wave functions are calculated."
    mode: str
    "Mode used to generate the **k**-point list."
    number: int
    "Number of **k** points specified in the generation."
    coordinates: np.ndarray
    "Coordinates of the **k** points as fraction of the reciprocal lattice vectors."
    weights: np.ndarray
    "Weight of the **k** points used for integration."
    cell: RawCell
    "Unit cell of the crystal."
    labels: np.ndarray = None
    "High symmetry label for specific **k** points used in band structures."
    label_indices: np.ndarray = None
    "Indices of the labeled **k** points in the generation list."
    __eq__ = _dataclass_equal


@dataclass
class RawDos:
    "Electronic density of states."
    fermi_energy: float
    "Fermi energy obtained by Vasp."
    energies: np.ndarray
    "Energy E at which the Dos is evaluated."
    dos: np.ndarray
    "Dos at the energies D(E)."
    projections: np.ndarray = None
    "If present, orbital projections of the Dos."
    projectors: RawProjectors = None
    "If present, projector information (element, angular momentum, spin)."
    __eq__ = _dataclass_equal


@dataclass
class RawBand:
    "Electronic band structure"
    fermi_energy: float
    "Fermi energy obtained by Vasp."
    kpoints: RawKpoints
    "**k** points at which the bands are calculated."
    eigenvalues: np.ndarray
    "Calculated eigenvalues at the **k** points."
    occupations: np.ndarray
    "The occupations of the different bands."
    projections: np.ndarray = None
    "If present, orbital projections of the bands."
    projectors: RawProjectors = None
    "If present, projector information (element, angular momentum, spin)."
    __eq__ = _dataclass_equal


@dataclass
class RawEnergy:
    "Various energies during ionic relaxation or MD simulation."
    labels: np.ndarray
    "Label identifying which energy is contained."
    values: np.ndarray
    "Energy specified by labels for all iteration steps."
    __eq__ = _dataclass_equal


@dataclass
class RawDensity:
    "The electronic charge and magnetization density."
    structure: RawStructure
    "The atomic structure to represent the densities."
    charge: np.ndarray
    "The raw data of electronic charge and magnetization density."
    __eq__ = _dataclass_equal


@dataclass
class RawDielectric:
    "The electronic or ionic dielectric function."
    energies: np.ndarray
    "The energies at which the dielectric function is evaluated."
    function: np.ndarray
    "The values of the dielectric tensor at these energies."
    __eq__ = _dataclass_equal
