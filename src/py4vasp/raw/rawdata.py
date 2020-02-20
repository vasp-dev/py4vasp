from dataclasses import dataclass
import numpy as np


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


@dataclass
class Projectors:
    number_ion_types: np.ndarray
    ion_types: np.ndarray
    orbital_types: np.ndarray
    number_spins: int
    __eq__ = _dataclass_equal


@dataclass
class Cell:
    scale: float
    lattice_vectors: np.ndarray
    __eq__ = _dataclass_equal


@dataclass
class Kpoints:
    mode: str
    number: int
    coordinates: np.ndarray
    weights: np.ndarray
    cell: Cell
    labels: np.ndarray = None
    label_indices: np.ndarray = None
    __eq__ = _dataclass_equal


@dataclass
class Dos:
    fermi_energy: float
    energies: np.ndarray
    dos: np.ndarray
    projections: np.ndarray = None
    projectors: Projectors = None
    __eq__ = _dataclass_equal


@dataclass
class Band:
    fermi_energy: float
    line_length: int
    kpoints: np.ndarray
    eigenvalues: np.ndarray
    cell: Cell
    labels: np.ndarray = None
    label_indices: np.ndarray = None
    projections: np.ndarray = None
    projectors: Projectors = None
    __eq__ = _dataclass_equal


@dataclass
class Convergence:
    labels: np.ndarray
    energies: np.ndarray
    __eq__ = _dataclass_equal
