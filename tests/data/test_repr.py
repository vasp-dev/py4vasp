from py4vasp.raw import *
from py4vasp.data import *
from .test_band import raw_band
from .test_density import raw_density
from .test_dos import nonmagnetic_Dos
from .test_energy import reference_energy
from .test_kpoints import raw_kpoints
from .test_magnetism import raw_magnetism
from .test_projectors import without_spin
from .test_structure import raw_structure
from .test_topology import raw_topology
from .test_trajectory import raw_trajectory
from numpy import array


def test_repr(
    raw_band,
    raw_density,
    nonmagnetic_Dos,
    reference_energy,
    raw_kpoints,
    raw_magnetism,
    without_spin,
    raw_structure,
    raw_topology,
    raw_trajectory,
):
    tests = (
        Band(raw_band),
        Density(raw_density),
        Dos(nonmagnetic_Dos),
        Energy(reference_energy),
        Kpoints(raw_kpoints),
        Magnetism(raw_magnetism),
        Projectors(without_spin),
        Structure(raw_structure),
        Topology(raw_topology),
        Trajectory(raw_trajectory),
    )
    for test in tests:
        copy = eval(repr(test))
        assert copy.__class__ == test.__class__
