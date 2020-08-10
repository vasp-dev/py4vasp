from .topology import Topology
from py4vasp.data import _util
import mdtraj


class Trajectory:
    def __init__(self, raw_trajectory):
        self._raw = raw_trajectory
        self._topology = Topology(raw_trajectory.topology)

    @classmethod
    def from_file(cls, file=None):
        return _util.from_file(cls, file, "trajectory")

    def read(self):
        return {
            "names": self._topology.names(),
            "elements": self._topology.elements(),
            "positions": self._raw.positions[:],
            "lattice_vectors": self._raw.lattice_vectors[:],
        }

    A_to_pm = 0.1

    def to_mdtraj(self):
        data = self.read()
        xyz = data["positions"] @ data["lattice_vectors"] * Trajectory.A_to_pm
        trajectory = mdtraj.Trajectory(xyz, self._topology.to_mdtraj())
        trajectory.unitcell_vectors = data["lattice_vectors"] * Trajectory.A_to_pm
        return trajectory
