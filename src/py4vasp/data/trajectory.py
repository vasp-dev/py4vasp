from .topology import Topology
from py4vasp.data import _util
import mdtraj
import functools


@_util.add_wrappers
class Trajectory:
    """ The trajectory of the ionic positions during the simulation.

    This class provides the functionality to extract trajectories of MD
    simulations for visualization or analysis.

    Parameters
    ----------
    raw_trajectory : raw.Trajectory
        Dataclass containing the raw data of the trajectory.
    """

    def __init__(self, raw_trajectory):
        self._raw = raw_trajectory
        self._topology = Topology(raw_trajectory.topology)

    @classmethod
    @_util.add_doc(_util.from_file_doc("trajectory"))
    def from_file(cls, file=None):
        return _util.from_file(cls, file, "trajectory")

    def to_dict(self):
        """ Extract the trajectory as a dictionary.

        Returns
        -------
        dict
            The dictionary contains the labels and elements of all the atoms in
            the simulation and tracks their position as well as the shape of the
            simulation cell throughout the calculation.
        """
        return {
            "names": self._topology.names(),
            "elements": self._topology.elements(),
            "positions": self._raw.positions[:],
            "lattice_vectors": self._raw.lattice_vectors[:],
        }

    A_to_pm = 0.1

    def to_mdtraj(self):
        """ Convert the trajectory to mdtraj.Trajectory

        Returns
        -------
        mdtraj.Trajectory
            The mdtraj package offers many functionalities to analyze a MD
            trajectory. By converting the Vasp data to their format, we facilitate
            using all functions of that package.
        """
        data = self.read()
        xyz = data["positions"] @ data["lattice_vectors"] * Trajectory.A_to_pm
        trajectory = mdtraj.Trajectory(xyz, self._topology.to_mdtraj())
        trajectory.unitcell_vectors = data["lattice_vectors"] * Trajectory.A_to_pm
        return trajectory
