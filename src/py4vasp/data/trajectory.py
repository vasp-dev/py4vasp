from .topology import Topology
from py4vasp.data import _util, Structure
from IPython.lib.pretty import pretty
import py4vasp.raw as raw
import py4vasp.exceptions as exception
import mdtraj
import functools


@_util.add_wrappers
class Trajectory(_util.Data):
    """The trajectory of the ionic positions during the simulation.

    This class provides the functionality to extract trajectories of MD
    simulations for visualization or analysis.

    Parameters
    ----------
    raw_trajectory : raw.Trajectory
        Dataclass containing the raw data of the trajectory.
    """

    def __init__(self, raw_trajectory):
        super().__init__(raw_trajectory)
        self._topology = Topology(raw_trajectory.topology)

    @classmethod
    @_util.add_doc(_util.from_file_doc("trajectory"))
    def from_file(cls, file=None):
        return _util.from_file(cls, file, "trajectory")

    def _repr_pretty_(self, p, cycle):
        p.text(self._create_repr(pretty(self.to_structure()), "\n"))

    def _repr_html_(self):
        return self._create_repr(self.to_structure()._repr_html_(), "<br>")

    def _create_repr(self, structure_repr, end_line):
        old_first_line = structure_repr.partition(end_line)[0]
        new_first_line = f"current structure of {len(self)} step trajectory"
        return structure_repr.replace(old_first_line, new_first_line)

    def __len__(self):
        return len(self._raw.positions)

    def to_dict(self):
        """Extract the trajectory as a dictionary.

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
        """Convert the trajectory to mdtraj.Trajectory

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

    def to_structure(self, step=-1):
        """Convert the trajectory of a particular step to a Structure

        Parameters
        ----------
        step : int
            Specify the step from which the structure is extracted.

        Returns
        -------
        data.Structure
            The structure the trajectory assumes for the specified step.
        """
        _util.raise_error_if_not_number(step, "You can only exctract an integer step.")
        try:
            struct = raw.Structure(
                version=self._raw.version,
                topology=self._raw.topology,
                cell=raw.Cell(
                    self._raw.version, lattice_vectors=self._raw.lattice_vectors[step]
                ),
                positions=self._raw.positions[step],
            )
        except (ValueError, IndexError) as err:
            error_message = (
                f"Error reading step `{step}` from array, please check it is a valid "
                "index within the boundaries of the array."
            )
            raise exception.IncorrectUsage(error_message) from err
        return Structure(struct)
