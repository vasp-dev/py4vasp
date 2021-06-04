from .topology import Topology
from py4vasp.data import _util, Structure, Viewer3d
from py4vasp.data._base import DataBase, RefinementDescriptor
from py4vasp.raw import RawStructure, RawCell
from IPython.lib.pretty import pretty
import py4vasp.exceptions as exception
import functools
import mdtraj


class Trajectory(DataBase):
    """The trajectory of the ionic positions during the simulation.

    This class provides the functionality to extract trajectories of MD
    simulations for visualization or analysis.

    Parameters
    ----------
    raw_trajectory : RawTrajectory
        Dataclass containing the raw data of the trajectory.
    """

    A_to_pm = 0.1

    read = RefinementDescriptor("_to_dict")
    to_dict = RefinementDescriptor("_to_dict")
    plot = RefinementDescriptor("_to_viewer3d")
    to_viewer3d = RefinementDescriptor("_to_viewer3d")
    to_mdtraj = RefinementDescriptor("_to_mdtraj")
    to_structure = RefinementDescriptor("_to_structure")
    __str__ = RefinementDescriptor("_to_string")
    _repr_html_ = RefinementDescriptor("_to_html")
    __len__ = RefinementDescriptor("_length")


def _to_string(raw_traj):
    return _create_repr(str(_to_structure(raw_traj)), _length(raw_traj), "\n")


def _to_html(raw_traj):
    structure_repr = _to_structure(raw_traj)._repr_html_()
    return _create_repr(structure_repr, _length(raw_traj), "<br>")


def _create_repr(structure_repr, step, end_line):
    old_first_line = structure_repr.partition(end_line)[0]
    new_first_line = f"current structure of {step} step trajectory"
    return structure_repr.replace(old_first_line, new_first_line)


def _to_dict(raw_traj):
    """Extract the trajectory as a dictionary.

    Returns
    -------
    dict
        The dictionary contains the labels and elements of all the atoms in
        the simulation and tracks their position as well as the shape of the
        simulation cell throughout the calculation.
    """
    topology = Topology(raw_traj.topology)
    return {
        "names": topology.names(),
        "elements": topology.elements(),
        "positions": raw_traj.positions[:],
        "lattice_vectors": raw_traj.lattice_vectors[:],
    }


def _to_viewer3d(raw_traj):
    """Generate a 3d representation of the trajectory.

    Returns
    -------
    Viewer3d
        Visualize the trajectory as a 3d figure with controls to loop over the history.
    """
    viewer = Viewer3d.from_trajectory(Trajectory(raw_traj))
    viewer.show_cell()
    return viewer


def _to_mdtraj(raw_traj):
    """Convert the trajectory to mdtraj.Trajectory

    Returns
    -------
    mdtraj.Trajectory
        The mdtraj package offers many functionalities to analyze a MD
        trajectory. By converting the Vasp data to their format, we facilitate
        using all functions of that package.
    """
    data = _to_dict(raw_traj)
    xyz = data["positions"] @ data["lattice_vectors"] * Trajectory.A_to_pm
    trajectory = mdtraj.Trajectory(xyz, Topology(raw_traj.topology).to_mdtraj())
    trajectory.unitcell_vectors = data["lattice_vectors"] * Trajectory.A_to_pm
    return trajectory


def _to_structure(raw_traj, step=-1):
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
        struct = RawStructure(
            version=raw_traj.version,
            topology=raw_traj.topology,
            cell=RawCell(
                raw_traj.version, lattice_vectors=raw_traj.lattice_vectors[step]
            ),
            positions=raw_traj.positions[step],
        )
    except (ValueError, IndexError) as err:
        error_message = (
            f"Error reading step `{step}` from array, please check it is a valid "
            "index within the boundaries of the array."
        )
        raise exception.IncorrectUsage(error_message) from err
    return Structure(struct)


def _length(raw_traj):
    return len(raw_traj.positions)
