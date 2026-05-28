# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import copy
import pathlib

import numpy as np

from py4vasp import _config, exception, raw
from py4vasp._calculation import slice_
from py4vasp._calculation.dispatch import (
    DataSource,
    FileSource,
    merge_default,
    merge_strings,
    quantity,
    slice_steps,
)
from py4vasp._calculation.structure import StructureHandler
from py4vasp._raw.data_db import Force_DB
from py4vasp._third_party import view
from py4vasp._util import check


class ForceHandler:
    """Handler for force data — performs all data access and transformation logic."""

    force_rescale = 1.5
    "Scaling constant to convert forces to Å."

    def __init__(self, raw_force: raw.Force, steps=None):
        self._raw_force = raw_force
        self._steps = steps

    @classmethod
    def from_data(cls, raw_force: raw.Force, steps=None) -> "ForceHandler":
        return cls(raw_force, steps=steps)

    def __str__(self) -> str:
        result = """
POSITION                                       TOTAL-FORCE (eV/Angst)
-----------------------------------------------------------------------------------
        """.strip()
        step = self._last_step
        structure = StructureHandler.from_data(self._raw_force.structure, steps=step)
        positions = structure.cartesian_positions()
        forces = np.array(self._raw_force.forces)[step]
        position_to_string = lambda pos: " ".join(f"{x:12.5f}" for x in pos)
        force_to_string = lambda f: " ".join(f"{x:13.6f}" for x in f)
        for position, force in zip(positions, forces):
            result += f"\n{position_to_string(position)}    {force_to_string(force)}"
        return result

    def to_dict(self) -> dict:
        """Read the forces into a dictionary.

        Forces and associated structural information for one or more selected steps of
        the trajectory are returned in a dictionary. This includes the lattice vectors,
        atomic positions, and atomic species in addition to the forces acting on each atom.
        The forces are in Cartesian coordinates and in units of eV/Å.

        Returns
        -------
        dict
            Contains the forces for all selected steps and the structural information
            to know on which atoms the forces act.
        """
        structure = StructureHandler.from_data(self._raw_force.structure, steps=self._steps)
        return {
            "structure": structure.read(),
            "forces": slice_steps(np.array(self._raw_force.forces), self._steps, default_ndim=2),
        }

    def to_database(self) -> dict:
        """Serialize force statistics to the database format."""
        if check.is_none(self._raw_force.forces):
            raise exception.NoData("No force data available to write to database.")
        forces = np.array(self._raw_force.forces)
        if forces.ndim == 2:
            final_force_norms = np.linalg.norm(forces, axis=-1)
            initial_force_norms = final_force_norms.copy()
        else:
            final_force_norms = np.linalg.norm(forces[-1], axis=-1)
            initial_force_norms = np.linalg.norm(forces[0], axis=-1)
        return {
            "force": Force_DB(
                final_force_min=float(np.min(final_force_norms)),
                final_force_median=float(np.median(final_force_norms)),
                final_force_mean=float(np.mean(final_force_norms)),
                final_force_max=float(np.max(final_force_norms)),
                final_index_force_max=int(np.argmax(final_force_norms)),
                initial_force_min=float(np.min(initial_force_norms)),
                initial_force_max=float(np.max(initial_force_norms)),
                initial_index_force_max=int(np.argmax(initial_force_norms)),
            ),
        }

    def to_view(self, supercell=None):
        """Visualize the forces showing arrows at the atoms."""
        structure = StructureHandler.from_data(self._raw_force.structure)
        viewer = structure.to_view(supercell)
        forces = self.force_rescale * slice_steps(
            np.array(self._raw_force.forces), self._steps, default_ndim=2
        )
        if forces.ndim == 2:
            forces = forces[np.newaxis]
        ion_arrow = view.IonArrow(
            quantity=forces,
            label="forces",
            color=_config.VASP_COLORS["purple"],
            radius=0.2,
        )
        viewer.ion_arrows = [ion_arrow]
        return viewer

    def number_steps(self) -> int:
        """Return the number of forces in the trajectory."""
        n = len(np.array(self._raw_force.forces))
        return len(range(n)[self._to_slice])

    @property
    def _last_step(self):
        if self._steps is None or self._steps == -1:
            return -1
        if isinstance(self._steps, slice):
            stop = self._steps.stop
            return (stop - 1) if stop is not None else -1
        return self._steps

    @property
    def _to_slice(self):
        if self._steps is None or self._steps == -1:
            return slice(-1, None)
        if isinstance(self._steps, slice):
            return self._steps
        return slice(self._steps, self._steps + 1)


@quantity("force")
class Force(view.Mixin):
    """The forces determine the path of the atoms in a trajectory.

    You can use this class to analyze the forces acting on the atoms. The forces
    are the first derivative of the DFT total energy. The forces being small is
    an important criterion for the convergence of a relaxation calculation. The
    size of the forces is also related to the maximal time step in MD simulations.
    When you choose a too large time step, the forces become large and the atoms
    may move too much in a single step leading to an unstable trajectory. You can
    use this class to visualize the forces in a trajectory or read the values to
    analyze them numerically.

    Examples
    --------
    Let us create some example data so that we can illustrate how to use this class.
    Of course you can also use your own VASP calculation data if you have it available.

    >>> from py4vasp import demo
    >>> calculation = demo.calculation(path)

    If you access the forces, the result will depend on the steps that you selected
    with the [] operator. Without any selection the results from the final step will be
    used.

    >>> calculation.force.number_steps()
    1

    To select the results for all steps, you don't specify the array boundaries.

    >>> calculation.force[:].number_steps()
    4

    You can also select specific steps or a subset of steps as follows

    >>> calculation.force[3].number_steps()
    1
    >>> calculation.force[1:4].number_steps()
    3
    """

    force_rescale = ForceHandler.force_rescale

    def __init__(self, source, quantity_name: str = "force", steps=None):
        self._source = source
        self._quantity_name = quantity_name
        self._steps = steps

    @classmethod
    def from_data(cls, raw_force: raw.Force) -> "Force":
        """Create a Force dispatcher from raw data (convenience for testing)."""
        return cls(source=DataSource(raw_force))

    @classmethod
    def from_path(cls, path=".") -> "Force":
        """Create a Force dispatcher that reads from HDF5 files at *path*."""
        return cls(source=FileSource(path))

    @classmethod
    def from_file(cls, file_name) -> "Force":
        """Create a Force dispatcher that reads from a specific HDF5 file."""
        resolved = pathlib.Path(file_name).expanduser().resolve()
        return cls(source=FileSource(resolved.parent, file=file_name))

    @property
    def _path(self):
        return self._source.path

    def __getitem__(self, steps) -> "Force":
        new = copy.copy(self)
        new._steps = steps
        return new

    def _handler_factory(self, raw):
        return ForceHandler.from_data(raw, steps=self._steps)

    def __str__(self, selection: str | None = None) -> str:
        "Convert the forces to a format similar to the OUTCAR file."
        return merge_strings(
            self._source,
            self._quantity_name,
            selection,
            self._handler_factory,
            ForceHandler.__str__,
        )

    def _repr_pretty_(self, p, cycle):
        p.text(str(self) if not cycle else "...")

    def read(self, selection: str | None = None) -> dict:
        """Read the forces into a dictionary.

        Forces and associated structural information for one or more selected steps of
        the trajectory are returned in a dictionary. This includes the lattice vectors,
        atomic positions, and atomic species in addition to the forces acting on each atom.
        The forces are in Cartesian coordinates and in units of eV/Å.

        Returns
        -------
        dict
            Contains the forces for all selected steps and the structural information
            to know on which atoms the forces act.

        Examples
        --------
        First, we create some example data so that we can illustrate how to use this method.
        You can also use your own VASP calculation data if you have it available.

        >>> from py4vasp import demo
        >>> calculation = demo.calculation(path)

        If you use the `read` method, the result will depend on the steps that you
        selected with the [] operator. Without any selection the results from the final
        step will be used. The structure is included to provide the necessary context for
        the forces.

        >>> calculation.force.read()
        {'structure': {...}, 'forces': array([[...]])}

        To select the results for all steps, you don't specify the array boundaries.
        Notice that in this case the forces contain an additional dimension for the
        different steps.

        >>> calculation.force[:].read()
        {'structure': {...}, 'forces': array([[[...]]])}

        You can also select specific steps or a subset of steps as follows

        >>> calculation.force[1].read()
        {'structure': {...}, 'forces': array([[...]])}
        >>> calculation.force[0:2].read()
        {'structure': {...}, 'forces': array([[[...]]])}
        """
        return merge_default(
            self._source,
            self._quantity_name,
            selection,
            self._handler_factory,
            ForceHandler.to_dict,
        )

    def to_dict(self, selection: str | None = None) -> dict:
        """Convenient alias for :py:meth:`read`. Please read the documentation there."""
        return self.read(selection=selection)

    def to_view(self, supercell=None) -> view.View:
        """Visualize the forces showing arrows at the atoms.

        This method adds arrows to the atoms in the structure sized according to the
        strength of the force. The length of the arrows is scaled by a constant
        `force_rescale` to convert from eV/Å to a length in Å.

        Parameters
        ----------
        supercell : int or np.ndarray
            If present the structure is replicated the specified number of times
            along each direction.

        Returns
        -------
        View
            Shows the structure with cell and all atoms adding arrows to the atoms
            sized according to the strength of the force.

        Examples
        --------
        First, we create some example data so that we can illustrate how to use this method.
        You can also use your own VASP calculation data if you have it available.

        >>> from py4vasp import demo
        >>> calculation = demo.calculation(path)

        If you use the `to_view` method, the result will depend on the steps that you
        selected with the [] operator. Without any selection the results from the final
        step will be used.

        >>> calculation.force.to_view()
        View(..., ion_arrows=[IonArrow(quantity=array([[...]]), label='forces', ...)], ...)

        To select the results for all steps, you don't specify the array boundaries.

        >>> calculation.force[:].to_view()
        View(..., ion_arrows=[IonArrow(quantity=array([[[...]]]), label='forces', ...)], ...)

        You can also select specific steps or a subset of steps as follows

        >>> calculation.force[1].to_view()
        View(..., ion_arrows=[IonArrow(quantity=array([[...]]), label='forces', ...)], ...)
        >>> calculation.force[0:2].to_view()
        View(..., ion_arrows=[IonArrow(quantity=array([[[...], [...]]]), label='forces', ...)], ...)

        You may also replicate the structure by specifying a supercell.

        >>> calculation.force.to_view(supercell=2)
        View(..., supercell=array([2, 2, 2]), ...)

        The supercell size can also be different for the different directions.

        >>> calculation.force.to_view(supercell=[2,3,1])
        View(..., supercell=array([2, 3, 1]), ...)
        """
        return merge_default(
            self._source,
            self._quantity_name,
            None,
            self._handler_factory,
            ForceHandler.to_view,
            supercell,
        )

    def number_steps(self, selection: str | None = None) -> int:
        """Return the number of forces in the trajectory."""
        return merge_default(
            self._source,
            self._quantity_name,
            selection,
            self._handler_factory,
            ForceHandler.number_steps,
        )

