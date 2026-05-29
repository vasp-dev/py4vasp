# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import copy
import pathlib

import numpy as np

from py4vasp import _config, raw
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
from py4vasp._raw.data_db import Velocity_DB
from py4vasp._third_party import view


class VelocityHandler:
    """Handler for velocity data — performs all data access and transformation logic."""

    velocity_rescale = 200

    def __init__(self, raw_velocity: raw.Velocity, steps=None):
        self._raw_velocity = raw_velocity
        self._steps = steps

    @classmethod
    def from_data(cls, raw_velocity: raw.Velocity, steps=None) -> "VelocityHandler":
        return cls(raw_velocity, steps=steps)

    def __str__(self) -> str:
        step = self._last_step
        structure = StructureHandler.from_data(self._raw_velocity.structure, steps=step)
        velocities = self.to_numpy()
        if velocities.ndim == 3:
            velocities = velocities[-1]
        velocities_str = self._vectors_to_string(velocities)
        return f"{structure}\n\n{velocities_str}"

    def _vectors_to_string(self, vectors):
        return "\n".join(self._vector_to_string(vector) for vector in vectors)

    def _vector_to_string(self, vector):
        return " ".join(self._element_to_string(element) for element in vector)

    def _element_to_string(self, element):
        return f"{element:21.16f}"

    def to_dict(self) -> dict:
        """Return the structure and ion velocities in a dictionary.

        Returns
        -------
        dict
            The dictionary contains the ion velocities as well as the structural
            information for reference.
        """
        structure = StructureHandler.from_data(
            self._raw_velocity.structure, steps=self._steps
        )
        return {
            "structure": structure.to_dict(),
            "velocities": self.to_numpy(),
        }

    def to_numpy(self) -> np.ndarray:
        """Convert the ion velocities for the selected steps into a numpy array."""
        return slice_steps(
            np.array(self._raw_velocity.velocities), self._steps, default_ndim=2
        )

    def to_database(self) -> dict:
        """Serialize velocity statistics to the database format."""
        velocities = np.array(self._raw_velocity.velocities)
        if velocities.ndim == 2:
            final_velocity_norms = np.linalg.norm(velocities, axis=-1)
            initial_velocity_norms = final_velocity_norms.copy()
        else:
            final_velocity_norms = np.linalg.norm(velocities[-1], axis=-1)
            initial_velocity_norms = np.linalg.norm(velocities[0], axis=-1)
        return {
            "velocity": Velocity_DB(
                final_velocity_min=float(np.min(final_velocity_norms)),
                final_velocity_max=float(np.max(final_velocity_norms)),
                final_velocity_mean=float(np.mean(final_velocity_norms)),
                final_velocity_std=(
                    float(np.std(final_velocity_norms))
                    if len(final_velocity_norms) > 1
                    else 0.0
                ),
                final_velocity_median=float(np.median(final_velocity_norms)),
                final_index_velocity_max=int(np.argmax(final_velocity_norms)),
                initial_velocity_min=float(np.min(initial_velocity_norms)),
                initial_velocity_max=float(np.max(initial_velocity_norms)),
                initial_index_velocity_max=int(np.argmax(initial_velocity_norms)),
            )
        }

    def to_view(self, supercell=None):
        """Plot the velocities as vectors in the structure."""
        structure = StructureHandler.from_data(self._raw_velocity.structure)
        viewer = structure.to_view(supercell)
        velocities = self.velocity_rescale * self.to_numpy()
        if velocities.ndim == 2:
            velocities = velocities[np.newaxis]
        ion_arrow = view.IonArrow(
            quantity=velocities,
            label="velocities",
            color=_config.VASP_COLORS["gray"],
            radius=0.2,
        )
        viewer.ion_arrows = [ion_arrow]
        return viewer

    def number_steps(self) -> int:
        """Return the number of velocities in the trajectory."""
        n = len(np.array(self._raw_velocity.velocities))
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


@quantity("velocity")
class Velocity(view.Mixin):
    """The velocities describe the ionic motion during an MD simulation.

    The velocities of the ions are a metric for the temperature of the system. Most
    of the time, it is not necessary to consider them explicitly. VASP will set the
    velocities automatically according to the temperature settings (:tag:`TEBEG` and
    :tag:`TEEND`) unless you set them explicitly in the POSCAR file. Since the
    velocities are not something you typically need, VASP will only store them during
    the simulation if you set :tag:`VELOCITY` = T in the INCAR file. In that case you
    can read the velocities of each step along the trajectory.

    Examples
    --------
    Let us create some example data so that we can illustrate how to use this class.
    Of course you can also use your own VASP calculation data if you have it available.

    >>> from py4vasp import demo
    >>> calculation = demo.calculation(path)

    If you access the velocities, the result will depend on the steps that you selected
    with the [] operator. Without any selection the results from the final step will be
    used.

    >>> calculation.velocity.number_steps()
    1

    To select the results for all steps, you don't specify the array boundaries.

    >>> calculation.velocity[:].number_steps()
    4

    You can also select specific steps or a subset of steps as follows

    >>> calculation.velocity[3].number_steps()
    1
    >>> calculation.velocity[1:4].number_steps()
    3
    """

    velocity_rescale = VelocityHandler.velocity_rescale

    def __init__(self, source, quantity_name: str = "velocity", steps=None):
        self._source = source
        self._quantity_name = quantity_name
        self._steps = steps

    @classmethod
    def from_data(cls, raw_velocity: raw.Velocity) -> "Velocity":
        """Create a Velocity dispatcher from raw data (convenience for testing)."""
        return cls(source=DataSource(raw_velocity))

    @classmethod
    def from_path(cls, path=".") -> "Velocity":
        """Create a Velocity dispatcher that reads from HDF5 files at *path*."""
        return cls(source=FileSource(path))

    @classmethod
    def from_file(cls, file_name) -> "Velocity":
        """Create a Velocity dispatcher that reads from a specific HDF5 file."""
        resolved = pathlib.Path(file_name).expanduser().resolve()
        return cls(source=FileSource(resolved.parent, file=file_name))

    @property
    def _path(self):
        return self._source.path

    def __getitem__(self, steps) -> "Velocity":
        new = copy.copy(self)
        new._steps = steps
        return new

    def _handler_factory(self, raw):
        return VelocityHandler.from_data(raw, steps=self._steps)

    def __str__(self) -> str:
        return merge_strings(
            self._source,
            self._quantity_name,
            None,
            self._handler_factory,
            VelocityHandler.__str__,
        )

    def _repr_pretty_(self, p, cycle):
        p.text(str(self) if not cycle else "...")

    def to_dict(self, selection: str | None = None) -> dict:
        """Convenient alias for :py:meth:`read`."""
        return self.read()

    def read(self) -> dict:
        """Return the structure and ion velocities in a dictionary.

        Returns
        -------
        dict
            The dictionary contains the ion velocities as well as the structural
            information for reference.

        Examples
        --------
        First, we create some example data so that we can illustrate how to use this
        method. You can also use your own VASP calculation data if you have it available.

        >>> from py4vasp import demo
        >>> calculation = demo.calculation(path)

        If you use the `read` method, the result will depend on the steps that you
        selected with the [] operator. Without any selection the results from the final
        step will be used. The structure is included to provide the necessary context
        for the velocities.

        >>> calculation.velocity.read()
        {'structure': {...}, 'velocities': array([[...]])}

        To select the results for all steps, you don't specify the array boundaries.
        Notice that in this case the velocities contain an additional dimension for the
        different steps.

        >>> calculation.velocity[:].read()
        {'structure': {...}, 'velocities': array([[[...]]])}

        You can also select specific steps or a subset of steps as follows

        >>> calculation.velocity[1].read()
        {'structure': {...}, 'velocities': array([[...]])}
        >>> calculation.velocity[0:2].read()
        {'structure': {...}, 'velocities': array([[[...]]])}
        """
        return merge_default(
            self._source,
            self._quantity_name,
            None,
            self._handler_factory,
            VelocityHandler.to_dict,
        )

    def to_numpy(self) -> np.ndarray:
        """Convert the ion velocities for the selected steps into a numpy array.

        The velocities are given in units of Å/fs.

        Returns
        -------
        np.ndarray
            A numpy array of the velocities of the selected steps.

        Examples
        --------
        First, we create some example data so that we can illustrate how to use this
        method. You can also use your own VASP calculation data if you have it available.

        >>> from py4vasp import demo
        >>> calculation = demo.calculation(path)

        If you use the `to_numpy` method, the result will depend on the steps that you
        selected with the [] operator. Without any selection the results from the final
        step will be used.

        >>> calculation.velocity.to_numpy()
        array([[...]])

        To select the results for all steps, you don't specify the array boundaries.
        Notice that in this case the velocities contain an additional dimension for the
        different steps.

        >>> calculation.velocity[:].to_numpy()
        array([[[...]]])

        You can also select specific steps or a subset of steps as follows

        >>> calculation.velocity[1].to_numpy()
        array([[...]])
        >>> calculation.velocity[0:2].to_numpy()
        array([[[...], [...]]])
        """
        return merge_default(
            self._source,
            self._quantity_name,
            None,
            self._handler_factory,
            VelocityHandler.to_numpy,
        )

    def to_view(self, supercell=None) -> view.View:
        """Plot the velocities as vectors in the structure.

        This method adds arrows to the atoms in the structure sized according to the
        size of the velocity. The length of the arrows is scaled by a constant
        `velocity_rescale` to convert from Å/fs to a length in Å.

        Parameters
        ----------
        supercell : int or np.ndarray
            If present the structure is replicated the specified number of times
            along each direction.

        Returns
        -------
        View
            Contains all atoms and the velocities are drawn as vectors.

        Examples
        --------
        First, we create some example data so that we can illustrate how to use this
        method. You can also use your own VASP calculation data if you have it available.

        >>> from py4vasp import demo
        >>> calculation = demo.calculation(path)

        If you use the `to_view` method, the result will depend on the steps that you
        selected with the [] operator. Without any selection the results from the final
        step will be used.

        >>> calculation.velocity.to_view()
        View(..., ion_arrows=[IonArrow(quantity=array([[...]]), label='velocities', ...)], ...)

        To select the results for all steps, you don't specify the array boundaries.

        >>> calculation.velocity[:].to_view()
        View(..., ion_arrows=[IonArrow(quantity=array([[[...]]]), label='velocities', ...)], ...)

        You can also select specific steps or a subset of steps as follows

        >>> calculation.velocity[1].to_view()
        View(..., ion_arrows=[IonArrow(quantity=array([[...]]), label='velocities', ...)], ...)
        >>> calculation.velocity[0:2].to_view()
        View(..., ion_arrows=[IonArrow(quantity=array([[[...], [...]]]), label='velocities', ...)], ...)

        You may also replicate the structure by specifying a supercell.

        >>> calculation.velocity.to_view(supercell=2)
        View(..., supercell=array([2, 2, 2]), ...)

        The supercell size can also be different for the different directions.

        >>> calculation.velocity.to_view(supercell=[2,3,1])
        View(..., supercell=array([2, 3, 1]), ...)
        """
        return merge_default(
            self._source,
            self._quantity_name,
            None,
            self._handler_factory,
            VelocityHandler.to_view,
            supercell,
        )

    def number_steps(self) -> int:
        """Return the number of velocities in the trajectory."""
        return merge_default(
            self._source,
            self._quantity_name,
            None,
            self._handler_factory,
            VelocityHandler.number_steps,
        )
