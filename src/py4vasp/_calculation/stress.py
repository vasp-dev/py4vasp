# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import copy

import numpy as np

from py4vasp import raw
from py4vasp._calculation import slice_
from py4vasp._calculation.dispatch import (
    DataSource,
    _dispatch,
    merge_default,
    merge_strings,
    merge_to_database,
    quantity,
    slice_steps,
)
from py4vasp._calculation.structure import StructureHandler
from py4vasp._raw.models import StressModel
from py4vasp._util import tensor


class StressHandler:
    """Handler for stress data — performs all data access and transformation logic."""

    def __init__(self, raw_stress: raw.Stress, steps=None):
        self._raw_stress = raw_stress
        self._steps = steps

    @classmethod
    def from_data(cls, raw_stress: raw.Stress, steps=None) -> "StressHandler":
        return cls(raw_stress, steps=steps)

    def __str__(self) -> str:
        "Convert the stress to a format similar to the OUTCAR file."
        step = self._last_step
        structure = StructureHandler.from_data(self._raw_stress.structure, steps=step)
        eV_to_kB = 1.602176634e3 / structure.volume()
        stress_arr = np.array(self._raw_stress.stress)[step]
        stress = tensor.symmetry_reduce(stress_arr)
        stress_to_string = lambda s: " ".join(f"{x:11.5f}" for x in s)
        return f"""
FORCE on cell =-STRESS in cart. coord.  units (eV):
Direction    XX          YY          ZZ          XY          YZ          ZX
-------------------------------------------------------------------------------------
Total   {stress_to_string(stress / eV_to_kB)}
in kB   {stress_to_string(stress)}
""".strip()

    def to_dict(self) -> dict:
        """Read the stress and associated structural information for one or more
        selected steps of the trajectory.

        Returns
        -------
        dict
            Contains the stress for all selected steps and the structural information
            to know on which cell the stress acts.
        """
        structure = StructureHandler.from_data(
            self._raw_stress.structure, steps=self._steps
        )
        return {
            "stress": slice_steps(
                np.array(self._raw_stress.stress), self._steps, default_ndim=2
            ),
            "structure": structure.to_dict(),
        }

    def to_database(self) -> dict:
        """Serialize stress statistics to the database format."""
        stress = np.array(self._raw_stress.stress)
        if stress.ndim == 3:
            initial_stress_tensor = stress[0]
            final_stress_tensor = stress[-1]
        else:
            initial_stress_tensor = stress
            final_stress_tensor = stress
        return StressModel(
            initial_stress_mean=np.trace(initial_stress_tensor) / 3.0,
            final_stress_mean=np.trace(final_stress_tensor) / 3.0,
            final_stress_tensor=tensor.symmetry_reduce(final_stress_tensor),
        )

    def number_steps(self) -> int:
        """Return the number of stress components in the trajectory."""
        n = len(np.array(self._raw_stress.stress))
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


@quantity("stress")
class Stress:
    """The stress describes the force acting on the shape of the unit cell.

    The stress refers to the force applied to the cell per unit area. Specifically,
    VASP computes the stress for a given unit cell and relaxing to vanishing stress
    determines the predicted ground-state cell. The stress is 3 x 3 matrix; the trace
    indicates changes to the volume and the rest of the matrix changes the shape of
    the cell. You can impose an external stress with the tag :tag:`PSTRESS`.

    When you relax the system or in a MD simulation, VASP computes and stores the
    stress in every iteration. You can use this class to read the stress for specific
    steps along the trajectory.

    Examples
    --------
    Let us create some example data so that we can illustrate how to use this class.
    Of course you can also use your own VASP calculation data if you have it available.

    >>> from py4vasp import demo
    >>> calculation = demo.calculation(path)

    If you access the stress, the result will depend on the steps that you selected
    with the [] operator. Without any selection the results from the final step will be
    used.

    >>> calculation.stress.number_steps()
    1

    To select the results for all steps, you don't specify the array boundaries.

    >>> calculation.stress[:].number_steps()
    4

    You can also select specific steps or a subset of steps as follows

    >>> calculation.stress[3].number_steps()
    1
    >>> calculation.stress[1:4].number_steps()
    3
    """

    def __init__(self, source, quantity_name: str = "stress", steps=None):
        self._source = source
        self._quantity_name = quantity_name
        self._steps = steps

    @classmethod
    def from_data(cls, raw_stress: raw.Stress) -> "Stress":
        """Create a Stress dispatcher from raw data (convenience for testing)."""
        return cls(source=DataSource(raw_stress))

    def __getitem__(self, steps) -> "Stress":
        new = copy.copy(self)
        new._steps = steps
        return new

    def _handler_factory(self, raw):
        return StressHandler.from_data(raw, steps=self._steps)

    def __str__(self, selection=None) -> str:
        "Convert the stress to a format similar to the OUTCAR file."
        return merge_strings(
            self._source,
            self._quantity_name,
            selection,
            self._handler_factory,
            StressHandler.__str__,
        )

    def _repr_pretty_(self, p, cycle):
        p.text(str(self) if not cycle else "...")

    def to_dict(self, selection: str | None = None) -> dict:
        """Convenient alias for :py:meth:`read`. Please read the documentation there."""
        return self.read()

    def read(self) -> dict:
        """Read the stress and associated structural information for one or more
        selected steps of the trajectory.

        Returns
        -------
        dict
            Contains the stress for all selected steps and the structural information
            to know on which cell the stress acts.

        Examples
        --------
        First, we create some example data so that we can illustrate how to use this method.
        You can also use your own VASP calculation data if you have it available.

        >>> from py4vasp import demo
        >>> calculation = demo.calculation(path)

        If you use the `read` method, the result will depend on the steps that you
        selected with the [] operator. Without any selection the results from the final
        step will be used. The structure is included to provide the necessary context for
        the stress.

        >>> calculation.stress.read()
        {'structure': {...}, 'stress': array([[...]])}

        To select the results for all steps, you don't specify the array boundaries.
        Notice that in this case the stress contains an additional dimension for the
        different steps.

        >>> calculation.stress[:].read()
        {'structure': {...}, 'stress': array([[[...]]])}

        You can also select specific steps or a subset of steps as follows

        >>> calculation.stress[1].read()
            {'structure': {...}, 'stress': array([[...]])}
        >>> calculation.stress[0:2].read()
        {'structure': {...}, 'stress': array([[[...]]])}
        """
        return merge_default(
            self._source,
            self._quantity_name,
            None,
            self._handler_factory,
            StressHandler.to_dict,
        )

    def number_steps(self) -> int:
        """Return the number of stress components in the trajectory."""
        return merge_default(
            self._source,
            self._quantity_name,
            None,
            self._handler_factory,
            StressHandler.number_steps,
        )

    def _to_database(self) -> dict:
        """Return {quantity[_selection]: handler_result} for database storage."""
        return merge_to_database(
            self._source,
            self._quantity_name,
            StressHandler.from_data,
            StressHandler.to_database,
        )
