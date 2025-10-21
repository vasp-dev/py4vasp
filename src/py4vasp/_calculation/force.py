# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import numpy as np

from py4vasp import _config
from py4vasp._calculation import base, slice_, structure
from py4vasp._third_party import view
from py4vasp._util import documentation, reader


class Force(slice_.Mixin, base.Refinery, structure.Mixin, view.Mixin):
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

    force_rescale = 1.5
    "Scaling constant to convert forces to Å."

    @base.data_access
    def __str__(self):
        "Convert the forces to a format similar to the OUTCAR file."
        result = """
POSITION                                       TOTAL-FORCE (eV/Angst)
-----------------------------------------------------------------------------------
        """.strip()
        step = self._last_step_in_slice
        position_to_string = lambda position: " ".join(f"{x:12.5f}" for x in position)
        positions = self._structure[step].cartesian_positions()
        force_to_string = lambda force: " ".join(f"{x:13.6f}" for x in force)
        for position, force in zip(positions, self._force[step]):
            result += f"\n{position_to_string(position)}    {force_to_string(force)}"
        return result

    @base.data_access
    def to_dict(self):
        """Read the forces and associated structural information for one or more
        selected steps of the trajectory.

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

        If you use the `to_dict` method, the result will depend on the steps that you
        selected with the [] operator. Without any selection the results from the final
        step will be used. The structure is included to provide the necessary context for
        the forces.

        >>> calculation.force.to_dict()
        {'structure': {...}, 'forces': array([[...]])}

        To select the results for all steps, you don't specify the array boundaries.
        Notice that in this case the forces contain an additional dimension for the
        different steps.

        >>> calculation.force[:].to_dict()
        {'structure': {...}, 'forces': array([[[...]]])}

        You can also select specific steps or a subset of steps as follows

        >>> calculation.force[1].to_dict()
        {'structure': {...}, 'forces': array([[...]])}
        >>> calculation.force[0:2].to_dict()
        {'structure': {...}, 'forces': array([[[...]]])}
        """
        return {
            "structure": self._structure[self._steps].read(),
            "forces": self._force[self._steps],
        }

    @base.data_access
    def to_view(self, supercell=None):
        """Visualize the forces showing arrows at the atoms.

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
        Notice that in this case the lattice vectors and positions contain an additional
        dimension for the different steps.

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
        viewer = self._structure.plot(supercell)
        forces = self.force_rescale * self._force[self._steps]
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

    @base.data_access
    def number_steps(self):
        """Return the number of forces in the trajectory."""
        range_ = range(len(self._raw_data.forces))
        return len(range_[self._slice])

    @property
    def _force(self):
        return _ForceReader(self._raw_data.forces)


class _ForceReader(reader.Reader):
    def error_message(self, key, err):
        key = np.array(key)
        steps = key if key.ndim == 0 else key[0]
        return (
            f"Error reading the forces. Please check if the steps "
            f"`{steps}` are properly formatted and within the boundaries. "
            "Additionally, you may consider the original error message:\n" + err.args[0]
        )
