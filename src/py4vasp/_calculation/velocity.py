# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import numpy as np

from py4vasp import _config
from py4vasp._calculation import base, slice_, structure
from py4vasp._third_party import view
from py4vasp._util import reader


class Velocity(slice_.Mixin, base.Refinery, structure.Mixin, view.Mixin):
    """The velocities describe the ionic motion during an MD simulation.

    The velocities of the ions are a metric for the temperature of the system. Most
    of the time, it is not necessary to consider them explicitly. VASP will set the
    velocities automatically according to the temperature settings (:tag:`TEBEG` and
    :tag:`TEEND`) unless you set them explicitly in the POSCAR file. Since the
    velocities are not something you typically need, VASP will only store them during
    the simulation if you set :tag:`VELOCITY` = T in the INCAR file. In that case you
    can read the velocities of each step along the trajectory. If you are only
    interested in the final velocities, please consider the :data:'~py4vasp.data.CONTCAR`
    class.

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

    velocity_rescale = 200

    @base.data_access
    def __str__(self):
        step = self._last_step_in_slice
        velocities = _VelocityReader(self._raw_data.velocities)[
            self._last_step_in_slice
        ]
        velocities = self._vectors_to_string(velocities)
        return f"{self._structure[step]}\n\n{velocities}"

    def _vectors_to_string(self, vectors):
        return "\n".join(self._vector_to_string(vector) for vector in vectors)

    def _vector_to_string(self, vector):
        return " ".join(self._element_to_string(element) for element in vector)

    def _element_to_string(self, element):
        return f"{element:21.16f}"

    @base.data_access
    def to_dict(self):
        """Return the structure and ion velocities in a dictionary

        Returns
        -------
        dict
            The dictionary contains the ion velocities as well as the structural
            information for reference.

        Examples
        --------
        First, we create some example data so that we can illustrate how to use this method.
        You can also use your own VASP calculation data if you have it available.

        >>> from py4vasp import demo
        >>> calculation = demo.calculation(path)

        If you use the `to_dict` method, the result will depend on the steps that you
        selected with the [] operator. Without any selection the results from the final
        step will be used. The structure is included to provide the necessary context for
        the velocities.

        >>> calculation.velocity.to_dict()
        {'structure': {...}, 'velocities': array([[...]])}

        To select the results for all steps, you don't specify the array boundaries.
        Notice that in this case the velocities contain an additional dimension for the
        different steps.

        >>> calculation.velocity[:].to_dict()
        {'structure': {...}, 'velocities': array([[[...]]])}

        You can also select specific steps or a subset of steps as follows

        >>> calculation.velocity[1].to_dict()
        {'structure': {...}, 'velocities': array([[...]])}
        >>> calculation.velocity[0:2].to_dict()
        {'structure': {...}, 'velocities': array([[[...]]])}
        """
        return {
            "structure": self._structure[self._steps].read(),
            "velocities": self.to_numpy(),
        }

    @base.data_access
    def to_view(self, supercell=None):
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
        First, we create some example data so that we can illustrate how to use this method.
        You can also use your own VASP calculation data if you have it available.

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
        viewer = self._structure.plot(supercell)
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

    @base.data_access
    def to_numpy(self) -> np.ndarray:
        """Convert the ion velocities for the selected steps into a numpy array.

        The velocities are given in units of Å/fs.

        Returns
        -------
        A numpy array of the velocities of the selected steps.

        Examples
        --------
        First, we create some example data so that we can illustrate how to use this method.
        You can also use your own VASP calculation data if you have it available.

        >>> from py4vasp import demo
        >>> calculation = demo.calculation(path)

        If you use the `to_numpy` method, the result will depend on the steps that you
        selected with the [] operator. Without any selection the results from the final
        step will be used. The structure is included to provide the necessary context for
        the velocities.

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
        return _VelocityReader(self._raw_data.velocities)[self._steps]

    @base.data_access
    def number_steps(self):
        """Return the number of velocities in the trajectory."""
        range_ = range(len(self._raw_data.velocities))
        return len(range_[self._slice])


class _VelocityReader(reader.Reader):
    def error_message(self, key, err):
        key = np.array(key)
        steps = key if key.ndim == 0 else key[0]
        return (
            f"Error reading the velocities. Please check if the steps "
            f"`{steps}` are properly formatted and within the boundaries. "
            "Additionally, you may consider the original error message:\n" + err.args[0]
        )
