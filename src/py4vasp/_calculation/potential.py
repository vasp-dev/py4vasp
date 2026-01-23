# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import itertools

import numpy as np

from py4vasp import _config, exception
from py4vasp._calculation import _stoichiometry, base, structure
from py4vasp._third_party import view
from py4vasp._util import density, documentation, index, select, slicing, suggest

VALID_KINDS = ("total", "ionic", "xc", "hartree")
_COMPONENTS = {
    0: ["0", "unity", "sigma_0", "scalar"],
    1: ["1", "sigma_x", "x", "sigma_1"],
    2: ["2", "sigma_y", "y", "sigma_2"],
    3: ["3", "sigma_z", "z", "sigma_3"],
}
_COMMON_PARAMETERS = f"""\
{slicing.PARAMETERS}
supercell : int or np.ndarray
    Replicate the contour plot periodically a given number of times. If you
    provide two different numbers, the resulting cell will be the two remaining
    lattice vectors multiplied by the specific number.
"""


class Potential(base.Refinery, structure.Mixin, view.Mixin):
    """The local potential describes the interactions between electrons and ions.

    In DFT calculations, the local potential consists of various contributions, each
    representing different aspects of the electron-electron and electron-ion
    interactions. The ionic potential arises from the attraction between electrons and
    the atomic nuclei. The Hartree potential accounts for the repulsion between
    electrons resulting from the electron density itself. Additionally, the
    exchange-correlation (xc) potential approximates the effects of electron exchange
    and correlation. The accuracy of this approximation directly influences the
    accuracy of the calculated properties.

    In VASP, the local potential is defined in real space on the FFT grid. You control
    which potentials are written with the :tag:`WRT_POTENTIAL` tag. This class provides
    the methods to read and visualize the potential. If you are interested in the
    average potential, you may also look at the :data:`~py4vasp.calculation.workfunction`.
    """

    @base.data_access
    def __str__(self):
        potential = self._raw_data.total_potential
        if _is_collinear(potential):
            description = "collinear potential:"
        elif _is_noncollinear(potential):
            description = "noncollinear potential:"
        else:
            description = "nonpolarized potential:"
        stoichiometry = _stoichiometry.Stoichiometry.from_data(
            self._raw_data.structure.stoichiometry
        )
        structure = f"structure: {stoichiometry}"
        grid = f"grid: {potential.shape[3]}, {potential.shape[2]}, {potential.shape[1]}"
        available = "available: " + ", ".join(
            kind for kind in VALID_KINDS if not self._get_potential(kind).is_none()
        )
        return "\n    ".join([description, structure, grid, available])

    @base.data_access
    def to_dict(self):
        """Store all available contributions to the potential in a dictionary.

        Returns
        -------
        dict
            The dictionary contains the total potential as well as the potential
            differences between up and down for collinear or the directional potential
            for noncollinear calculations. If individual contributions to the potential
            are available, these are returned, too. Structural information is given for
            reference.
        """
        result = {"structure": self._structure.read()}
        items = [self._generate_items(kind) for kind in VALID_KINDS]
        result.update(itertools.chain(*items))
        return result

    def _generate_items(self, kind):
        potential = self._get_potential(kind)
        if potential.is_none():
            return
        potential = np.moveaxis(potential, 0, -1).T
        yield kind, potential[0]
        if _is_collinear(potential):
            yield f"{kind}_up", potential[0] + potential[1]
            yield f"{kind}_down", potential[0] - potential[1]
        elif _is_noncollinear(potential):
            yield f"{kind}_magnetization", potential[1:]

    @base.data_access
    def to_view(self, selection="total", supercell=None, **user_options):
        """Plot an isosurface of a selected potential.

        Parameters
        ----------
        selection : str
            Select the kind of potential of which you want the isosurface.

        supercell : int or np.ndarray
            If present the data is replicated the specified number of times along each
            direction.

        user_options
            Further arguments with keyword that get directly passed on to the
            visualizer. Most importantly, you can set isolevel (in eV) to adjust the
            value at which the isosurface is drawn.

        Returns
        -------
        View
            A visualization of the potential isosurface within the crystal structure.
        """
        viewer = self._structure.plot(supercell)
        potentials = dict(self._get_potentials(selection))
        visualizer = density.Visualizer(self._structure)
        viewer = visualizer.to_view(potentials, supercell)
        for grid_scalar in viewer.grid_scalars:
            grid_scalar.isosurfaces = [self._create_isosurface(**user_options)]
        return viewer

    def _create_isosurface(self, isolevel=0, color=None, opacity=0.6):
        color = color or _config.VASP_COLORS["cyan"]
        return view.Isosurface(isolevel, color, opacity)

    @base.data_access
    @documentation.format(plane=slicing.PLANE, parameters=_COMMON_PARAMETERS)
    def to_contour(
        self, selection="total", *, a=None, b=None, c=None, normal=None, supercell=None
    ):
        """Generate a 2D contour plot of the selected potential on a slice through the cell.

        {plane}

        Parameters
        ----------
        selection : str, optional
            Specifies which potential to plot. Can be any of the valid kinds
            ("total", "ionic", "xc", "hartree"). For "total" and "xc" potentials
            in spin-polarized calculations, you can select "up" or "down" components
            (e.g., "total_up"). For noncollinear calculations, you can select
            spin components ("x", "y", "z").

        {parameters}

        Returns
        -------
        Graph
            A Graph object containing the contour plot. The plot shows the selected
            potential component on the specified 2D slice.

        Examples
        --------

        Cut a plane through the potential at the origin of the third lattice vector.

        >>> calculation.potential.to_contour(c=0)

        Plot the Hartree potential in the (100) plane crossing at 0.5 fractional coordinate

        >>> calculation.potential.to_contour(selection="hartree", a=0.5)

        Plot the sigma_z-component of the xc potential in a 2x2 supercell in the plane
        defined by the first two lattice vectors.

        >>> calculation.potential.to_contour(selection="xc_z", c=0.2, supercell=2)
        """
        potentials = dict(self._get_potentials(selection))
        visualizer = density.Visualizer(self._structure)
        slice_arguments = density.SliceArguments(a, b, c, normal, supercell)
        return visualizer.to_contour(potentials, slice_arguments, isolevels=False)

    @base.data_access
    @documentation.format(plane=slicing.PLANE, parameters=_COMMON_PARAMETERS)
    def to_quiver(
        self, selection="total", *, a=None, b=None, c=None, normal=None, supercell=None
    ):
        """Generate a 2D quiver plot of the magnetic part of the potential on a slice.

        This method visualizes the vector field of the magnetization potential
        (difference between spin-up and spin-down potentials for collinear cases,
        or the vector components for noncollinear cases) as arrows on a 2D slice
        through the simulation cell.

        {plane}

        Parameters
        ----------
        selection : str, optional
            Specifies which magnetic potential to plot. It must be a kind that
            can have a magnetic component, i.e., "total" or "xc".
            Component selection is not allowed for quiver plots as it inherently plots
            the vector nature of the magnetization.

        {parameters}

        Returns
        -------
        Graph
            A Graph object containing the quiver plot. The plot shows
            arrows representing the magnitude and direction of the magnetic
            potential in the specified 2D slice.

        Examples
        --------
        Plot the magnetization of the total potential in the a-b plane

        >>> calculation.potential.to_quiver(c=0)

        Plot the magnetization of the xc potential in the (010) plane
        crossing at 0.25 fractional coordinate, using a 2x2 supercell.

        >>> calculation.potential.to_quiver(selection="xc", b=0.25, supercell=2)
        """
        potentials = dict(self._get_potentials(selection, is_magnetic=True))
        visualizer = density.Visualizer(self._structure)
        slice_arguments = density.SliceArguments(a, b, c, normal, supercell)
        return visualizer.to_quiver(potentials, slice_arguments)

    def _get_potentials(self, selection, is_magnetic=False):
        tree = select.Tree.from_selection(selection)
        for selection in tree.selections():
            kind, component = self._determine_kind_and_component(selection)
            selector = self._create_selector(kind, component, is_magnetic)
            component_label = component[0] if component else ""
            yield self._get_label(kind, component_label), selector[component].T

    def _determine_kind_and_component(self, selection):
        for kind in VALID_KINDS:
            if kind in selection:
                remaining = list(selection)
                remaining.remove(kind)
                return kind, tuple(remaining)
        return "total", selection

    def _get_label(self, kind, component):
        return f"{kind} potential" + (f"({component})" if component else "")

    def _create_selector(self, kind, component, is_magnetic):
        if is_magnetic:
            return self._create_magnetic_selector(kind, component)
        else:
            return self._create_nonmagnetic_selector(kind)

    def _create_magnetic_selector(self, kind, component):
        _raise_error_if_kind_incorrect(kind, ("total", "xc"))
        _raise_error_if_component_selected(component)
        potential = self._get_potential(kind)
        _raise_error_if_nonpolarized_potential(potential)
        return index.Selector(maps={}, data=potential, reduction=_PotentialReduction)

    def _create_nonmagnetic_selector(self, kind):
        potential = self._get_potential(kind)
        maps = {0: self._create_map(potential)}
        return index.Selector(maps, potential, reduction=_PotentialReduction)

    def _get_potential(self, kind):
        return getattr(self._raw_data, f"{kind}_potential")

    def _create_map(self, potential):
        if _is_nonpolarized(potential):
            return {choice: 0 for choice in _COMPONENTS[0]}
        elif _is_collinear(potential):
            return {
                **{choice: 0 for choice in _COMPONENTS[0]},
                **{choice: 1 for choice in _COMPONENTS[3]},
                **{"up": slice(None), "down": slice(None)},
            }
        return {
            choice: component
            for component, choices in _COMPONENTS.items()
            for choice in choices
        }


class _PotentialReduction(index.Reduction):
    def __init__(self, keys):
        self._selection = keys[0]

    def __call__(self, array, axis):
        if self._is_magnetic_potential(axis):
            return np.moveaxis(array[1:], 0, -1)
        if self._selection == "up":
            return array[0] + array[1]
        if self._selection == "down":
            return array[0] - array[1]
        return array[0]

    def _is_magnetic_potential(self, axis):
        return axis == ()  # for magnetic potentials, we do not remove the first axis


def _is_nonpolarized(potential):
    return potential.shape[0] == 1


def _is_collinear(potential):
    return potential.shape[0] == 2


def _is_noncollinear(potential):
    return potential.shape[0] == 4


def _raise_error_if_kind_incorrect(kind, valid_kinds=VALID_KINDS):
    if kind in valid_kinds:
        return
    message = f"""\
The selection "{kind}" is not a selection for the potential. Only the following \
selections are allowed: "{'", "'.join(VALID_KINDS)}". \
{suggest.did_you_mean(kind, valid_kinds)}Please check for spelling errors."""
    raise exception.IncorrectUsage(message)


def _raise_error_if_component_selected(component):
    if not component:
        return
    message = f"Selecting a component {component} is not implemented for quiver plots."
    raise exception.NotImplemented(message)


def _raise_error_if_no_data(data, kind="total"):
    if data.is_none():
        message = f"Cannot find the {kind} potential data. "
        if kind == "total":
            message += (
                "Did you set LVTOT = T or WRT_POTENTIAL = total in the INCAR file?"
            )
        else:
            message += f"Did you set WRT_POTENTIAL = {kind} in the INCAR file?"
        raise exception.NoData(message)


def _raise_error_if_nonpolarized_potential(potential):
    if _is_nonpolarized(potential):
        message = "Cannot visualize nonpolarized potential as quiver plot."
        raise exception.DataMismatch(message)
