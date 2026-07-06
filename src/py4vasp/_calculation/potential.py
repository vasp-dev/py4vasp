# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import copy
import itertools
from typing import Optional, Union

import numpy as np

from py4vasp import _config, exception
from py4vasp._calculation import _stoichiometry
from py4vasp._calculation.dispatch import (
    DataSource,
    _dispatch,
    merge_default,
    merge_strings,
    merge_to_database,
    quantity,
)
from py4vasp._calculation.structure import StructureHandler
from py4vasp._raw import data as raw
from py4vasp._raw.data_db import Potential_DB
from py4vasp._third_party import view
from py4vasp._util import (
    check,
    density,
    documentation,
    index,
    select,
    slicing,
    suggest,
)

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


class PotentialHandler:
    """Handler for potential data — performs all data access and transformation."""

    def __init__(self, raw_potential: raw.Potential):
        self._raw_potential = raw_potential

    @classmethod
    def from_data(cls, raw_potential: raw.Potential) -> "PotentialHandler":
        return cls(raw_potential)

    def __str__(self) -> str:
        potential = self._raw_potential.total_potential
        if _is_collinear(potential):
            description = "collinear potential:"
        elif _is_noncollinear(potential):
            description = "noncollinear potential:"
        else:
            description = "nonpolarized potential:"
        stoichiometry = _stoichiometry.Stoichiometry.from_data(
            self._raw_potential.structure.stoichiometry
        )
        structure = f"structure: {stoichiometry}"
        grid = f"grid: {potential.shape[3]}, {potential.shape[2]}, {potential.shape[1]}"
        available = "available: " + ", ".join(
            kind for kind in VALID_KINDS if not self._get_potential(kind).is_none()
        )
        return "\n    ".join([description, structure, grid, available])

    def to_dict(self) -> dict:
        result = {"structure": self._structure().to_dict()}
        items = [self._generate_items(kind) for kind in VALID_KINDS]
        result.update(itertools.chain(*items))
        return result

    def to_database(self) -> Potential_DB:
        total_potential_mean = None
        total_potential_mean_up = None
        total_potential_mean_down = None
        total_potential_mean_magnetization = None
        total_potential = self._get_potential("total")
        if not check.is_none(total_potential):
            total_potential = np.moveaxis(total_potential, 0, -1).T
            total_potential_mean = float(np.mean(total_potential[0]))
            total_potential_mean_up = (
                float(np.mean(total_potential[0] + total_potential[1]))
                if _is_collinear(total_potential)
                else total_potential_mean / 2.0
            )
            total_potential_mean_down = (
                float(np.mean(total_potential[0] - total_potential[1]))
                if _is_collinear(total_potential)
                else total_potential_mean / 2.0
            )
            total_potential_mean_magnetization = (
                float(np.mean(np.linalg.norm(total_potential[1:], axis=-1)))
                if _is_noncollinear(total_potential)
                else None
            )

        has_potential_dict = {
            f"has_{kind}_potential": not check.is_none(self._get_potential(kind))
            for kind in VALID_KINDS
        }

        return Potential_DB(
            **has_potential_dict,
            total_potential_mean=total_potential_mean,
            total_potential_mean_up=total_potential_mean_up,
            total_potential_mean_down=total_potential_mean_down,
            total_potential_mean_magnetization=total_potential_mean_magnetization,
        )

    def to_view(
        self,
        selection: str = "total",
        supercell: Optional[Union[int, np.ndarray]] = None,
        **user_options,
    ):
        potentials = dict(self._get_potentials(selection))
        isosurface = self._create_isosurface(**user_options)
        viewer = self._structure().to_view(supercell)
        viewer.grid_scalars = [
            view.GridQuantity(
                quantity=data[np.newaxis],
                label=label,
                isosurfaces=[isosurface],
            )
            for label, data in potentials.items()
        ]
        return viewer

    def to_contour(
        self,
        selection: str = "total",
        *,
        a: Optional[float] = None,
        b: Optional[float] = None,
        c: Optional[float] = None,
        normal: Optional[str] = None,
        supercell: Optional[Union[int, np.ndarray]] = None,
    ):
        potentials = dict(self._get_potentials(selection))
        visualizer = density.Visualizer(self._structure())
        slice_arguments = density.SliceArguments(a, b, c, normal, supercell)
        return visualizer.to_contour(potentials, slice_arguments, isolevels=False)

    def to_quiver(
        self, selection="total", *, a=None, b=None, c=None, normal=None, supercell=None
    ):
        potentials = dict(self._get_potentials(selection, is_magnetic=True))
        visualizer = density.Visualizer(self._structure())
        slice_arguments = density.SliceArguments(a, b, c, normal, supercell)
        return visualizer.to_quiver(potentials, slice_arguments)

    def _structure(self):
        return StructureHandler.from_data(self._raw_potential.structure)

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
        return getattr(self._raw_potential, f"{kind}_potential")

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

    def _create_isosurface(self, isolevel=0, color=None, opacity=0.6):
        color = color or _config.VASP_COLORS["cyan"]
        return view.Isosurface(isolevel, color, opacity)

    def _generate_items(self, kind):
        potential = self._get_potential(kind)
        if check.is_none(potential):
            return
        potential = np.moveaxis(potential, 0, -1).T
        yield kind, potential[0]
        if _is_collinear(potential):
            yield f"{kind}_up", potential[0] + potential[1]
            yield f"{kind}_down", potential[0] - potential[1]
        elif _is_noncollinear(potential):
            yield f"{kind}_magnetization", potential[1:]


@quantity("potential")
class Potential(view.Mixin):
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
    average potential, you may also look at the
    :data:`~py4vasp.calculation.workfunction`.

    Examples
    --------
    First, we create some example data do that you can follow along. Please define a
    variable `path` with the path to a directory that exists and does not contain any
    VASP calculation data. Alternatively, you can use your own data if you have run
    VASP and construct `calculation` from it.

    >>> from py4vasp import demo
    >>> calculation = demo.calculation(path)

    See some basic information about the Potential by printing the object:

    >>> print(calculation.potential)
    nonpolarized potential:...

    For your own postprocessing, you can read the potential data into a Python dictionary:

    >>> calculation.potential.read()
    {'structure': {...}, 'total': array([[[...]]], ...), 'ionic': array([[[...]]], ...), 'xc': array([[[...]]], ...), 'hartree': array([[[...]]], ...)}

    You can also plot the 3d isosurface of the selected potential:

    >>> calculation.potential.plot()
    View(...)

    Alternatively, you can visualize a contour plot of the potential in a plane:

    >>> calculation.potential.to_contour(c=0)
    Graph(series=[Contour(data=array([[...]]...), ..., cut='c', ...)], ...)

    You can check possible selections for the potential:

    >>> calculation.potential.selections()
    {'potential': ['default'...]...}

    Please check the documentation of each of these methods for more details on
    how to use them and which options they provide.
    """

    def __init__(self, source, quantity_name="potential"):
        self._source = source
        self._quantity_name = quantity_name

    @classmethod
    def from_data(cls, raw_potential):
        return cls(source=DataSource(raw_potential))

    def _handler_factory(self, raw):
        return PotentialHandler.from_data(raw)

    def __str__(self, selection=None):
        return merge_strings(
            self._source,
            self._quantity_name,
            selection,
            self._handler_factory,
            PotentialHandler.__str__,
        )

    def _repr_pretty_(self, p, cycle):
        p.text(str(self))

    def read(self) -> dict:
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
        return merge_default(
            self._source,
            self._quantity_name,
            None,
            self._handler_factory,
            PotentialHandler.to_dict,
        )

    def to_dict(self) -> dict:
        """Convenient alias for :py:meth:`read`. Please read the documentation there."""
        return self.read()

    def to_view(
        self,
        selection: str = "total",
        supercell: Optional[Union[int, np.ndarray]] = None,
        **user_options,
    ):
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
        return merge_default(
            self._source,
            self._quantity_name,
            selection,
            self._handler_factory,
            PotentialHandler.to_view,
            supercell=supercell,
            **user_options,
        )

    @documentation.format(plane=slicing.PLANE, parameters=_COMMON_PARAMETERS)
    def to_contour(
        self,
        selection: str = "total",
        *,
        a: Optional[float] = None,
        b: Optional[float] = None,
        c: Optional[float] = None,
        normal: Optional[str] = None,
        supercell: Optional[Union[int, np.ndarray]] = None,
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
        return merge_default(
            self._source,
            self._quantity_name,
            selection,
            self._handler_factory,
            PotentialHandler.to_contour,
            a=a,
            b=b,
            c=c,
            normal=normal,
            supercell=supercell,
        )

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
        return merge_default(
            self._source,
            self._quantity_name,
            selection,
            self._handler_factory,
            PotentialHandler.to_quiver,
            a=a,
            b=b,
            c=c,
            normal=normal,
            supercell=supercell,
        )

    def _to_database(self) -> dict:
        """Return {quantity[_selection]: handler_result} for database storage."""
        return merge_to_database(
            self._source,
            self._quantity_name,
            PotentialHandler.from_data,
            PotentialHandler.to_database,
        )


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
        return axis == ()


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
