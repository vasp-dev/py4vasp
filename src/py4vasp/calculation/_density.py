# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import numpy as np

from py4vasp import _config, calculation, exception
from py4vasp._third_party import graph, view
from py4vasp._util import documentation, import_, index, select, slicing
from py4vasp.calculation import _base, _structure

pretty = import_.optional("IPython.lib.pretty")


_DEFAULT = 0
_INTERNAL = "_density"
_COMPONENTS = {
    0: ["0", "unity", "sigma_0", "scalar", _INTERNAL],
    1: ["1", "sigma_x", "x", "sigma_1"],
    2: ["2", "sigma_y", "y", "sigma_2"],
    3: ["3", "sigma_z", "z", "sigma_3"],
}
_MAGNETIZATION = ("magnetization", "mag", "m")

_PLANE = """\
You need to specify a plane defined by two of the lattice vectors by selecting
a *cut* along the third one. You must only select a single cut and the value
should correspond to the fractional length along this third lattice vector.
py4vasp will then create a plane from the other two lattice vectors and
generate a contour plot within this plane.

Usually, the first remaining lattice vector is aligned with the x-axis and the
second one such that the angle between the vectors is preserved. You can
overwrite this choice by defining a normal direction. Then py4vasp will rotate
the normal vector of the plane to align with the specified direction. This is
particularly useful if the lattice vector you cut is aligned with a Cartesian
direction.
"""

_COMMON_PARAMETERS = """\
a, b, c : float
    You must select exactly one of these to specify which of the three lattice
    vectors you want to remove to form a plane. The assigned value represents
    the fractional length along this lattice vector, so `a = 0.3` will remove
    the first lattice vector and then take the grid points at 30% of the length
    of the first vector in the b-c plane. The fractional height uses periodic
    boundary conditions.

supercell : int or np.ndarray
    Replicate the contour plot periodically a given number of times. If you
    provide two different numbers, the resulting cell will be the two remaining
    lattice vectors multiplied by the specific number.

normal : str or None
    If not set, py4vasp will align the first remaining lattice vector with the
    x-axis and the second one such that the angle between the lattice vectors
    is preserved. You can set it to "x", "y", or "z"; then py4vasp will rotate
    the plane in such a way that the normal direction aligns with the specified
    Cartesian axis. This may look better if the normal direction is close to a
    Cartesian axis. You may also set it to "auto" so that py4vasp chooses a
    close Cartesian axis if it can find any.
"""


def _join_with_emphasis(data):
    emph_data = [f"*{x}*" for x in filter(lambda key: key != _INTERNAL, data)]
    if len(data) < 3:
        return " and ".join(emph_data)
    emph_data.insert(-1, "and")
    return ", ".join(emph_data)


class Density(_base.Refinery, _structure.Mixin, view.Mixin):
    """This class accesses various densities (charge, magnetization, ...) of VASP.

    The charge density is one key quantity optimized by VASP. With this class you
    can extract the final density and visualize it within the structure of the
    system. For collinear calculations, one can also consider the magnetization
    density. For noncollinear calculations, the magnetization density has three
    components. One may also be interested in the kinetic energy density for
    metaGGA calculations.
    """

    @_base.data_access
    def __str__(self):
        _raise_error_if_no_data(self._raw_data.charge)
        grid = self._raw_data.charge.shape[1:]
        topology = calculation.topology.from_data(self._raw_data.structure.topology)
        if self._selection == "kinetic_energy":
            name = "Kinetic energy"
        elif self.is_nonpolarized():
            name = "Nonpolarized"
        elif self.is_collinear():
            name = "Collinear"
        else:
            name = "Noncollinear"
        return f"""{name} density:
    structure: {pretty.pretty(topology)}
    grid: {grid[2]}, {grid[1]}, {grid[0]}"""

    @documentation.format(
        component0=_join_with_emphasis(_COMPONENTS[0]),
        component1=_join_with_emphasis(_COMPONENTS[1]),
        component2=_join_with_emphasis(_COMPONENTS[2]),
        component3=_join_with_emphasis(_COMPONENTS[3]),
    )
    @_base.data_access
    def selections(self):
        """Returns possible densities VASP can produce along with all available components.

        In the dictionary, the key *density* lists all different densities you can access
        from the VASP output provided you set the relevant INCAR tags. You can combine
        any of these with any possible choice from the key *component* to further
        specify the particular output you will receive. If you do not specify a *density*
        or a *component* the other routines will default to the electronic charge and
        the 0-th component.

        To nest density and component, please use parentheses, e.g. ``charge(1, 2)`` or
        ``3(kinetic_energy)``.

        For convenience, py4vasp accepts the following aliases

        electronic charge density
            *charge*, *n*, *charge_density*, and *electronic_charge_density*

        kinetic energy density
            *kinetic_energy*, *kinetic_energy*, and *kinetic_energy_density*

        0th component
            {component0}

        1st component
            {component1}

        2nd component
            {component2}

        3rd component
            {component3}

        Returns
        -------
        dict
            Possible densities and components to pass as selection in other functions on density.

        Notes
        -----
        In the special case of collinear calculations, *magnetization*, *mag*, and *m*
        are another alias for the 3rd component of the charge density.

        Examples
        --------
        >>> calc = py4vasp.Calculation.from_path(".")
        >>> calc.density.to_dict("n")
        >>> calc.density.plot("magnetization")
        Using synonyms and nesting
        >>> calc.density.plot("n m(1,2) mag(sigma_z)")
        """
        sources = super().selections()
        if self._raw_data.charge.is_none():
            return sources
        if self.is_nonpolarized():
            components = [_COMPONENTS[0][_DEFAULT]]
        elif self.is_collinear():
            components = [_COMPONENTS[0][_DEFAULT], _COMPONENTS[3][_DEFAULT]]
        else:
            components = [_COMPONENTS[i][_DEFAULT] for i in range(4)]
        return {**sources, "component": components}

    @_base.data_access
    def to_dict(self):
        """Read the density into a dictionary.

        Parameters
        ----------
        selection : str
            VASP computes different densities depending on the INCAR settings. With this
            parameter, you can control which one of them is returned. Please use the
            `selections` routine to get a list of all possible choices.

        Returns
        -------
        dict
            Contains the structure information as well as the density represented
            on a grid in the unit cell.
        """
        _raise_error_if_no_data(self._raw_data.charge)
        result = {"structure": self._structure.read()}
        result.update(self._read_density())
        return result

    def _read_density(self):
        density = self.to_numpy()
        if self._selection:
            yield self._selection, density
        else:
            yield "charge", density[0]
            if self.is_collinear():
                yield "magnetization", density[1]
            elif self.is_noncollinear():
                yield "magnetization", density[1:]

    @_base.data_access
    def to_numpy(self):
        """Convert the density to a numpy array.

        The number of components is 1 for nonpolarized calculations, 2 for collinear
        calculations, and 4 for noncollinear calculations. Each component is 3
        dimensional according to the grid VASP uses for the FFTs.

        Returns
        -------
        np.ndarray
            All components of the selected density.
        """
        return np.moveaxis(self._raw_data.charge, 0, -1).T

    @_base.data_access
    def to_view(self, selection=None, supercell=None, **user_options):
        """Plot the selected density as a 3d isosurface within the structure.

        Parameters
        ----------
        selection : str
            Can be either *charge* or *magnetization*, depending on which quantity
            should be visualized.  For a noncollinear calculation, the density has
            4 components which can be represented in a 2x2 matrix. Specify the
            component of the density in terms of the Pauli matrices: sigma_1,
            sigma_2, sigma_3.

        supercell : int or np.ndarray
            If present the data is replicated the specified number of times along each
            direction.

        user_options
            Further arguments with keyword that get directly passed on to the
            visualizer. Most importantly, you can set isolevel to adjust the
            value at which the isosurface is drawn.

        Returns
        -------
        View
            Visualize an isosurface of the density within the 3d structure.

        Examples
        --------
        >>> calc = py4vasp.Calculation.from_path(".")
        Plot an isosurface of the electronic charge density
        >>> calc.density.plot()
        Plot isosurfaces for positive (blue) and negative (red) magnetization
        of a spin-polarized calculation (ISPIN=2)
        >>> calc.density.plot("m")
        Plot the isosurface for the third component of a noncollinear magnetization
        >>> calc.density.plot("m(3)")
        """
        _raise_error_if_no_data(self._raw_data.charge)
        selection = selection or _INTERNAL
        viewer = self._structure.plot(supercell)
        map_ = self._create_map()
        selector = index.Selector({0: map_}, self._raw_data.charge)
        tree = select.Tree.from_selection(selection)
        selections = self._filter_noncollinear_magnetization_from_selections(tree)
        viewer.grid_scalars = [
            self._grid_quantity(selector, selection, map_, user_options)
            for selection in selections
        ]
        return viewer

    def _filter_noncollinear_magnetization_from_selections(self, tree):
        if self._selection or not self.is_noncollinear():
            yield from tree.selections()
        else:
            filtered_selections = tree.selections(filter=set(_MAGNETIZATION))
            for filtered, unfiltered in zip(filtered_selections, tree.selections()):
                if filtered != unfiltered and len(filtered) != 1:
                    _raise_component_not_specified_error(unfiltered)
                yield filtered

    def _create_map(self):
        map_ = {
            choice: self._index_component(component)
            for component, choices in _COMPONENTS.items()
            for choice in choices
        }
        self._add_magnetization_for_charge_and_collinear(map_)
        return map_

    def _index_component(self, component):
        if self.is_collinear():
            component = (0, 2, 3, 1)[component]
        return component

    def _add_magnetization_for_charge_and_collinear(self, map_):
        if self._selection or not self.is_collinear():
            return
        for key in _MAGNETIZATION:
            map_[key] = 1

    def _grid_quantity(self, selector, selection, map_, user_options):
        component_label = selector.label(selection)
        component = map_.get(component_label, -1)
        return view.GridQuantity(
            quantity=(selector[selection].T)[np.newaxis],
            label=self._label(component_label),
            isosurfaces=self._isosurfaces(component, **user_options),
        )

    def _label(self, component_label):
        if component_label == _INTERNAL:
            return self._selection or "charge"
        elif self._selection:
            return f"{self._selection}({component_label})"
        else:
            return component_label

    def _isosurfaces(self, component, isolevel=0.2, color=None, opacity=0.6):
        if self._use_symmetric_isosurface(component):
            _raise_error_if_color_is_specified(color)
            return [
                view.Isosurface(isolevel, _config.VASP_COLORS["blue"], opacity),
                view.Isosurface(-isolevel, _config.VASP_COLORS["red"], opacity),
            ]
        else:
            color = color or _config.VASP_COLORS["cyan"]
            return [view.Isosurface(isolevel, color, opacity)]

    def _use_symmetric_isosurface(self, component):
        if component > 0 and self.is_nonpolarized():
            _raise_is_nonpolarized_error()
        if component > 1 and self.is_collinear():
            _raise_is_collinear_error()
        return component > 0

    @_base.data_access
    @documentation.format(plane=_PLANE, common_parameters=_COMMON_PARAMETERS)
    def to_contour(
        self, selection=None, *, a=None, b=None, c=None, supercell=None, normal=None
    ):
        """Generate a contour plot of the selected component of the density.

        {plane}

        Parameters
        ----------
        selection : str
            Select which component of the density you want to visualize. Please use the
            `selections` method to get all available choices.

        {common_parameters}

        Returns
        -------
        graph
            A contour plot in the plane spanned by the 2 remaining lattice vectors.


        Examples
        --------

        Cut a plane through the magnetization density at the origin of the third lattice
        vector.

        >>> calc.density.to_contour("3", c=0)

        Replicate a plane in the middle of the second lattice vector 2 times in each
        direction.

        >>> calc.density.to_contour(b=0.5, supercell=2)

        Take a slice of the kinetic energy density along the first lattice vector and
        rotate it such that the normal of the plane aligns with the x axis.

        >>> calc.density.to_contour("kinetic_energy", a=0.3, normal="x")
        """
        cut, fraction = self._get_cut(a, b, c)
        plane = slicing.plane(self._structure.lattice_vectors(), cut, normal)
        map_ = self._create_map()
        selector = index.Selector({0: map_}, self._raw_data.charge)
        tree = select.Tree.from_selection(selection)
        selections = self._filter_noncollinear_magnetization_from_selections(tree)
        contours = [
            self._contour(selector, selection, plane, fraction, supercell)
            for selection in selections
        ]
        return graph.Graph(contours)

    def _contour(self, selector, selection, plane, fraction, supercell):
        density = selector[selection].T
        data = slicing.grid_scalar(density, plane, fraction)
        label = self._label(selector.label(selection)) or "charge"
        contour = graph.Contour(data, plane, label, isolevels=True)
        if supercell is not None:
            contour.supercell = np.ones(2, dtype=np.int_) * supercell
        return contour

    @_base.data_access
    @documentation.format(plane=_PLANE, common_parameters=_COMMON_PARAMETERS)
    def to_quiver(self, *, a=None, b=None, c=None, supercell=None, normal=None):
        """Generate a quiver plot of magnetization density.

        {plane}

        For a collinear calculation, the magnetization density will be aligned with the
        y axis of the plane. For noncollinear calculations, the magnetization density
        is projected into the plane.

        Parameters
        ----------
        {common_parameters}

        Returns
        -------
        graph
            A quiver plot in the plane spanned by the 2 remaining lattice vectors.


        Examples
        --------

        Cut a plane at the origin of the third lattice vector.

        >>> calc.density.to_quiver(c=0)

        Replicate a plane in the middle of the second lattice vector 2 times in each
        direction.

        >>> calc.density.to_quiver(b=0.5, supercell=2)

        Take a slice of the spin components of the kinetic energy density along the
        first lattice vector and rotate it such that the normal of the plane aligns with
        the x axis.

        >>> calc.density.to_quiver("kinetic_energy", a=0.3, normal="x")
        """
        cut, fraction = self._get_cut(a, b, c)
        plane = slicing.plane(self._structure.lattice_vectors(), cut, normal)
        if self.is_collinear():
            data = slicing.grid_scalar(self._raw_data.charge[1].T, plane, fraction)
            data = np.array((np.zeros_like(data), data))
        else:
            data = slicing.grid_vector(self.to_numpy()[1:], plane, fraction)
        label = self._selection or "magnetization"
        quiver_plot = graph.Contour(5.0 * data, plane, label)
        if supercell is not None:
            quiver_plot.supercell = np.ones(2, dtype=np.int_) * supercell
        return graph.Graph([quiver_plot])

    def _get_cut(self, a, b, c):
        _raise_error_cut_selection_incorrect(a, b, c)
        if a is not None:
            return "a", a
        if b is not None:
            return "b", b
        return "c", c

    @_base.data_access
    def is_nonpolarized(self):
        "Returns whether the density is not spin polarized."
        return len(self._raw_data.charge) == 1

    @_base.data_access
    def is_collinear(self):
        "Returns whether the density has a collinear magnetization."
        return len(self._raw_data.charge) == 2

    @_base.data_access
    def is_noncollinear(self):
        "Returns whether the density has a noncollinear magnetization."
        return len(self._raw_data.charge) == 4

    @property
    def _selection(self):
        selection_map = {
            "kinetic_energy": "kinetic_energy",
            "kinetic_energy_density": "kinetic_energy",
        }
        return selection_map.get(super()._selection)


def _raise_error_if_color_is_specified(color):
    if color is not None:
        msg = "Specifying the color of a magnetic isosurface is not implemented."
        raise exception.NotImplemented(msg)


def _raise_component_not_specified_error(selec_tuple):
    msg = (
        "Invalid selection: selection='"
        + ", ".join(selec_tuple)
        + "'. For a noncollinear calculation, the density has 4 components which can be represented in a 2x2 matrix. Specify the component of the density in terms of the Pauli matrices: sigma_1, sigma_2, sigma_3. E.g.: m(sigma_1)."
    )
    raise exception.IncorrectUsage(msg)


def _raise_is_nonpolarized_error():
    msg = "Density does not contain magnetization. Please rerun VASP with ISPIN = 2 or LNONCOLLINEAR = T to obtain it."
    raise exception.NoData(msg)


def _raise_is_collinear_error():
    msg = "Density does not contain noncollinear magnetization. Please rerun VASP with LNONCOLLINEAR = T to obtain it."
    raise exception.NoData(msg)


def _raise_error_if_no_data(data):
    if data.is_none():
        raise exception.NoData(
            "Density data was not found. Note that the density information is written "
            "on the demand to a different file (vaspwave.h5). Please make sure that "
            "this file exists and LCHARGH5 = T is set in the INCAR file. Another "
            'common issue is when you create `Calculation.from_file("vaspout.h5")` '
            "because this will overwrite the default file behavior."
        )


def _raise_error_cut_selection_incorrect(*selections):
    # only a single element may be selected
    selected_elements = sum(selection is not None for selection in selections)
    if selected_elements == 0:
        raise exception.IncorrectUsage(
            "You have not selected a lattice vector along which the slice should be "
            "constructed. Please set exactly one of the keyword arguments (a, b, c) "
            "to a real number that specifies at which fraction of the lattice vector "
            "the plane is."
        )
    if selected_elements > 1:
        raise exception.IncorrectUsage(
            "You have selected more than a single element. Please use only one of "
            "(a, b, c) and not multiple choices."
        )
