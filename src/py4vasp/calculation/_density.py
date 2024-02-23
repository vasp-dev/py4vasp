# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import numpy as np

from py4vasp import _config, calculation, exception
from py4vasp._third_party import view
from py4vasp._util import documentation, import_, index, select
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
        if self._selection == "tau":
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
        ``3(tau)``.

        For convenience, py4vasp accepts the following aliases

        electronic charge density
            *charge*, *n*, *charge_density*, and *electronic_charge_density*

        kinetic energy density
            *tau*, *kinetic_energy*, and *kinetic_energy_density*

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
                view.Isosurface(isolevel, _config.VASP_BLUE, opacity),
                view.Isosurface(-isolevel, _config.VASP_RED, opacity),
            ]
        else:
            return [view.Isosurface(isolevel, color or _config.VASP_CYAN, opacity)]

    def _use_symmetric_isosurface(self, component):
        if component > 0 and self.is_nonpolarized():
            _raise_is_nonpolarized_error()
        if component > 1 and self.is_collinear():
            _raise_is_collinear_error()
        return component > 0

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
            "tau": "tau",
            "kinetic_energy": "tau",
            "kinetic_energy_density": "tau",
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
