# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import numpy as np

from py4vasp import data, exception
from py4vasp._data import base, structure
from py4vasp._util import documentation, import_, select

pretty = import_.optional("IPython.lib.pretty")


class _ViewerWrapper:
    def __init__(self, viewer):
        self._viewer = viewer
        self._options = {"isolevel": 0.2, "color": "yellow", "opacity": 0.6}

    def show_isosurface(self, data, **options):
        options = {**self._options, **options}
        self._viewer.show_isosurface(data, **options)

_SELECTIONS = {
        "quantity": {
            "electronic charge density": ["electronic_charge_density", "charge_density", "charge", "n"],
            "magnetization": ["magnetization", "mag", "m"],
            "kinetic energy density": ["kinetic_energy_density", "tau"],
            "current density": ["paramagnetic_current_density", "current_density", "current", "j"],
            },
        "component": {
            0: ["unity", "sigma_0", "scalar", "0"],
            3: ["sigma_z", "z", "sigma_3", "3"],
            1: ["sigma_x", "x", "sigma_1", "1"],
            2: ["sigma_y", "y", "sigma_2", "2"]
            }
}

selections_doc="""\
        Returns
        -------
        Possible quantities and components to choose for the selection of functions on Density.

        Note: Currently only the electronic charge density and (if available) the magnetization are implemented. For plotting the magnetization of a noncollinear calculation (LNONCOLLINEAR=True), the component of the magnetization must also be selected in terms of Pauli matrices: sigma_1, sigma_2, sigma_3.

        Synonyms to access the ...
        ... electronic charge density: *{}*, *{}*, *{}*, and *{}*
        ... magnetization: *{}*, *{}*, and *{}*
        ... 1st component: *{}*, *{}*, *{}*, and *{}*
        ... 2nd component: *{}*, *{}*, *{}*, and *{}*
        ... 3rd component: *{}*, *{}*, *{}*, and *{}*

        Nesting: '3(m,tau) m(1,2)'

        Examples
        --------
        >>> calc = py4vasp.Calculation.from_path(".")
        >>> calc.density.to_dict("n")
        >>> calc.density.plot("magnetization")
        Using synonyms and nesting
        >>> calc.density.plot("n m(1,2) mag(sigma_z)")
        """.format(*_SELECTIONS["quantity"]["electronic charge density"],
                *_SELECTIONS["quantity"]["magnetization"],
                *_SELECTIONS["component"][1],
                *_SELECTIONS["component"][2],
                *_SELECTIONS["component"][3])


class Density(base.Refinery, structure.Mixin):
    """The charge and magnetization density.

    You can use this class to extract the density data of the VASP calculation
    and to have a quick glance at the resulting density.
    """

    @base.data_access
    def __str__(self):
        _raise_error_if_no_data(self._raw_data.charge, "electronic charge density")
        grid = self._raw_data.charge.shape[1:]
        topology = data.Topology.from_data(self._raw_data.structure.topology)
        if self.is_nonpolarized():
            name = "Nonpolarized"
        elif self.is_collinear():
            name = "Collinear"
        else:
            name = "Noncollinear"
        return f"""{name} density:
    structure: {pretty.pretty(topology)}
    grid: {grid[2]}, {grid[1]}, {grid[0]}"""

    @documentation.format(selections_doc=selections_doc)
    @base.data_access
    def selections(self):
        """{selections_doc}"""
        if not self._raw_data.charge.is_none():
            if self.is_nonpolarized():
                return [_SELECTIONS["quantity"]["electronic charge density"][-1]]
            elif self.is_collinear():
                quantities = ["electronic charge density", "magnetization"]
                return [_SELECTIONS["quantity"][q][-1] for q in quantities]
            elif self.is_noncollinear():
                return [_SELECTIONS["quantity"]["electronic charge density"][-1],
                        _SELECTIONS["quantity"]["magnetization"][-1]+"(1,2,3)"]
        else:
            return None

    @base.data_access
    def to_dict(self, selection="charge"):
        """Read the density into a dictionary.

        Parameters
        ----------
        selection : str
            *charge* (default): Currently only the electronic charge density and (if available) the magnetization are implemented. Both via the keyword *charge*.

        Returns
        -------
        dict
            Contains the structure information as well as the density represented
            on a grid in the unit cell.
        """
        quantities_set = set(self._parse_quantity(selection))
        if "electronic charge density" in quantities_set:
            _raise_error_if_no_data(self._raw_data.charge, "electronic charge density")
        if "kinetic energy density" in quantities_set:
            _raise_quantity_not_implemented_error("Dictonary", "kinetic energy density")
        if "current density" in quantities_set:
            _raise_quantity_not_implemented_error("Dictonary", "current density")
        result = {"structure": self._structure.read()}
        for quantity in quantities_set:
            result.update(self._read_density(quantity))
        return result

    def _read_density(self, quantity):
        if quantity=="electronic charge density":
            density = np.moveaxis(self._raw_data.charge, 0, -1).T
            yield "charge", density[0]
            if self.is_collinear():
                yield "magnetization", density[1]
            elif self.is_noncollinear():
                yield "magnetization", density[1:]

    @base.data_access
    def plot(self, selection="charge", **user_options):
        """Plot the selected density as a 3d isosurface within the structure.

        Parameters
        ----------
        selection : str
            Can be either *charge* or *magnetization*, depending on which quantity
            should be visualized.  For a noncollinear calculation, the density has
            4 components which can be represented in a 2x2 matrix. Specify the
            component of the density in terms of the Pauli matrices: sigma_1,
            sigma_2, sigma_3.

        user_options
            Further arguments with keyword that get directly passed on to the
            visualizer. Most importantly, you can set isolevel to adjust the
            value at which the isosurface is drawn.

        Returns
        -------
        Viewer3d
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
        viewer = self._structure.plot()
        # _raise_error_if_no_data(self._raw_data.charge, quantity)
        for quantity, component in self._parse_selection(selection):
            self._add_isosurface(_ViewerWrapper(viewer), quantity, component, **user_options)
        return viewer

    def _add_isosurface(self, viewer, quantity, component, **user_options):
        density_data = self._get_density(quantity, component)
        if component>0:
            _raise_error_if_color_is_specified(**user_options)
            viewer.show_isosurface(density_data, color="blue", **user_options)
            viewer.show_isosurface(-density_data, color="red", **user_options)
        else:
            viewer.show_isosurface(density_data, color="yellow", **user_options)

    def _get_density(self, quantity, component):
        if quantity=="electronic charge density" or quantity=="magnetization":
            density_data = self._raw_data.charge[component]
        else:
            _raise_quantity_not_implemented_error("Plotting", quantity)
        return density_data

    @base.data_access
    def is_nonpolarized(self):
        "Returns whether the density is not spin polarized."
        return len(self._raw_data.charge) == 1

    @base.data_access
    def is_collinear(self):
        "Returns whether the density has a collinear magnetization."
        return len(self._raw_data.charge) == 2

    @base.data_access
    def is_noncollinear(self):
        "Returns whether the density has a noncollinear magnetization."
        return len(self._raw_data.charge) == 4

    def _parse_selection(self, selection):
        tree = select.Tree.from_selection(selection)
        translated_selections = []
        for selec_tuple in tree.selections():
            translated_selections.append(self._translate_selection(selec_tuple))
        return translated_selections

    def _translate_selection(self, selec_tuple):
        selec_tuple_string = str(selec_tuple).replace("magnetization", "m")
        quantity,component = "", -1
        for q in _SELECTIONS["quantity"]:
            for choice in _SELECTIONS["quantity"][q]:
                if choice in selec_tuple_string:
                    quantity = q
        for c in _SELECTIONS["component"]:
            for choice in _SELECTIONS["component"][c]:
                if choice in selec_tuple_string:
                    component = c
        if quantity=="electronic charge density" and component<0:
            component = 0
        elif quantity=="electronic charge density" and component>0:
            quantity="magnetization"
        if quantity=="magnetization":
            if self.is_nonpolarized():
                _raise_is_nonpolarized_error()
            elif component==0:
                _raise_error_if_selection_invalid(selec_tuple)
            elif self.is_collinear() and component<0:
                component = 1
            elif self.is_collinear() and component>1:
                _raise_is_collinear_error()
            elif self.is_noncollinear() and component<0:
                _raise_component_not_specified_error(selec_tuple)
        if quantity!="" and component<0:
            component = 0
        elif quantity=="" and component<0:
            _raise_error_if_selection_invalid(selec_tuple)
        elif quantity=="" and component==0:
            quantity="electronic charge density"
        elif quantity=="" and component>0:
            quantity="magnetization"
        return (quantity,component)

    def _parse_quantity(self, selection):
        tree = select.Tree.from_selection(selection)
        translated_quantities = []
        for selec_tuple in tree.selections():
            translated_quantities.append(self._translate_quantity(selec_tuple))
        return translated_quantities

    def _translate_quantity(self, selec_tuple):
        selec_tuple_string = str(selec_tuple).replace("magnetization", "m")
        quantity = ""
        for q in _SELECTIONS["quantity"]:
            for choice in _SELECTIONS["quantity"][q]:
                if choice in selec_tuple_string:
                    quantity = q
        if quantity=="magnetization":
            quantity="electronic charge density"
        if quantity=="":
            _raise_error_if_selection_invalid(selec_tuple)
        return quantity

def _raise_quantity_not_implemented_error(function_noun, quantity):
    msg = function_noun + " of the " + quantity + " is not yet implemented."
    raise exception.NotImplemented(msg)

def _raise_component_not_specified_error(selec_tuple):
    msg = "Invalid selection: selection='" + ", ".join(selec_tuple) + "'. For a noncollinear calculation, the density has 4 components which can be represented in a 2x2 matrix. Specify the component of the density in terms of the Pauli matrices: sigma_1, sigma_2, sigma_3. E.g.: m(sigma_1)."
    raise exception.IncorrectUsage(msg)

def _raise_is_nonpolarized_error():
    msg = "Density does not contain magnetization. Please rerun VASP with ISPIN = 2 or LNONCOLLINEAR = T to obtain it."
    raise exception.NoData(msg)

def _raise_is_collinear_error():
    msg = "Density does not contain noncollinear magnetization. Please rerun VASP with LNONCOLLINEAR = T to obtain it."
    raise exception.NoData(msg)

def _raise_error_if_no_data(data, quantity):
    if data.is_none() and quantity=="electronic charge density":
        raise exception.NoData(
            "Density data was not found. Note that the density information is written "
            "on the demand to a different file (vaspwave.h5). Please make sure that "
            "this file exists and LCHARGH5 = T is set in the INCAR file. Another "
            'common issue is when you create `Calculation.from_file("vaspout.h5")` '
            "because this will overwrite the default file behavior."
        )

def _raise_error_if_selection_invalid(selec_tuple):
    msg = "Invalid selection: selection='" + ", ".join(selec_tuple) + "'"
    raise exception.IncorrectUsage(msg)

def _raise_error_if_color_is_specified(**user_options):
    if "color" in user_options:
        msg = "Specifying the color of a magnetic isosurface is not implemented."
        raise exception.NotImplemented(msg)
