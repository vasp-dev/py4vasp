# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import itertools
import numpy as np
import pandas as pd
import py4vasp._util.sanity_check as _check
import py4vasp._util.documentation as _documentation
from IPython.lib.pretty import pretty
from .projector import _projectors_or_dummy, _selection_doc, _selection_examples
from .kpoint import Kpoint, _kpoints_opt_source
from py4vasp.data._base import DataBase, RefinementDescriptor
import py4vasp.data._export as _export
import py4vasp._third_party.graph as _graph


class Band(DataBase, _export.Image):
    """The electronic band structure.

    The most common use case of this class is to produce the electronic band
    structure along a path in the Brillouin zone used in a non self consistent
    Vasp calculation. In some cases you may want to use the `to_dict` function
    just to obtain the eigenvalue and projection data though in that case the
    **k**-point distances that are calculated are meaningless.

    Parameters
    ----------
    raw_band : RawBand
        Dataclass containing the raw data necessary to produce a band structure
        (eigenvalues, kpoints, ...).
    """

    read = RefinementDescriptor("_to_dict")
    to_dict = RefinementDescriptor("_to_dict")
    plot = RefinementDescriptor("_plot")
    to_plotly = RefinementDescriptor("_to_plotly")
    to_frame = RefinementDescriptor("_to_frame")
    __str__ = RefinementDescriptor("_to_string")

    def _to_string(self):
        return f"""
{"spin polarized" if self._spin_polarized() else ""} band data:
    {self._raw_data.eigenvalues.shape[1]} k-points
    {self._raw_data.eigenvalues.shape[2]} bands
{pretty(_projectors_or_dummy(self._raw_data.projectors))}
    """.strip()

    @_documentation.add(
        f"""Read the data into a dictionary.

Parameters
----------
{_selection_doc}
{_kpoints_opt_source}

Returns
-------
dict
    Contains the **k**-point path for plotting band structures with the
    eigenvalues shifted to bring the Fermi energy to 0. If available
    and a selection is passed, the projections of these bands on the
    selected projectors are included.

{_selection_examples("band", "read")}"""
    )
    def _to_dict(self, selection=None):
        kpoints = self._kpoints()
        return {
            "kpoint_distances": kpoints.distances(),
            "kpoint_labels": kpoints.labels(),
            "fermi_energy": self._raw_data.fermi_energy,
            **self._shift_bands_by_fermi_energy(),
            **self._read_occupations(),
            "projections": self._read_projections(selection),
        }

    @_documentation.add(
        f"""Read the data and generate a plotly figure.

Parameters
----------
{_selection_doc}
width : float
Specifies the width of the flatbands if a selection of projections is specified.
{_kpoints_opt_source}

Returns
-------
plotly.graph_objects.Figure
plotly figure containing the spin-up and spin-down bands. If a selection
is provided the width of the bands represents the projections of the
bands onto the specified projectors.

{_selection_examples("band", "plot")}"""
    )
    def _plot(self, selection=None, width=0.5):
        return _graph.Graph(
            series=self._band_structure(selection, width),
            xticks=self._xticks(),
            ylabel="Energy (eV)",
        )
        # data = self._band_structure(selection, width)
        # default = {
        #     "xaxis": {"tickmode": "array", "tickvals": ticks, "ticktext": labels},
        #     "yaxis": {"title": {"text": "Energy (eV)"}},
        # }
        # return go.Figure(data=data, layout=default)

    @_documentation.add(
        f"""Read the data and generate a plotly figure.

Parameters
----------
{_selection_doc}
width : float
    Specifies the width of the flatbands if a selection of projections is specified.
{_kpoints_opt_source}

Returns
-------
plotly.graph_objects.Figure
    plotly figure containing the spin-up and spin-down bands. If a selection
    is provided the width of the bands represents the projections of the
    bands onto the specified projectors.

{_selection_examples("band", "to_plotly")}"""
    )
    def _to_plotly(self, selection=None, width=0.5):
        return self._plot(selection, width).to_plotly()

    @_documentation.add(
        f"""Read the data into a DataFrame.

Parameters
----------
{_selection_doc}
{_kpoints_opt_source}

Returns
-------
pd.DataFrame
    Contains the eigenvalues and corresponding occupations for all k-points and
    bands. If a selection string is given, in addition the orbital projections
    on these bands are returned.

{_selection_examples("band", "to_frame")}"""
    )
    def _to_frame(self, selection=None):
        index = self._setup_dataframe_index()
        data = self._extract_relevant_data(selection)
        return pd.DataFrame(data, index)

    def _spin_polarized(self):
        return len(self._raw_data.eigenvalues) == 2

    def _kpoints(self):
        return Kpoint.from_data(self._raw_data.kpoints)

    def _read_projections(self, selection):
        projectors = _projectors_or_dummy(self._raw_data.projectors)
        return projectors.read(selection, self._raw_data.projections)

    def _shift_bands_by_fermi_energy(self):
        if self._spin_polarized():
            return {
                "bands_up": self._shift_particular_spin_by_fermi_energy(0),
                "bands_down": self._shift_particular_spin_by_fermi_energy(1),
            }
        else:
            return {"bands": self._shift_particular_spin_by_fermi_energy(0)}

    def _shift_particular_spin_by_fermi_energy(self, spin):
        return self._raw_data.eigenvalues[spin] - self._raw_data.fermi_energy

    def _read_occupations(self):
        if self._spin_polarized():
            return {
                "occupations_up": self._raw_data.occupations[0],
                "occupations_down": self._raw_data.occupations[1],
            }
        else:
            return {"occupations": self._raw_data.occupations[0]}

    def _band_structure(self, selection, width):
        bands = self._shift_bands_by_fermi_energy()
        projections = self._read_projections(selection)
        if len(projections) == 0:
            return self._regular_band_structure(bands)
        else:
            return self._fat_band_structure(bands, projections, width)

    def _regular_band_structure(self, bands):
        kdists = self._kpoints().distances()
        return [_graph.Series(kdists, lines.T, name) for name, lines in bands.items()]

    def _fat_band_structure(self, bands, projections, width):
        error_message = "Width of fat band structure must be a number."
        _check.raise_error_if_not_number(width, error_message)
        data = (
            self._fat_band(args, width)
            for args in itertools.product(bands.items(), projections.items())
        )
        return list(filter(None, data))

    def _fat_band(self, args, width):
        (key, lines), (name, projection) = args
        key = key.lstrip("bands_")
        if self._spin_polarized() and not key in name:
            return None
        kdists = self._kpoints().distances()
        return _graph.Series(kdists, lines.T, name, width=width * projection)

    def _xticks(self):
        ticks, labels = self._degenerate_ticks_and_labels()
        return self._filter_unique(ticks, labels)

    def _degenerate_ticks_and_labels(self):
        labels = self._kpoint_labels()
        mask = np.logical_or(self._edge_of_line(), labels != "")
        return self._kpoints().distances()[mask], labels[mask]

    def _filter_unique(self, ticks, labels):
        result = {}
        for tick, label in zip(ticks, labels):
            if tick in result:
                previous_label = result[tick]
                if previous_label != "" and previous_label != label:
                    label = previous_label + "|" + label
            result[tick] = label
        return result

    def _kpoint_labels(self):
        labels = self._kpoints().labels()
        if labels is None:
            labels = [""] * len(self._raw_data.kpoints.coordinates)
        return np.array(labels)

    def _edge_of_line(self):
        indices = np.arange(len(self._raw_data.kpoints.coordinates))
        edge_of_line = (indices + 1) % self._kpoints().line_length() == 0
        edge_of_line[0] = True
        return edge_of_line

    def _setup_dataframe_index(self):
        return [
            _index_string(kpoint, band)
            for kpoint in self._raw_data.kpoints.coordinates
            for band in range(self._raw_data.eigenvalues.shape[2])
        ]

    def _extract_relevant_data(self, selection):
        relevant_keys = (
            "bands",
            "bands_up",
            "bands_down",
            "occupations",
            "occupations_up",
            "occupations_down",
        )
        data = {}
        for key, value in self._to_dict().items():
            if key in relevant_keys:
                data[key] = _to_series(value)
        for key, value in self._read_projections(selection).items():
            data[key] = _to_series(value)
        return data


def _index_string(kpoint, band):
    if band == 0:
        return np.array2string(kpoint, formatter={"float": lambda x: f"{x:.2f}"}) + " 1"
    else:
        return str(band + 1)


def _to_series(array):
    return array.T.flatten()
