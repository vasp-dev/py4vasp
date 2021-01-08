import functools
import itertools
import numpy as np
import plotly.graph_objects as go
from IPython.lib.pretty import pretty
from .projectors import _projectors_or_dummy, _selection_doc
from .kpoints import Kpoints
from py4vasp.data import _util

_to_dict_doc = (
    """ Read the data into a dictionary.

Parameters
----------
{}

Returns
-------
dict
    Contains the **k**-point path for plotting band structures with the
    eigenvalues shifted to bring the Fermi energy to 0. If available
    and a selection is passed, the projections of these bands on the
    selected projectors are included.
"""
).format(_selection_doc)

_to_plotly_doc = (
    """ Read the data and generate a plotly figure.

Parameters
----------
{}
width : float
    Specifies the width of the flatbands if a selection of projections is specified.
Returns
-------
plotly.graph_objects.Figure
    plotly figure containing the spin-up and spin-down bands. If a selection
    is provided the width of the bands represents the projections of the
    bands onto the specified projectors.
"""
).format(_selection_doc)


@_util.add_wrappers
class Band(_util.Data):
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

    def __init__(self, raw_band):
        super().__init__(raw_band)
        self._kpoints = Kpoints(raw_band.kpoints)
        self._spin_polarized = len(raw_band.eigenvalues) == 2
        self._projectors = _projectors_or_dummy(raw_band.projectors)

    def _repr_pretty_(self, p, cycle):
        path = self._create_path_if_available()
        text = f"""
{"spin polarized" if self._spin_polarized else ""} band structure{path}:
   {self._raw.eigenvalues.shape[1]} k-points
   {self._raw.eigenvalues.shape[2]} bands
{pretty(self._projectors)}
        """.strip()
        p.text(text)

    @classmethod
    @_util.add_doc(_util.from_file_doc("electronic band structure"))
    def from_file(cls, file=None):
        return _util.from_file(cls, file, "band")

    @_util.add_doc(_to_dict_doc)
    def to_dict(self, selection=None):
        return {
            "kpoint_distances": self._kpoints.distances(),
            "kpoint_labels": self._kpoints.labels(),
            "fermi_energy": self._raw.fermi_energy,
            **self._shift_bands_by_fermi_energy(),
            "projections": self._projectors.read(selection, self._raw.projections),
        }

    @_util.add_doc(_to_plotly_doc)
    def to_plotly(self, selection=None, width=0.5):
        ticks, labels = self._ticks_and_labels()
        data = self._band_structure(selection, width)
        default = {
            "xaxis": {"tickmode": "array", "tickvals": ticks, "ticktext": labels},
            "yaxis": {"title": {"text": "Energy (eV)"}},
        }
        return go.Figure(data=data, layout=default)

    def _shift_bands_by_fermi_energy(self):
        if self._spin_polarized:
            return {
                "up": self._raw.eigenvalues[0] - self._raw.fermi_energy,
                "down": self._raw.eigenvalues[1] - self._raw.fermi_energy,
            }
        else:
            return {"bands": self._raw.eigenvalues[0] - self._raw.fermi_energy}

    def _band_structure(self, selection, width):
        bands = self._shift_bands_by_fermi_energy()
        projections = self._projectors.read(selection, self._raw.projections)
        if len(projections) == 0:
            return self._regular_band_structure(bands)
        else:
            return self._fat_band_structure(bands, projections, width)

    def _regular_band_structure(self, bands):
        kdists = self._kpoints.distances()
        return [self._scatter(name, kdists, lines) for name, lines in bands.items()]

    def _fat_band_structure(self, bands, projections, width):
        error_message = "Width of fat band structure must be a number."
        _util.raise_error_if_not_number(width, error_message)
        data = (
            self._fat_band(args, width)
            for args in itertools.product(bands.items(), projections.items())
        )
        return list(filter(None, data))

    def _fat_band(self, args, width):
        (key, lines), (name, projection) = args
        if self._spin_polarized and not key in name:
            return None
        kdists = self._kpoints.distances()
        fatband_kdists = np.concatenate((kdists, np.flip(kdists)))
        upper = lines + width * projection
        lower = lines - width * projection
        fatband_lines = np.concatenate((lower, np.flip(upper, axis=0)), axis=0)
        plot = self._scatter(name, fatband_kdists, fatband_lines)
        plot.fill = "toself"
        plot.mode = "none"
        return plot

    def _scatter(self, name, kdists, lines):
        # insert NaN to split separate lines
        num_bands = lines.shape[-1]
        kdists = np.tile([*kdists, np.NaN], num_bands)
        lines = np.append(lines, [np.repeat(np.NaN, num_bands)], axis=0)
        return go.Scatter(x=kdists, y=lines.flatten(order="F"), name=name)

    def _create_path_if_available(self):
        _, labels = self._ticks_and_labels()
        if any(len(label.strip()) > 0 for label in labels):
            return " (" + " - ".join(labels) + ")"
        else:
            return ""

    def _ticks_and_labels(self):
        def filter_unique(current, item):
            tick, label = item
            previous_tick = current[-2]
            if tick == previous_tick:
                previous_label = current[-1]
                label = previous_label + "|" + label if previous_label else label
                return current[:-1] + (label,)
            else:
                return current + item

        ticks_and_labels = self._degenerate_ticks_and_labels()
        ticks_and_labels = functools.reduce(filter_unique, ticks_and_labels)
        return self._split_and_replace_empty_labels(ticks_and_labels)

    def _split_and_replace_empty_labels(self, ticks_and_labels):
        ticks = [tick for tick in ticks_and_labels[::2]]
        labels = [label or " " for label in ticks_and_labels[1::2]]
        # plotly replaces empty labels with tick position, so we replace them
        return ticks, labels

    def _degenerate_ticks_and_labels(self):
        labels = self._kpoint_labels()
        mask = np.logical_or(self._edge_of_line(), labels != "")
        return zip(self._kpoints.distances()[mask], labels[mask])

    def _kpoint_labels(self):
        labels = self._kpoints.labels()
        if labels is None:
            labels = [""] * len(self._raw.kpoints.coordinates)
        return np.array(labels)

    def _edge_of_line(self):
        indices = np.arange(len(self._raw.kpoints.coordinates))
        edge_of_line = (indices + 1) % self._kpoints.line_length() == 0
        edge_of_line[0] = True
        return edge_of_line
