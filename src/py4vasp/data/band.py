import functools
import itertools
import numpy as np
import plotly.graph_objects as go
from .projectors import _projectors_or_dummy
from .kpoints import Kpoints
from py4vasp.data import _util


class Band:
    def __init__(self, raw_band):
        self._raw = raw_band
        self._kpoints = Kpoints(raw_band.kpoints)
        self._spin_polarized = len(raw_band.eigenvalues) == 2
        self._projectors = _projectors_or_dummy(raw_band.projectors)

    @classmethod
    def from_file(cls, file=None):
        return _util.from_file(cls, file, "band")

    def read(self, selection=None):
        res = {
            "kpoint_distances": self._kpoints.distances(),
            "kpoint_labels": self._kpoints.labels(),
            "fermi_energy": self._raw.fermi_energy,
            **self._shift_bands_by_fermi_energy(),
            "projections": self._read_projections(selection),
        }
        return res

    def plot(self, selection=None, width=0.5):
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
        projections = self._read_projections(selection)
        if len(projections) == 0:
            return self._regular_band_structure(bands)
        else:
            return self._fat_band_structure(bands, projections, width)

    def _regular_band_structure(self, bands):
        kdists = self._kpoints.distances()
        return [self._scatter(name, kdists, lines) for name, lines in bands.items()]

    def _fat_band_structure(self, bands, projections, width):
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

    def _read_projections(self, selection):
        if selection is None:
            return {}
        return self._projectors._read_elements(selection, self._raw.projections)

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
