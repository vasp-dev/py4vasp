from contextlib import contextmanager
import functools
import itertools
import numpy as np
import plotly.graph_objects as go
from .projectors import Projectors
import py4vasp.raw as raw


class Band:
    def __init__(self, raw_band):
        self._raw = raw_band
        self._fermi_energy = raw_band.fermi_energy
        self._kpoints = raw_band.kpoints
        self._kdists = None
        self._bands = raw_band.eigenvalues
        self._spin_polarized = len(self._bands) == 2
        scale = raw_band.cell.scale
        lattice_vectors = raw_band.cell.lattice_vectors
        self._cell = scale * lattice_vectors
        self._line_length = raw_band.line_length
        self._num_lines = len(self._kpoints) // self._line_length
        self._indices = raw_band.label_indices
        self._labels = raw_band.labels
        if raw_band.projectors is not None:
            self._projectors = Projectors(raw_band.projectors)
        self._projections = raw_band.projections

    @classmethod
    @contextmanager
    def from_file(cls, file=None):
        if file is None or isinstance(file, str):
            with raw.File(file) as local_file:
                yield cls(local_file.band())
        else:
            yield cls(file.band())

    def read(self, selection=None):
        res = {
            "kpoints": self._kpoints[:],
            "kpoint_labels": self._kpoint_labels(),
            "fermi_energy": self._fermi_energy,
            **self._shift_bands_by_fermi_energy(),
            "projections": self._read_projections(selection),
        }
        res["kpoint_distances"] = self._kpoint_distances(res["kpoints"])
        return res

    def plot(self, selection=None, width=0.5):
        ticks = self._ticks()
        labels = self._ticklabels()
        data = self._band_structure(selection, width)
        default = {
            "xaxis": {"tickmode": "array", "tickvals": ticks, "ticktext": labels},
            "yaxis": {"title": {"text": "Energy (eV)"}},
        }
        return go.Figure(data=data, layout=default)

    def _shift_bands_by_fermi_energy(self):
        if self._spin_polarized:
            return {
                "up": self._bands[0] - self._fermi_energy,
                "down": self._bands[1] - self._fermi_energy,
            }
        else:
            return {"bands": self._bands[0] - self._fermi_energy}

    def _kpoint_distances(self, kpoints=None):
        if self._kdists is not None:
            return self._kdists
        if kpoints is None:
            kpoints = self._kpoints[:]
        cartesian_kpoints = np.linalg.solve(self._cell, kpoints.T).T
        kpoint_lines = np.split(cartesian_kpoints, self._num_lines)
        kpoint_norms = [np.linalg.norm(line - line[0], axis=1) for line in kpoint_lines]
        concatenate_distances = lambda current, addition: (
            np.concatenate((current, addition + current[-1]))
        )
        self._kdists = functools.reduce(concatenate_distances, kpoint_norms)
        return self._kdists

    def _band_structure(self, selection, width):
        bands = self._shift_bands_by_fermi_energy()
        projections = self._read_projections(selection)
        if len(projections) == 0:
            return self._regular_band_structure(bands)
        else:
            return self._fat_band_structure(bands, projections, width)

    def _regular_band_structure(self, bands):
        kdists = self._kpoint_distances()
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
        kdists = self._kpoint_distances()
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
        return self._read_elements(selection)

    def _read_elements(self, selection):
        res = {}
        for select in self._projectors.parse_selection(selection):
            atom, orbital, spin = self._projectors.select(*select)
            label = self._merge_labels([atom.label, orbital.label, spin.label])
            index = (spin.indices, atom.indices, orbital.indices)
            res[label] = self._read_element(index)
        return res

    def _merge_labels(self, labels):
        return "_".join(filter(None, labels))

    def _read_element(self, index):
        sum_weight = lambda weight, i: weight + self._projections[i]
        zero_weight = np.zeros(self._bands.shape[1:])
        return functools.reduce(sum_weight, itertools.product(*index), zero_weight)

    def _kpoint_labels(self):
        if self._indices is None or self._labels is None:
            return None
        # convert from input kpoint list to full list
        labels = np.zeros(len(self._kpoints), dtype=self._labels.dtype)
        indices = np.array(self._indices)
        indices = self._line_length * (indices // 2) + indices % 2 - 1
        labels[indices] = self._labels
        return [l.decode().strip() for l in labels]

    def _ticks(self):
        kdists = self._kpoint_distances()
        return [*kdists[:: self._line_length], kdists[-1]]

    def _ticklabels(self):
        labels = [" "] * (self._num_lines + 1)
        if self._indices is None or self._labels is None:
            return labels
        for index, label in zip(self._indices, self._labels):
            i = index // 2  # line has 2 ends
            label = label.decode().strip()
            labels[i] = (labels[i] + "|" + label) if labels[i].strip() else label
        return labels
