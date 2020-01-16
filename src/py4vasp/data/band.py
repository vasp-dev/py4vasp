import re
import functools
import itertools
import numpy as np
import plotly.graph_objects as go
from collections import namedtuple


class Band:
    _Index = namedtuple("_Index", "spin, atom, orbital")
    _Atom = namedtuple("_Atom", "indices, label")
    _Orbital = namedtuple("_Orbital", "indices, label")
    _Spin = namedtuple("_Spin", "indices, label")

    def __init__(self, vaspout):
        self._fermi_energy = vaspout["results/dos/efermi"][()]
        self._kpoints = vaspout["results/eigenvalues/kpoint_coords"]
        self._bands = vaspout["results/eigenvalues/eigenvalues"]
        self._spin_polarized = len(self._bands) == 2
        scale = vaspout["results/positions/scale"][()]
        lattice_vectors = vaspout["results/positions/lattice_vectors"]
        self._cell = scale * lattice_vectors
        self._line_length = vaspout["input/kpoints/number_kpoints"][()]
        self._num_lines = len(self._kpoints) // self._line_length
        indices_key = "input/kpoints/positions_labels_kpoints"
        self._indices = vaspout[indices_key] if indices_key in vaspout else []
        labels_key = "input/kpoints/labels_kpoints"
        self._labels = vaspout[labels_key] if labels_key in vaspout else []
        self._has_projectors = "results/projectors" in vaspout
        if self._has_projectors:
            self._init_projectors(vaspout)

    def _init_projectors(self, vaspout):
        self._projections = vaspout["results/projectors/par"]
        ion_types = vaspout["results/positions/ion_types"]
        ion_types = [type.decode().strip() for type in ion_types]
        self._init_atom_dict(ion_types, vaspout["results/positions/number_ion_types"])
        orbitals = vaspout["results/projectors/lchar"]
        orbitals = [orb.decode().strip() for orb in orbitals]
        self._init_orbital_dict(orbitals)
        self._init_spin_dict()

    def _init_atom_dict(self, ion_types, number_ion_types):
        num_atoms = self._projections.shape[1]
        all_atoms = self._Atom(indices=range(num_atoms), label=None)
        self._atom_dict = {"*": all_atoms}
        start = 0
        for type, number in zip(ion_types, number_ion_types):
            _range = range(start, start + number)
            self._atom_dict[type] = self._Atom(indices=_range, label=type)
            for i in _range:
                # create labels like Si_1, Si_2, Si_3 (starting at 1)
                label = type + "_" + str(_range.index(i) + 1)
                self._atom_dict[str(i + 1)] = self._Atom(indices=[i], label=label)
            start += number
        # atoms may be preceeded by :
        for key in self._atom_dict.copy():
            self._atom_dict[key + ":"] = self._atom_dict[key]

    def _init_orbital_dict(self, orbitals):
        num_orbitals = self._projections.shape[2]
        all_orbitals = self._Orbital(indices=range(num_orbitals), label=None)
        self._orbital_dict = {"*": all_orbitals}
        for i, orbital in enumerate(orbitals):
            self._orbital_dict[orbital] = self._Orbital(indices=[i], label=orbital)
        if "px" in self._orbital_dict:
            self._orbital_dict["p"] = self._Orbital(indices=range(1, 4), label="p")
            self._orbital_dict["d"] = self._Orbital(indices=range(4, 9), label="d")
            self._orbital_dict["f"] = self._Orbital(indices=range(9, 16), label="f")

    def _init_spin_dict(self):
        labels = ["up", "down"] if self._spin_polarized else [None]
        self._spin_dict = {
            key: self._Spin(indices=[i], label=key) for i, key in enumerate(labels)
        }

    def read(self, selection=None):
        kpoints = self._kpoints[:]
        return {
            "kpoints": kpoints,
            "kpoint_distances": self._kpoint_distances(kpoints),
            "kpoint_labels": self._kpoint_labels(),
            "fermi_energy": self._fermi_energy,
            **self._shift_bands_by_fermi_energy(),
            "projections": self._read_projections(selection),
        }

    def plot(self, selection=None, width=0.5):
        kdists = self._kpoint_distances(self._kpoints[:])
        fatband_kdists = np.concatenate((kdists, np.flip(kdists)))
        bands = self._shift_bands_by_fermi_energy()
        projections = self._read_projections(selection)
        ticks = [*kdists[:: self._line_length], kdists[-1]]
        labels = self._ticklabels()
        data = []
        for key, lines in bands.items():
            if len(projections) == 0:
                data.append(self._scatter(key, kdists, lines))
            for name, proj in projections.items():
                if self._spin_polarized and not key in name:
                    continue
                upper = lines + width * proj
                lower = lines - width * proj
                fatband_lines = np.concatenate((lower, np.flip(upper, axis=0)), axis=0)
                plot = self._scatter(name, fatband_kdists, fatband_lines)
                plot.fill = "toself"
                plot.mode = "none"
                data.append(plot)
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

    def _read_projections(self, selection):
        if selection is None:
            return {}
        parts = self._parse_selection(selection)
        return self._read_elements(parts)

    def _scatter(self, name, kdists, lines):
        # insert NaN to split separate lines
        num_bands = lines.shape[-1]
        kdists = np.tile([*kdists, np.NaN], num_bands)
        lines = np.append(lines, [np.repeat(np.NaN, num_bands)], axis=0)
        return go.Scatter(x=kdists, y=lines.flatten(order="F"), name=name)

    def _kpoint_distances(self, kpoints):
        cartesian_kpoints = np.linalg.solve(self._cell, kpoints.T).T
        kpoint_lines = np.split(cartesian_kpoints, self._num_lines)
        kpoint_norms = [np.linalg.norm(line - line[0], axis=1) for line in kpoint_lines]
        concatenate_distances = lambda current, addition: (
            np.concatenate((current, addition + current[-1]))
        )
        return functools.reduce(concatenate_distances, kpoint_norms)

    def _parse_selection(self, selection):
        atom = self._atom_dict["*"]
        selection = re.sub("\s*:\s*", ": ", selection)
        for part in re.split("[ ,]+", selection):
            if part in self._orbital_dict:
                orbital = self._orbital_dict[part]
            else:
                atom = self._atom_dict[part]
                orbital = self._orbital_dict["*"]
            if ":" not in part:  # exclude ":" because it starts a new atom
                for spin in self._spin_dict.values():
                    yield atom, orbital, spin

    def _read_elements(self, parts):
        res = {}
        for atom, orbital, spin in parts:
            label = self._merge_labels([atom.label, orbital.label, spin.label])
            index = self._Index(spin.indices, atom.indices, orbital.indices)
            res[label] = self._read_element(index)
        return res

    def _merge_labels(self, labels):
        return "_".join(filter(None, labels))

    def _read_element(self, index):
        sum_weight = lambda weight, i: weight + self._projections[i]
        zero_weight = np.zeros(self._bands.shape[1:])
        return functools.reduce(sum_weight, itertools.product(*index), zero_weight)

    def _kpoint_labels(self):
        if len(self._labels) == 0:
            return None
        # convert from input kpoint list to full list
        labels = np.zeros(len(self._kpoints), dtype=self._labels.dtype)
        indices = self._indices[:]
        indices = self._line_length * (indices // 2) + indices % 2 - 1
        labels[indices] = self._labels
        return [l.decode().strip() for l in labels]

    def _ticklabels(self):
        labels = [" "] * (self._num_lines + 1)
        for index, label in zip(self._indices, self._labels):
            i = index // 2  # line has 2 ends
            label = label.decode().strip()
            labels[i] = (labels[i] + "|" + label) if labels[i].strip() else label
        return labels
