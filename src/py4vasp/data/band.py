import functools
import numpy as np
import plotly.graph_objects as go


class Band:
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

    def read(self):
        kpoints = self._kpoints[:]
        return {
            "kpoints": kpoints,
            "kpoint_distances": self._kpoint_distances(kpoints),
            "kpoint_labels": self._kpoint_labels(),
            "fermi_energy": self._fermi_energy,
            **self._shift_bands_by_fermi_energy(),
        }

    def plot(self):
        band = self.read()
        num_bands = band["bands"].shape[-1]
        kdists = band["kpoint_distances"]
        # insert NaN to split separate bands
        kdist = np.tile([*kdists, np.NaN], num_bands)
        bands = np.append(
            band["bands"], [np.repeat(np.NaN, num_bands)], axis=0
        ).flatten(order="F")
        ticks = [*kdists[:: self._line_length], kdists[-1]]
        labels = self._ticklabels()
        default = {
            "xaxis": {"tickmode": "array", "tickvals": ticks, "ticktext": labels},
            "yaxis": {"title": {"text": "Energy (eV)"}},
        }
        return go.Figure(data=go.Scatter(x=kdist, y=bands), layout=default)

    def _shift_bands_by_fermi_energy(self):
        if self._spin_polarized:
            return {
                "up": self._bands[0] - self._fermi_energy,
                "down": self._bands[1] - self._fermi_energy,
            }
        else:
            return {"bands": self._bands[0] - self._fermi_energy}

    def _kpoint_distances(self, kpoints):
        cartesian_kpoints = np.linalg.solve(self._cell, kpoints.T).T
        kpoint_lines = np.split(cartesian_kpoints, self._num_lines)
        kpoint_norms = [np.linalg.norm(line - line[0], axis=1) for line in kpoint_lines]
        concatenate_distances = lambda current, addition: (
            np.concatenate((current, addition + current[-1]))
        )
        return functools.reduce(concatenate_distances, kpoint_norms)

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
        labels = [""] * (self._num_lines + 1)
        for index, label in zip(self._indices, self._labels):
            i = index // 2  # line has 2 ends
            label = label.decode().strip()
            labels[i] = (labels[i] + "|" + label) if labels[i] else label
        return labels
