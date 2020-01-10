import re
import functools
import itertools
import numpy as np
import pandas as pd
import cufflinks as cf
from collections import namedtuple

cf.go_offline()


class Dos:
    _Index = namedtuple("_Index", "spin, atom, orbital")
    _Atom = namedtuple("_Atom", "indices, label")
    _Orbital = namedtuple("_Orbital", "indices, label")
    _Spin = namedtuple("_Spin", "indices, label")

    def __init__(self, vaspout):
        self._fermi_energy = vaspout["results/dos/efermi"][()]
        self._energies = vaspout["results/dos/energies"]
        self._dos = vaspout["results/dos/dos"]
        self._spin_polarized = self._dos.shape[0] == 2
        self._has_partial_dos = vaspout["results/dos/jobpar"][()] == 1
        if self._has_partial_dos:
            self._init_partial_dos(vaspout)

    def _init_partial_dos(self, vaspout):
        self._partial_dos = vaspout["results/dos/dospar"]
        ion_types = vaspout["results/positions/ion_types"]
        ion_types = [type.decode().strip() for type in ion_types]
        self._init_atom_dict(ion_types, vaspout["results/positions/number_ion_types"])
        orbitals = vaspout["results/projectors/lchar"]
        orbitals = [orb.decode().strip() for orb in orbitals]
        self._init_orbital_dict(orbitals)
        self._init_spin_dict()

    def _init_atom_dict(self, ion_types, number_ion_types):
        num_atoms = self._partial_dos.shape[1]
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
        num_orbitals = self._partial_dos.shape[2]
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

    def plot(self, selection=None):
        df = self.to_frame(selection)
        if self._spin_polarized:
            for col in filter(lambda col: "down" in col, df):
                df[col] = -df[col]
        default = {
            "x": "energies",
            "xTitle": "Energy (eV)",
            "yTitle": "DOS (1/eV)",
            "asFigure": True,
        }
        return df.iplot(**default)

    def read(self, selection=None):
        return self.to_dict(selection)

    def to_dict(self, selection=None):
        return {**self._read_data(selection), "fermi_energy": self._fermi_energy}

    def to_frame(self, selection=None):
        df = pd.DataFrame(self._read_data(selection))
        df.fermi_energy = self._fermi_energy
        return df

    def _read_data(self, selection):
        return {
            **self._read_energies(),
            **self._read_total_dos(),
            **self._read_partial_dos(selection),
        }

    def _read_energies(self):
        return {"energies": self._energies[:] - self._fermi_energy}

    def _read_total_dos(self):
        if self._spin_polarized:
            return {"up": self._dos[0, :], "down": self._dos[1, :]}
        else:
            return {"total": self._dos[0, :]}

    def _read_partial_dos(self, selection):
        if selection is None:
            return {}
        self._raise_error_if_partial_Dos_not_available()
        parts = self._parse_filter(selection)
        return self._read_elements(parts)

    def _raise_error_if_partial_Dos_not_available(self):
        if not self._has_partial_dos:
            raise ValueError(
                "Filtering requires partial DOS which was not found in HDF5 file."
            )

    def _parse_filter(self, selection):
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
        sum_dos = lambda dos, i: dos + self._partial_dos[i]
        zero_dos = np.zeros(len(self._energies))
        return functools.reduce(sum_dos, itertools.product(*index), zero_dos)
