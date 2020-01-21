import h5py
import py4vasp.raw as raw


class File:
    def __init__(self, filename="vaspout.h5"):
        self._h5f = h5py.File(filename, "r")

    def dos(self):
        return raw.Dos(
            fermi_energy=self._h5f["results/dos/efermi"],
            energies=self._h5f["results/dos/energies"],
            dos=self._h5f["results/dos/dos"],
            projectors=self.projectors(),
        )

    def band(self):
        return raw.Band(
            fermi_energy=self._h5f["results/dos/efermi"],
            line_length=self._h5f["input/kpoints/number_kpoints"],
            kpoints=self._h5f["results/eigenvalues/kpoint_coords"],
            eigenvalues=self._h5f["results/eigenvalues/eigenvalues"],
            labels=self._safe_get_key("input/kpoints/labels_kpoints"),
            label_indices=self._safe_get_key("input/kpoints/positions_labels_kpoints"),
            cell=self.cell(),
            projectors=self.projectors(),
        )

    def projectors(self):
        if "results/projectors" not in self._h5f:
            return None
        return raw.Projectors(
            ion_types=self._h5f["results/positions/ion_types"],
            number_ion_types=self._h5f["results/positions/number_ion_types"],
            orbital_types=self._h5f["results/projectors/lchar"],
            dos=self._h5f["results/dos/dospar"],
            bands=self._h5f["results/projectors/par"],
        )

    def cell(self):
        return raw.Cell(
            scale=self._h5f["results/positions/scale"],
            lattice_vectors=self._h5f["results/positions/lattice_vectors"],
        )

    def close(self):
        self._h5f.close()

    def _safe_get_key(self, key):
        if key in self._h5f:
            return self._h5f[key]
        else:
            return None
