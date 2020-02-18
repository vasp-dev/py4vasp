from contextlib import AbstractContextManager
import h5py
import py4vasp.raw as raw


class File(AbstractContextManager):
    default_filename = "vaspout.h5"

    def __init__(self, filename=None):
        filename = filename or File.default_filename
        self._h5f = h5py.File(filename, "r")
        self.closed = False

    def dos(self):
        self._assert_not_closed()
        return raw.Dos(
            fermi_energy=self._h5f["results/electron_dos/efermi"][()],
            energies=self._h5f["results/electron_dos/energies"],
            dos=self._h5f["results/electron_dos/dos"],
            projectors=self.projectors(),
            projections=self._safe_get_key("results/electron_dos/dospar"),
        )

    def band(self):
        self._assert_not_closed()
        return raw.Band(
            fermi_energy=self._h5f["results/electron_dos/efermi"][()],
            line_length=self._h5f["input/kpoints/number_kpoints"][()],
            kpoints=self._h5f["results/electron_eigenvalues/kpoint_coords"],
            eigenvalues=self._h5f["results/electron_eigenvalues/eigenvalues"],
            labels=self._safe_get_key("input/kpoints/labels_kpoints"),
            label_indices=self._safe_get_key("input/kpoints/positions_labels_kpoints"),
            cell=self.cell(),
            projectors=self.projectors(),
            projections=self._safe_get_key("results/projectors/par"),
        )

    def projectors(self):
        self._assert_not_closed()
        if "results/projectors" not in self._h5f:
            return None
        return raw.Projectors(
            ion_types=self._h5f["results/positions/ion_types"],
            number_ion_types=self._h5f["results/positions/number_ion_types"],
            orbital_types=self._h5f["results/projectors/lchar"],
            number_spins=self._h5f["results/electron_eigenvalues/ispin"][()],
        )

    def cell(self):
        self._assert_not_closed()
        return raw.Cell(
            scale=self._h5f["results/positions/scale"][()],
            lattice_vectors=self._h5f["results/positions/lattice_vectors"],
        )

    def convergence(self):
        self._assert_not_closed()
        return raw.Convergence(
            labels=self._h5f["intermediate/history/energies_tags"],
            energies=self._h5f["intermediate/history/energies"],
        )

    def close(self):
        self._h5f.close()
        self.closed = True

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def _assert_not_closed(self):
        assert not self.closed, "I/O operation on closed file."

    def _safe_get_key(self, key):
        if key in self._h5f:
            return self._h5f[key]
        else:
            return None
