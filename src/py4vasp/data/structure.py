from py4vasp.data import _util
from py4vasp.exceptions import RefinementException
import functools
import numpy as np


class StructureViewer:
    """Collection of data and elements to be displayed in a structure viewer"""

    def __init__(self, structure, show_cell=True, supercell=None, show_axes=False, axes_length=3, arrows=None):
        self.structure = structure
        self.show_cell = show_cell
        self.supercell = supercell
        self.show_axes = show_axes
        self.axes_length = axes_length
        self.arrows = arrows

    def with_arrows(self, arrows):
        self.arrows = arrows
        return self

    def show(self):
        import nglview
        from nglview.shape import Shape

        structure = self.structure.to_pymatgen()
        if self.supercell is not None:
            structure.make_supercell(self.supercell)

        view = nglview.show_pymatgen(structure)
        if self.show_cell:
            view.add_representation(repr_type="unitcell")
        if self.show_axes or self.arrows is not None:
            shape = Shape(view=view)
            if self.show_axes:
                shape.add_arrow(
                    [0, 0, 0], [self.axes_length, 0, 0], [1, 0, 0], 0.2)
                shape.add_arrow(
                    [0, 0, 0], [0, self.axes_length, 0], [0, 1, 0], 0.2)
                shape.add_arrow(
                    [0, 0, 0], [0, 0, self.axes_length], [0, 0, 1], 0.2)
            if self.arrows is not None:
                for (x, y, z), (vx, vy, vz) in zip(structure.cart_coords, self.arrows):
                    shape.add_arrow(
                        [x, y, z], [x+vx, y+vy, z+vz], [0.1, 0.1, 0.8], 0.2)

        return view


class Structure:
    def __init__(self, raw_structure):
        self._raw = raw_structure
        self.structure_viewer = None

    def read(self):
        return self.to_dict()

    def to_dict(self):
        return {
            "cell": self._raw.cell.lattice_vectors[:],
            "cartesian_positions": self._raw.cartesian_positions[:],
            "species": list(self._raw.species),
        }

    def __len__(self):
        return len(self._raw.cartesian_positions)
        
    def to_pymatgen(self):
        import pymatgen as mg
        return mg.Structure(
            lattice=mg.Lattice(self._raw.cell.lattice_vectors),
            species=[specie.decode("ascii") for specie in self._raw.species],
            coords=self._raw.cartesian_positions,
            coords_are_cartesian=True
        )

    def plot(self, show_cell=True, supercell=None, show_axes=False, axes_length=3):
        self.structure_viewer = StructureViewer(
            self, show_cell=show_cell, supercell=supercell, show_axes=show_axes, axes_length=axes_length)
        return self.structure_viewer.show()

    def plot_arrows(self, arrows):
        if self.structure_viewer is None:
            self.plot()
        self.structure_viewer = self.structure_viewer.with_arrows(arrows)
        return self.structure_viewer.show()