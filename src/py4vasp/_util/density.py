from py4vasp._util import slicing, index
from py4vasp._calculation.structure import Structure
from py4vasp._third_party.view import GridQuantity

import numpy as np

#Density, CurrentDensity, ExcitonDensity, [PartialDensity], (BField/MagneticField)
class Visualizer:
    def __init__(self, structure: Structure, selector: index.Selector):
        self._selector = selector
        self._structure = structure

    def to_view(self, selections, supercell=1):
        viewer = self._structure.plot(supercell)
        viewer.grid_scalars = [
            GridQuantity((self._selector[selection].T)[np.newaxis], label=self._selector.label(selection)) #user_options)
            for selection in selections
        ]
        return viewer

    # def to_contour(self, selections, a, b, c, supercell, normal):
    #     cut, fraction = slicing.get_cut(a, b, c)
    #     plane = slicing.plane(self._structure.lattice_vectors(), cut, normal)
    #     contours = [
    #         self._contour(selector, selection, plane, fraction, supercell)
    #         for selection in selections
    #     ]
    #     return graph.Graph(contours)

    # def to_quiver(self, selections, a, b,c, supercell, normal):
    #     cut, fraction = slicing.get_cut(a, b, c)
    #     plane = slicing.plane(self._structure.lattice_vectors(), cut, normal)
    #     quiver_plots = [
    #         self._quiver_plot(selector, selection, plane, fraction, supercell)
    #         for selection in selections
    #     ]