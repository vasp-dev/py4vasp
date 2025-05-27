from py4vasp._third_party import graph
from py4vasp._util import slicing, index
from py4vasp._calculation.structure import Structure
from py4vasp._third_party.view import GridQuantity
from py4vasp._third_party.graph import Contour

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

    def to_contour(self, selections, a=None, b=None, c=None, normal=None, supercell=None):
        cut, fraction = slicing.get_cut(a, b, c)
        plane = slicing.plane(self._structure.lattice_vectors(), cut, normal)

        def _make_contour(selection):
            contour = Contour(
                slicing.grid_scalar(self._selector[selection].T, plane, fraction), 
                plane, 
                label=self._selector.label(selection) or "", 
                isolevels=True, 
            )
            if supercell is not None: contour.supercell = np.ones(2, dtype=np.int_) * supercell
            return contour
        
        contours = [
            _make_contour(selection)
            for selection in selections
        ]
        return graph.Graph(contours)
    
    

    def to_quiver(self, selections, a=None, b=None, c=None, supercell=None, normal=None):
        cut, fraction = slicing.get_cut(a, b, c)
        plane = slicing.plane(self._structure.lattice_vectors(), cut, normal)

        def _make_contour(selection):
            contour = Contour(
                slicing.grid_vector(self._selector[selection].T, plane, fraction), 
                plane, 
                label=self._selector.label(selection) or "", 
                isolevels=True, 
            )
            if supercell is not None: contour.supercell = np.ones(2, dtype=np.int_) * supercell
            return contour
        
        contours = [
            _make_contour(selection)
            for selection in selections
        ]
        return graph.Graph(contours)