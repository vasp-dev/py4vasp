# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
from py4vasp import _demo, raw


def Sr2TiO4(included_potential):
    structure = _demo.structure.Sr2TiO4()
    shape = (_demo.NONPOLARIZED, *_demo.GRID_DIMENSIONS)
    include_xc = included_potential in ("xc", "all")
    include_hartree = included_potential in ("hartree", "all")
    include_ionic = included_potential in ("ionic", "all")
    return raw.Potential(
        structure=structure,
        total_potential=_demo.wrap_random_data(shape),
        xc_potential=_demo.wrap_random_data(shape, present=include_xc),
        hartree_potential=_demo.wrap_random_data(shape, present=include_hartree),
        ionic_potential=_demo.wrap_random_data(shape, present=include_ionic),
    )


def Fe3O4(selection, included_potential):
    structure = _demo.structure.Fe3O4()
    shape_polarized = (_demo.number_components(selection), *_demo.GRID_DIMENSIONS)
    shape_simple = (_demo.NONPOLARIZED, *_demo.GRID_DIMENSIONS)
    include_xc = included_potential in ("xc", "all")
    include_hartree = included_potential in ("hartree", "all")
    include_ionic = included_potential in ("ionic", "all")
    return raw.Potential(
        structure=structure,
        total_potential=_demo.wrap_random_data(shape_polarized),
        xc_potential=_demo.wrap_random_data(shape_polarized, present=include_xc),
        hartree_potential=_demo.wrap_random_data(shape_simple, present=include_hartree),
        ionic_potential=_demo.wrap_random_data(shape_simple, present=include_ionic),
    )
