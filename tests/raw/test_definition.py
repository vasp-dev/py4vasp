# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
from unittest.mock import patch

from py4vasp import raw
from py4vasp._raw.definition import schema


def test_all_quantities_have_default():
    for quantity, source in schema.sources.items():
        if quantity == "current_density":
            # currently no default current density is implemented
            assert "default" not in source
        else:
            assert "default" in source


def test_schema_is_valid():
    schema.verify()


def test_structure_links_symmetry():
    from py4vasp._raw.schema import Link

    for name in ("default", "final"):
        symmetry = schema.sources["structure"][name].data.symmetry
        assert isinstance(symmetry, Link)
        assert symmetry.quantity == "symmetry"


def test_structure_has_phonon_source():
    # A dispersion run writes no other structure to the file, so the primitive cell of
    # the phonon calculation is a source of the structure in its own right.
    from py4vasp._raw.schema import Link

    source = schema.sources["structure"]["phonon"].data
    assert source.cell == Link("cell", "phonon")
    assert source.stoichiometry == Link("stoichiometry", "phonon")
    assert source.positions == "results/phonons/primitive/position_ions"


def test_phonon_mode_has_dispersion_source():
    # A dispersion reports the same quantity as a linear response calculation, only
    # resolved along a path instead of at the zone centre.
    from py4vasp._raw.schema import Link

    source = schema.sources["phonon_mode"]["dispersion"].data
    assert source.structure == Link("structure", "phonon")
    assert source.frequencies == "results/phonons/frequencies"
    assert source.eigenvectors == "results/phonons/eigenvectors"
    assert source.qpoints == Link("kpoint", "phonon")


def test_get_schema(complex_schema):
    mock_schema, _ = complex_schema
    with patch("py4vasp._raw.definition.schema", mock_schema):
        assert raw.get_schema() == str(mock_schema)


def test_get_selections(complex_schema):
    mock_schema, _ = complex_schema
    with patch("py4vasp._raw.definition.schema", mock_schema):
        for quantity in mock_schema.sources.keys():
            assert raw.selections(quantity) == mock_schema.selections(quantity)
