# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
from typing import Iterable, NamedTuple

import numpy as np
import pytest

from py4vasp import exception
from py4vasp._data.selection import Selection
from py4vasp._util import select
from py4vasp.data import Projector

Index = Projector.Index


class SelectionTestCase(NamedTuple):
    equivalent_formats: Iterable[str]
    reference_selections: Iterable[Index]


@pytest.fixture
def Sr2TiO4(raw_data):
    return Projector.from_data(raw_data.projector("Sr2TiO4"))


@pytest.fixture
def Fe3O4(raw_data):
    return Projector.from_data(raw_data.projector("Fe3O4"))


@pytest.fixture
def missing_orbitals(raw_data):
    return Projector.from_data(raw_data.projector("without_orbitals"))


def test_Sr2TiO4_selection(Sr2TiO4):
    ref = Sr2TiO4_selection()
    check_projector_selection(Sr2TiO4, ref)


def test_Fe3O4_selection(Fe3O4):
    ref = Fe3O4_selection()
    check_projector_selection(Fe3O4, ref)


def test_missing_orbital_selection(missing_orbitals):
    with pytest.raises(exception.IncorrectUsage):
        missing_orbitals.select()


def check_projector_selection(proj, ref):
    all_ = select.all
    default = Index(ref["atom"][all_], ref["orbital"][all_], ref["spin"][all_])
    for atom, ref_atom in ref["atom"].items():
        assert proj.select(atom=atom) == default._replace(atom=ref_atom)
    for orbital, ref_orbital in ref["orbital"].items():
        assert proj.select(orbital=orbital) == default._replace(orbital=ref_orbital)
    for spin, ref_spin in ref["spin"].items():
        assert proj.select(spin=spin) == default._replace(spin=ref_spin)


def Sr2TiO4_selection():
    return {
        "atom": {
            "Sr": Selection(indices=slice(0, 2), label="Sr"),
            "Ti": Selection(indices=slice(2, 3), label="Ti"),
            "O": Selection(indices=slice(3, 7), label="O"),
            "1": Selection(indices=slice(0, 1), label="Sr_1"),
            "2": Selection(indices=slice(1, 2), label="Sr_2"),
            "3": Selection(indices=slice(2, 3), label="Ti_1"),
            "4": Selection(indices=slice(3, 4), label="O_1"),
            "5": Selection(indices=slice(4, 5), label="O_2"),
            "6": Selection(indices=slice(5, 6), label="O_3"),
            "7": Selection(indices=slice(6, 7), label="O_4"),
            "1:3": Selection(indices=slice(0, 3), label="1:3"),
            "4:7": Selection(indices=slice(3, 7), label="4:7"),
            select.all: Selection(indices=slice(0, 7)),
        },
        "orbital": {
            "s": Selection(indices=slice(0, 1), label="s"),
            "px": Selection(indices=slice(3, 4), label="px"),
            "py": Selection(indices=slice(1, 2), label="py"),
            "pz": Selection(indices=slice(2, 3), label="pz"),
            "dxy": Selection(indices=slice(4, 5), label="dxy"),
            "dxz": Selection(indices=slice(7, 8), label="dxz"),
            "dyz": Selection(indices=slice(5, 6), label="dyz"),
            "dz2": Selection(indices=slice(6, 7), label="dz2"),
            "dx2y2": Selection(indices=slice(8, 9), label="dx2y2"),
            "fxyz": Selection(indices=slice(10, 11), label="fxyz"),
            "fxz2": Selection(indices=slice(13, 14), label="fxz2"),
            "fx3": Selection(indices=slice(15, 16), label="fx3"),
            "fyz2": Selection(indices=slice(11, 12), label="fyz2"),
            "fy3x2": Selection(indices=slice(9, 10), label="fy3x2"),
            "fzx2": Selection(indices=slice(14, 15), label="fzx2"),
            "fz3": Selection(indices=slice(12, 13), label="fz3"),
            "p": Selection(indices=slice(1, 4), label="p"),
            "d": Selection(indices=slice(4, 9), label="d"),
            "f": Selection(indices=slice(9, 16), label="f"),
            select.all: Selection(indices=slice(0, 16)),
        },
        "spin": {
            "total": Selection(indices=slice(0, 1), label="total"),
            select.all: Selection(indices=slice(0, 1)),
        },
    }


def Fe3O4_selection():
    return {
        "atom": {
            "Fe": Selection(indices=slice(0, 3), label="Fe"),
            "O": Selection(indices=slice(3, 7), label="O"),
            "1": Selection(indices=slice(0, 1), label="Fe_1"),
            "2": Selection(indices=slice(1, 2), label="Fe_2"),
            "3": Selection(indices=slice(2, 3), label="Fe_3"),
            "4": Selection(indices=slice(3, 4), label="O_1"),
            "5": Selection(indices=slice(4, 5), label="O_2"),
            "6": Selection(indices=slice(5, 6), label="O_3"),
            "7": Selection(indices=slice(6, 7), label="O_4"),
            "1:2": Selection(indices=slice(0, 2), label="1:2"),
            "4:5": Selection(indices=slice(3, 5), label="4:5"),
            select.all: Selection(indices=slice(0, 7)),
        },
        "orbital": {
            "s": Selection(indices=slice(0, 1), label="s"),
            "p": Selection(indices=slice(1, 2), label="p"),
            "d": Selection(indices=slice(2, 3), label="d"),
            "f": Selection(indices=slice(3, 4), label="f"),
            select.all: Selection(indices=slice(0, 4)),
        },
        "spin": {
            "total": Selection(indices=slice(0, 2), label="total"),
            "up": Selection(indices=slice(0, 1), label="up"),
            "down": Selection(indices=slice(1, 2), label="down"),
            select.all: Selection(indices=slice(0, 2)),
        },
    }


def test_Sr2TiO4_parse_selection(Sr2TiO4):
    testcases = Sr2TiO4_testcases()
    check_parse_selection(Sr2TiO4, testcases)


def test_Fe3O4_parse_selection(Fe3O4):
    testcases = Fe3O4_testcases()
    check_parse_selection(Fe3O4, testcases)


def test_missing_orbitals_selection(missing_orbitals):
    with pytest.raises(exception.IncorrectUsage):
        next(missing_orbitals.parse_selection())


def check_parse_selection(projectors, testcases):
    for testcase in testcases:
        for format in testcase.equivalent_formats:
            selections = projectors.parse_selection(format)
            assert list(selections) == list(testcase.reference_selections)


def Sr2TiO4_testcases():
    all_ = select.all
    return (
        SelectionTestCase(
            equivalent_formats=("Sr", f"Sr({all_})"),
            reference_selections=(Index(atom="Sr", orbital=all_, spin=all_),),
        ),
        SelectionTestCase(
            equivalent_formats=("Ti(s,p)", "Ti (s)  Ti (p)"),
            reference_selections=(
                Index(atom="Ti", orbital="s", spin=all_),
                Index(atom="Ti", orbital="p", spin=all_),
            ),
        ),
        SelectionTestCase(
            equivalent_formats=("Ti 5", f"Ti( {all_} ), 5( {all_} )"),
            reference_selections=(
                Index(atom="Ti", orbital=all_, spin=all_),
                Index(atom="5", orbital=all_, spin=all_),
            ),
        ),
        SelectionTestCase(
            equivalent_formats=("p, d", f"{all_}(p) {all_}(d)"),
            reference_selections=(
                Index(atom=all_, orbital="p", spin=all_),
                Index(atom=all_, orbital="d", spin=all_),
            ),
        ),
        SelectionTestCase(
            equivalent_formats=("O(d), 1 s", f"O(d), 1({all_}), {all_}(s)"),
            reference_selections=(
                Index(atom="O", orbital="d", spin=all_),
                Index(atom="1", orbital=all_, spin=all_),
                Index(atom=all_, orbital="s", spin=all_),
            ),
        ),
        SelectionTestCase(
            equivalent_formats=("Sr(p)Ti(s)O(s)", "p(Sr) s(Ti, O)"),
            reference_selections=(
                Index(atom="Sr", orbital="p", spin=all_),
                Index(atom="Ti", orbital="s", spin=all_),
                Index(atom="O", orbital="s", spin=all_),
            ),
        ),
        SelectionTestCase(
            equivalent_formats=("1 : 4", "1:4", "  1  :  4 "),
            reference_selections=(Index(atom="1:4", orbital=all_, spin=all_),),
        ),
    )


def Fe3O4_testcases():
    all_ = select.all
    return (
        SelectionTestCase(
            equivalent_formats=("Fe", "Fe(up,down)", f"Fe({all_}(up)), Fe(down)"),
            reference_selections=(
                Index(atom="Fe", orbital=all_, spin="up"),
                Index(atom="Fe", orbital=all_, spin="down"),
            ),
        ),
        SelectionTestCase(
            equivalent_formats=("O( s(up) p(down) )", "O(s(up))O(p(down))"),
            reference_selections=(
                Index(atom="O", orbital="s", spin="up"),
                Index(atom="O", orbital="p", spin="down"),
            ),
        ),
        SelectionTestCase(
            equivalent_formats=(
                "s p(up)d(total)",
                f"s(up,down),{all_}(p(up)),d(total)",
            ),
            reference_selections=(
                Index(atom=all_, orbital="s", spin="up"),
                Index(atom=all_, orbital="s", spin="down"),
                Index(atom=all_, orbital="p", spin="up"),
                Index(atom=all_, orbital="d", spin="total"),
            ),
        ),
        SelectionTestCase(
            equivalent_formats=("up (s)  down (p, d)", "s(up) p(down) d(down)"),
            reference_selections=(
                Index(atom=all_, orbital="s", spin="up"),
                Index(atom=all_, orbital="p", spin="down"),
                Index(atom=all_, orbital="d", spin="down"),
            ),
        ),
        SelectionTestCase(
            equivalent_formats=("2( p(up) )", "p(2(up))", "up(2(p))"),
            reference_selections=(Index(atom="2", orbital="p", spin="up"),),
        ),
        SelectionTestCase(
            equivalent_formats=("3:4(up)", "up (3 : 4)"),
            reference_selections=(Index(atom="3:4", orbital=all_, spin="up"),),
        ),
    )


def test_read_empty(Sr2TiO4):
    assert Sr2TiO4.read() == {}


@pytest.mark.xfail(reason="Not ported for new selection logic yet.")
def test_read_only_indices(Sr2TiO4):
    reference = {
        "Sr_p": (slice(0, 1), slice(0, 2), slice(1, 4)),
        "Ti_1_dxy": (slice(0, 1), slice(2, 3), slice(4, 5)),
    }
    assert Sr2TiO4.read(selection="Sr(p) 3(dxy)") == reference


@pytest.mark.xfail(reason="Not ported for new selection logic yet.")
def test_read_projections(Sr2TiO4, Assert):
    num_spins = 1
    num_atoms = 7
    num_orbitals = 10
    num_quantity = 25
    shape = (num_spins, num_atoms, num_orbitals, num_quantity)
    projections = np.arange(np.prod(shape)).reshape(shape)
    Sr_ref = np.sum(projections[0, 0:2, 1:4], axis=(0, 1))
    Ti_ref = projections[0, 2, 4]
    actual = Sr2TiO4.read(selection="Sr(p) 3(dxy)", projections=projections)
    Assert.allclose(actual["Sr_p"], Sr_ref)
    Assert.allclose(actual["Ti_1_dxy"], Ti_ref)


def test_missing_orbitals_read(missing_orbitals):
    with pytest.raises(exception.IncorrectUsage):
        missing_orbitals.read("any string")
    with pytest.raises(exception.IncorrectUsage):
        missing_orbitals.read("any string", "any data")


@pytest.mark.xfail(reason="Not ported for new selection logic yet.")
def test_error_parsing(Sr2TiO4):
    with pytest.raises(exception.IncorrectUsage):
        Sr2TiO4.read(selection="XX")
    with pytest.raises(exception.IncorrectUsage):
        number_instead_of_string = -1
        Sr2TiO4.read(selection=number_instead_of_string)


def test_incorrect_selection(Sr2TiO4):
    with pytest.raises(exception.IncorrectUsage):
        Sr2TiO4.select(atom="XX")
    with pytest.raises(exception.IncorrectUsage):
        Sr2TiO4.select(atom="100-900")
    with pytest.raises(exception.IncorrectUsage):
        Sr2TiO4.select(orbital="XX")
    with pytest.raises(exception.IncorrectUsage):
        Sr2TiO4.select(spin="XX")


def test_incorrect_reading_of_projections(Sr2TiO4):
    with pytest.raises(exception.IncorrectUsage):
        Sr2TiO4.read("Sr", [1, 2, 3])
    with pytest.raises(exception.IncorrectUsage):
        Sr2TiO4.read("Sr", np.zeros(3))


def test_print(Sr2TiO4, format_):
    actual, _ = format_(Sr2TiO4)
    reference = """
projectors:
    atoms: Sr, Ti, O
    orbitals: s, py, pz, px, dxy, dyz, dz2, dxz, dx2y2, fy3x2, fxyz, fyz2, fz3, fxz2, fzx2, fx3
    """.strip()
    assert actual == {"text/plain": reference}


def test_missing_orbitals_print(missing_orbitals, format_):
    actual, _ = format_(missing_orbitals)
    assert actual == {"text/plain": "no projectors"}


def test_factory_methods(raw_data, check_factory_methods):
    data = raw_data.projector("Sr2TiO4")
    check_factory_methods(Projector, data)
