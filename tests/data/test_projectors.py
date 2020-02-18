from py4vasp.data import Projectors
import py4vasp.raw as raw
import pytest
import numpy as np
from typing import NamedTuple, Iterable

Selection = Projectors.Selection
Index = Projectors.Index


class SelectionTestCase(NamedTuple):
    equivalent_formats: Iterable[str]
    reference_selections: Iterable[Index]


@pytest.fixture
def without_spin():
    proj = raw.Projectors(
        number_ion_types=np.array((2, 1, 4)),
        ion_types=np.array(("Sr", "Ti", "O "), dtype="S"),
        orbital_types=np.array(
            (" s", "py", "pz", "px", "dxy", "dyz", "dz2", "dxz", "x2-y2")
            + ("fy3x2", "fxyz", "fyz2", "fz3", "fxz2", "fzx2", "fx3"),
            dtype="S",
        ),
        number_spins=1,
    )
    return proj


def test_from_file(without_spin, mock_file, check_read):
    with mock_file("projectors", without_spin) as mocks:
        check_read(Projectors, mocks, without_spin)


@pytest.fixture
def spin_polarized(without_spin):
    without_spin.number_spins = 2
    return without_spin


@pytest.fixture
def for_selection(spin_polarized):
    index = np.cumsum(spin_polarized.number_ion_types)
    ref = {
        "atom": {
            "Sr": Selection(indices=range(index[0]), label="Sr"),
            "Ti": Selection(indices=range(index[0], index[1]), label="Ti"),
            "O": Selection(indices=range(index[1], index[2]), label="O"),
            "1": Selection(indices=(0,), label="Sr_1"),
            "2": Selection(indices=(1,), label="Sr_2"),
            "3": Selection(indices=(2,), label="Ti_1"),
            "4": Selection(indices=(3,), label="O_1"),
            "5": Selection(indices=(4,), label="O_2"),
            "6": Selection(indices=(5,), label="O_3"),
            "7": Selection(indices=(6,), label="O_4"),
            "1-3": Selection(indices=range(0, 3), label="1-3"),
            "4-7": Selection(indices=range(3, 7), label="4-7"),
            "*": Selection(indices=range(index[-1])),
        },
        "orbital": {
            "s": Selection(indices=(0,), label="s"),
            "px": Selection(indices=(3,), label="px"),
            "py": Selection(indices=(1,), label="py"),
            "pz": Selection(indices=(2,), label="pz"),
            "dxy": Selection(indices=(4,), label="dxy"),
            "dxz": Selection(indices=(7,), label="dxz"),
            "dyz": Selection(indices=(5,), label="dyz"),
            "dz2": Selection(indices=(6,), label="dz2"),
            "x2-y2": Selection(indices=(8,), label="x2-y2"),
            "fxyz": Selection(indices=(10,), label="fxyz"),
            "fxz2": Selection(indices=(13,), label="fxz2"),
            "fx3": Selection(indices=(15,), label="fx3"),
            "fyz2": Selection(indices=(11,), label="fyz2"),
            "fy3x2": Selection(indices=(9,), label="fy3x2"),
            "fzx2": Selection(indices=(14,), label="fzx2"),
            "fz3": Selection(indices=(12,), label="fz3"),
            "p": Selection(indices=range(1, 4), label="p"),
            "d": Selection(indices=range(4, 9), label="d"),
            "f": Selection(indices=range(9, 16), label="f"),
            "*": Selection(indices=range(len(spin_polarized.orbital_types))),
        },
        "spin": {
            "up": Selection(indices=(0,), label="up"),
            "down": Selection(indices=(1,), label="down"),
            "total": Selection(
                indices=range(spin_polarized.number_spins), label="total"
            ),
            "*": Selection(indices=range(spin_polarized.number_spins)),
        },
    }
    return Projectors(spin_polarized), ref


def test_selection(for_selection):
    proj, ref = for_selection
    default = Index(ref["atom"]["*"], ref["orbital"]["*"], ref["spin"]["*"])
    for atom, ref_atom in ref["atom"].items():
        assert proj.select(atom=atom) == default._replace(atom=ref_atom)
    for orbital, ref_orbital in ref["orbital"].items():
        assert proj.select(orbital=orbital) == default._replace(orbital=ref_orbital)
    for spin, ref_spin in ref["spin"].items():
        assert proj.select(spin=spin) == default._replace(spin=ref_spin)


@pytest.fixture
def for_parse_selection(without_spin):
    testcases = (
        SelectionTestCase(
            equivalent_formats=("Sr", "Sr(*)"),
            reference_selections=(Index(atom="Sr", orbital="*", spin="*"),),
        ),
        SelectionTestCase(
            equivalent_formats=("Ti(s,p)", "Ti (s)  Ti (p)"),
            reference_selections=(
                Index(atom="Ti", orbital="s", spin="*"),
                Index(atom="Ti", orbital="p", spin="*"),
            ),
        ),
        SelectionTestCase(
            equivalent_formats=("Ti 5", "Ti( * ), 5( * )"),
            reference_selections=(
                Index(atom="Ti", orbital="*", spin="*"),
                Index(atom="5", orbital="*", spin="*"),
            ),
        ),
        SelectionTestCase(
            equivalent_formats=("p, d", "*(p) *(d)"),
            reference_selections=(
                Index(atom="*", orbital="p", spin="*"),
                Index(atom="*", orbital="d", spin="*"),
            ),
        ),
        SelectionTestCase(
            equivalent_formats=("O(d), 1 s", "O(d), 1(*), *(s)"),
            reference_selections=(
                Index(atom="O", orbital="d", spin="*"),
                Index(atom="1", orbital="*", spin="*"),
                Index(atom="*", orbital="s", spin="*"),
            ),
        ),
        SelectionTestCase(
            equivalent_formats=("Sr(p)Ti(s)O(s)", "p(Sr) s(Ti, O)"),
            reference_selections=(
                Index(atom="Sr", orbital="p", spin="*"),
                Index(atom="Ti", orbital="s", spin="*"),
                Index(atom="O", orbital="s", spin="*"),
            ),
        ),
        SelectionTestCase(
            equivalent_formats=("1 - 4", "1-4", "  1  -  4 "),
            reference_selections=(Index(atom="1-4", orbital="*", spin="*"),),
        ),
    )
    return Projectors(without_spin), testcases


@pytest.fixture
def for_spin_polarized_parse_selection(spin_polarized):
    testcases = (
        SelectionTestCase(
            equivalent_formats=("Sr", "Sr(up,down)", "Sr(*(up)), Sr(down)"),
            reference_selections=(
                Index(atom="Sr", orbital="*", spin="up"),
                Index(atom="Sr", orbital="*", spin="down"),
            ),
        ),
        SelectionTestCase(
            equivalent_formats=("Ti( s(up) p(down) )", "Ti(s(up))Ti(p(down))"),
            reference_selections=(
                Index(atom="Ti", orbital="s", spin="up"),
                Index(atom="Ti", orbital="p", spin="down"),
            ),
        ),
        SelectionTestCase(
            equivalent_formats=("s p(up) d(total)", "s(up, down), *(p(up)), d(total)"),
            reference_selections=(
                Index(atom="*", orbital="s", spin="up"),
                Index(atom="*", orbital="s", spin="down"),
                Index(atom="*", orbital="p", spin="up"),
                Index(atom="*", orbital="d", spin="total"),
            ),
        ),
        SelectionTestCase(
            equivalent_formats=("up (s)  down (p, d)", "s(up) p(down) d(down)"),
            reference_selections=(
                Index(atom="*", orbital="s", spin="up"),
                Index(atom="*", orbital="p", spin="down"),
                Index(atom="*", orbital="d", spin="down"),
            ),
        ),
        SelectionTestCase(
            equivalent_formats=("2( px(up) )", "px(2(up))", "up(2(px))"),
            reference_selections=(Index(atom="2", orbital="px", spin="up"),),
        ),
        SelectionTestCase(
            equivalent_formats=("3-4(up)", "up (3 - 4)"),
            reference_selections=(Index(atom="3-4", orbital="*", spin="up"),),
        ),
    )
    return Projectors(spin_polarized), testcases


def test_parse_selection(for_parse_selection):
    run_parse_selection(for_parse_selection)


def test_spin_polarized_parse_selection(for_spin_polarized_parse_selection):
    run_parse_selection(for_spin_polarized_parse_selection)


def run_parse_selection(setup):
    proj, testcases = setup
    for testcase in testcases:
        for format in testcase.equivalent_formats:
            selections = proj.parse_selection(format)
            assert list(selections) == list(testcase.reference_selections)
