from py4vasp.data import Projectors, Topology
from py4vasp.raw import RawProjectors, RawTopology, RawVersion
from py4vasp.data._selection import Selection
import py4vasp.exceptions as exception
import pytest
import numpy as np
from typing import NamedTuple, Iterable

Index = Projectors.Index


class SelectionTestCase(NamedTuple):
    equivalent_formats: Iterable[str]
    reference_selections: Iterable[Index]


@pytest.fixture
def without_spin():
    proj = RawProjectors(
        topology=RawTopology(
            number_ion_types=np.array((2, 1, 4)),
            ion_types=np.array(("Sr", "Ti", "O "), dtype="S"),
        ),
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
    index = np.cumsum(spin_polarized.topology.number_ion_types)
    ref = {
        "atom": {
            "Sr": Selection(indices=slice(0, index[0]), label="Sr"),
            "Ti": Selection(indices=slice(index[0], index[1]), label="Ti"),
            "O": Selection(indices=slice(index[1], index[2]), label="O"),
            "1": Selection(indices=slice(0, 1), label="Sr_1"),
            "2": Selection(indices=slice(1, 2), label="Sr_2"),
            "3": Selection(indices=slice(2, 3), label="Ti_1"),
            "4": Selection(indices=slice(3, 4), label="O_1"),
            "5": Selection(indices=slice(4, 5), label="O_2"),
            "6": Selection(indices=slice(5, 6), label="O_3"),
            "7": Selection(indices=slice(6, 7), label="O_4"),
            "1-3": Selection(indices=slice(0, 3), label="1-3"),
            "4-7": Selection(indices=slice(3, 7), label="4-7"),
            "*": Selection(indices=slice(index[-1])),
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
            "x2-y2": Selection(indices=slice(8, 9), label="x2-y2"),
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
            "*": Selection(indices=slice(len(spin_polarized.orbital_types))),
        },
        "spin": {
            "up": Selection(indices=slice(1), label="up"),
            "down": Selection(indices=slice(1, 2), label="down"),
            "total": Selection(
                indices=slice(spin_polarized.number_spins), label="total"
            ),
            "*": Selection(indices=slice(spin_polarized.number_spins)),
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


def test_read(without_spin, Assert):
    projectors = Projectors(without_spin)
    assert projectors.read() == {}
    reference = {
        "Sr_p": (slice(1), slice(0, 2), slice(1, 4)),
        "Ti_1_dxy": (slice(1), slice(2, 3), slice(4, 5)),
    }
    assert projectors.read(selection="Sr(p) 3(dxy)") == reference
    num_atoms = np.sum(without_spin.topology.number_ion_types)
    num_orbitals = len(without_spin.orbital_types)
    num_quantity = 25
    shape = (without_spin.number_spins, num_atoms, num_orbitals, num_quantity)
    projections = np.arange(np.prod(shape)).reshape(shape)
    Sr_ref = np.sum(projections[0, 0:2, 1:4], axis=(0, 1))
    Ti_ref = projections[0, 2, 4]
    actual = projectors.read(selection="Sr(p) 3(dxy)", projections=projections)
    Assert.allclose(actual["Sr_p"], Sr_ref)
    Assert.allclose(actual["Ti_1_dxy"], Ti_ref)


def test_print(without_spin, format_):
    actual, _ = format_(Projectors(without_spin))
    reference = """
projectors:
    atoms: Sr, Ti, O
    orbitals: s, py, pz, px, dxy, dyz, dz2, dxz, x2-y2, fy3x2, fxyz, fyz2, fz3, fxz2, fzx2, fx3
    """.strip()
    assert actual == {"text/plain": reference}


def test_error_parsing(without_spin):
    projectors = Projectors(without_spin)
    with pytest.raises(exception.IncorrectUsage):
        projectors.read(selection="XX")
    with pytest.raises(exception.IncorrectUsage):
        number_instead_of_string = -1
        projectors.read(selection=number_instead_of_string)


def test_incorrect_selection(without_spin):
    projectors = Projectors(without_spin)
    with pytest.raises(exception.IncorrectUsage):
        projectors.select(atom="XX")
    with pytest.raises(exception.IncorrectUsage):
        projectors.select(atom="100-900")
    with pytest.raises(exception.IncorrectUsage):
        projectors.select(orbital="XX")
    with pytest.raises(exception.IncorrectUsage):
        projectors.select(spin="XX")


def test_nonexisting_projectors():
    with pytest.raises(exception.NoData):
        projectors = Projectors(None).read()


def test_incorrect_reading_of_projections(without_spin):
    projectors = Projectors(without_spin)
    with pytest.raises(exception.IncorrectUsage):
        projectors.read("Sr", [1, 2, 3])
    with pytest.raises(exception.IncorrectUsage):
        projectors.read("Sr", np.zeros(3))


def test_descriptor(without_spin, check_descriptors):
    projectors = Projectors(without_spin)
    descriptors = {
        "_to_dict": ["to_dict", "read"],
        "_select": ["select"],
        "_parse_selection": ["parse_selection"],
    }
    check_descriptors(projectors, descriptors)
