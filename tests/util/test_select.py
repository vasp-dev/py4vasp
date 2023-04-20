# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import pytest

from py4vasp import exception
from py4vasp._util import select


def test_empty_tree():
    tree = select.Tree.from_selection(None)
    assert len(tree) == 0
    assert tree.nodes == []


def test_one_level():
    tree = select.Tree.from_selection("foo bar baz")
    actual = [str(node) for node in tree.nodes]
    assert len(tree) == 3
    assert actual == ["foo", "bar", "baz"]


def test_multiple_level():
    tree = select.Tree.from_selection("foo(bar(baz))")
    assert len(tree) == 1
    level1 = tree.nodes[0]
    assert str(level1) == "foo"
    level2 = level1.nodes[0]
    assert str(level2) == "bar"
    level3 = level2.nodes[0]
    assert str(level3) == "baz"
    assert level3.nodes == []


def test_mixed_selection():
    tree = select.Tree.from_selection("foo(bar baz)")
    assert len(tree) == 2
    level1 = tree.nodes[0]
    assert str(level1) == "foo"
    assert [str(node) for node in level1.nodes] == ["bar", "baz"]


def test_comma_as_separator():
    tree = select.Tree.from_selection("foo, bar(1, 2)")
    assert len(tree) == 3
    assert str(tree.nodes[0]) == "foo"
    level1 = tree.nodes[1]
    assert str(level1) == "bar"
    assert str(level1.nodes[0]) == "1"
    assert str(level1.nodes[1]) == "2"


def test_excess_whitespace():
    tree = select.Tree.from_selection("  foo   (  bar,   baz  )")
    assert len(tree) == 2
    level1 = tree.nodes[0]
    assert str(level1) == "foo"
    assert len(level1.nodes) == 2
    assert str(level1.nodes[0]) == "bar"
    assert str(level1.nodes[1]) == "baz"


def test_no_whitespace():
    tree = select.Tree.from_selection("foo(bar)baz")
    assert len(tree) == 2
    level1 = tree.nodes[0]
    assert str(level1) == "foo"
    assert str(level1.nodes[0]) == "bar"
    assert str(tree.nodes[1]) == "baz"


def test_ranges():
    tree = select.Tree.from_selection("foo(1 : 3) 2 : 6 baz")
    assert len(tree) == 3
    level1 = tree.nodes[0]
    assert str(level1) == "foo"
    assert str(level1.nodes[0]) == "1:3"
    assert level1.nodes[0].content == select.Group(["1", "3"], separator=":")
    assert str(tree.nodes[1]) == "2:6"
    assert tree.nodes[1].content == select.Group(["2", "6"], separator=":")
    assert str(tree.nodes[2]) == "baz"


def test_pair_selection():
    tree = select.Tree.from_selection("foo  ~  bar, baz~foo")
    assert len(tree) == 2
    assert str(tree.nodes[0]) == "foo~bar"
    assert tree.nodes[0].content == select.Group(["foo", "bar"], separator="~")
    assert str(tree.nodes[1]) == "baz~foo"
    assert tree.nodes[1].content == select.Group(["baz", "foo"], separator="~")


def test_addition_and_subtraction():
    tree = select.Tree.from_selection("a  +  b, c-d")
    assert not tree.is_operation
    assert len(tree) == 2
    level1 = tree.nodes[0]
    assert len(level1) == 1
    assert str(level1) == "+"
    assert level1.is_operation
    assert str(level1.nodes[0]) == "a"
    assert str(level1.nodes[1]) == "b"
    level1 = tree.nodes[1]
    assert len(level1) == 1
    assert str(level1) == "-"
    assert level1.is_operation
    assert str(level1.nodes[0]) == "c"
    assert str(level1.nodes[1]) == "d"


def test_selections_simple_tree():
    tree = select.Tree.from_selection("foo")
    assert len(tree) == 1
    assert list(tree.selections()) == [("foo",)]


def test_selections_complex_tree():
    tree = select.Tree.from_selection("A(B(1:3), C~D(E F)) G(H, J) K")
    expected = [
        ("A", "B", select.Group(["1", "3"], ":")),
        ("A", select.Group(["C", "D"], "~"), "E"),
        ("A", select.Group(["C", "D"], "~"), "F"),
        ("G", "H"),
        ("G", "J"),
        ("K",),
    ]
    assert len(tree) == len(expected)
    assert list(tree.selections()) == expected


def test_selections_empty_tree():
    tree = select.Tree.from_selection(None)
    assert list(tree.selections()) == [()]


def test_selections_to_string():
    tree = select.Tree.from_selection("A(B(1:3), C~D(E F)) G(H, J) K")
    expected = "A(B(1:3)), A(C~D(E)), A(C~D(F)), G(H), G(J), K"
    assert select.selections_to_string(tree.selections()) == expected
    copy = select.Tree.from_selection(expected)
    assert list(tree.selections()) == list(copy.selections())


def test_incorrect_selection_raises_error():
    with pytest.raises(exception.IncorrectUsage):
        select.Tree.from_selection(1)
