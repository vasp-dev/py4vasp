# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
from py4vasp._util.selection import SelectionTree


def test_one_level():
    selection = "foo bar baz"
    tree = SelectionTree.from_selection(selection)
    actual = [str(node) for node in tree.nodes]
    reference = selection.split()
    assert actual == reference


def test_multiple_level():
    selection = "foo(bar(baz))"
    tree = SelectionTree.from_selection(selection)
    level1 = tree.nodes[0]
    assert str(level1) == "foo"
    level2 = level1.nodes[0]
    assert str(level2) == "bar"
    level3 = level2.nodes[0]
    assert str(level3) == "baz"
    assert level3.nodes == []


def test_mixed_selection():
    selection = "foo(bar baz)"
    tree = SelectionTree.from_selection(selection)
    level1 = tree.nodes[0]
    assert str(level1) == "foo"
    assert [str(node) for node in level1.nodes] == ["bar", "baz"]


def test_comma_as_separator():
    selection = "foo, bar(1, 2)"
    tree = SelectionTree.from_selection(selection)
    assert str(tree.nodes[0]) == "foo"
    level1 = tree.nodes[1]
    assert str(level1) == "bar"
    assert str(level1.nodes[0]) == "1"
    assert str(level1.nodes[1]) == "2"


def test_excess_whitespace():
    selection = "  foo   (  bar,   baz  )"
    tree = SelectionTree.from_selection(selection)
    level1 = tree.nodes[0]
    assert str(level1) == "foo"
    assert len(level1.nodes) == 2
    assert str(level1.nodes[0]) == "bar"
    assert str(level1.nodes[1]) == "baz"


def test_no_whitespace():
    selection = "foo(bar)baz"
    tree = SelectionTree.from_selection(selection)
    level1 = tree.nodes[0]
    assert str(level1) == "foo"
    assert str(level1.nodes[0]) == "bar"
    assert str(tree.nodes[1]) == "baz"


def test_ranges():
    selection = "foo(1 : 3) 2 : 6 baz"
    tree = SelectionTree.from_selection(selection)
    level1 = tree.nodes[0]
    assert str(level1) == "foo"
    assert str(level1.nodes[0]) == "1:3"
    assert str(tree.nodes[1]) == "2:6"
    assert str(tree.nodes[2]) == "baz"


def test_pair_selection():
    selection = "foo  ~  bar, baz~foo"
    tree = SelectionTree.from_selection(selection)
    assert str(tree.nodes[0]) == "foo~bar"
    assert str(tree.nodes[1]) == "baz~foo"
