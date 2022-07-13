# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
from py4vasp._util import selection


def test_empty_tree():
    tree = selection.Tree.from_selection(None)
    assert tree.nodes == []


def test_one_level():
    tree = selection.Tree.from_selection("foo bar baz")
    actual = [str(node) for node in tree.nodes]
    assert actual == ["foo", "bar", "baz"]


def test_multiple_level():
    tree = selection.Tree.from_selection("foo(bar(baz))")
    level1 = tree.nodes[0]
    assert str(level1) == "foo"
    level2 = level1.nodes[0]
    assert str(level2) == "bar"
    level3 = level2.nodes[0]
    assert str(level3) == "baz"
    assert level3.nodes == []


def test_mixed_selection():
    tree = selection.Tree.from_selection("foo(bar baz)")
    level1 = tree.nodes[0]
    assert str(level1) == "foo"
    assert [str(node) for node in level1.nodes] == ["bar", "baz"]


def test_comma_as_separator():
    tree = selection.Tree.from_selection("foo, bar(1, 2)")
    assert str(tree.nodes[0]) == "foo"
    level1 = tree.nodes[1]
    assert str(level1) == "bar"
    assert str(level1.nodes[0]) == "1"
    assert str(level1.nodes[1]) == "2"


def test_excess_whitespace():
    tree = selection.Tree.from_selection("  foo   (  bar,   baz  )")
    level1 = tree.nodes[0]
    assert str(level1) == "foo"
    assert len(level1.nodes) == 2
    assert str(level1.nodes[0]) == "bar"
    assert str(level1.nodes[1]) == "baz"


def test_no_whitespace():
    tree = selection.Tree.from_selection("foo(bar)baz")
    level1 = tree.nodes[0]
    assert str(level1) == "foo"
    assert str(level1.nodes[0]) == "bar"
    assert str(tree.nodes[1]) == "baz"


def test_ranges():
    tree = selection.Tree.from_selection("foo(1 : 3) 2 : 6 baz")
    level1 = tree.nodes[0]
    assert str(level1) == "foo"
    assert str(level1.nodes[0]) == "1:3"
    assert level1.nodes[0].content == selection.Group(["1", "3"], separator=":")
    assert str(tree.nodes[1]) == "2:6"
    assert tree.nodes[1].content == selection.Group(["2", "6"], separator=":")
    assert str(tree.nodes[2]) == "baz"


def test_pair_selection():
    tree = selection.Tree.from_selection("foo  ~  bar, baz~foo")
    assert str(tree.nodes[0]) == "foo~bar"
    assert tree.nodes[0].content == selection.Group(["foo", "bar"], separator="~")
    assert str(tree.nodes[1]) == "baz~foo"
    assert tree.nodes[1].content == selection.Group(["baz", "foo"], separator="~")
