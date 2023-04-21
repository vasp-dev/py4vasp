# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import pytest

from py4vasp import exception
from py4vasp._util import select


def test_empty_tree():
    tree = select.Tree.from_selection(None)
    assert len(tree) == 0
    assert tree.to_mermaid() == "graph LR"
    assert tree.nodes == []


def test_one_level():
    tree = select.Tree.from_selection("foo bar baz")
    actual = [str(node) for node in tree.nodes]
    assert len(tree) == 3
    assert actual == ["foo", "bar", "baz"]
    expected = """graph LR
    foo
    bar
    baz"""
    assert tree.to_mermaid() == expected


def test_multiple_level():
    tree = select.Tree.from_selection("foo(bar(baz))")
    assert len(tree) == 1
    expected = """graph LR
    foo --> bar
    bar --> baz"""
    assert tree.to_mermaid() == expected


def test_mixed_selection():
    tree = select.Tree.from_selection("foo(bar baz)")
    assert len(tree) == 2
    expected = """graph LR
    foo --> bar
    foo --> baz"""
    assert tree.to_mermaid() == expected


def test_comma_as_separator():
    tree = select.Tree.from_selection("foo, bar(1, 2)")
    assert len(tree) == 3
    expected = """graph LR
    foo
    bar --> 1
    bar --> 2"""
    assert tree.to_mermaid() == expected


def test_excess_whitespace():
    tree = select.Tree.from_selection("  foo   (  bar,   baz  )")
    assert len(tree) == 2
    expected = """graph LR
    foo --> bar
    foo --> baz"""
    assert tree.to_mermaid() == expected


def test_no_whitespace():
    tree = select.Tree.from_selection("foo(bar)baz")
    assert len(tree) == 2
    expected = """graph LR
    foo --> bar
    baz"""
    assert tree.to_mermaid() == expected


def test_ranges():
    tree = select.Tree.from_selection("foo(1 : 3) 2 : 6 baz")
    assert len(tree) == 3
    expected = """graph LR
    foo --> 1:3
    2:6
    baz"""
    assert tree.to_mermaid() == expected


def test_pair_selection():
    tree = select.Tree.from_selection("foo  ~  bar, baz~foo")
    assert len(tree) == 2
    expected = """graph LR
    foo~bar
    baz~foo"""
    assert tree.to_mermaid() == expected


@pytest.mark.parametrize("selection", ["a + b, c - d", "a+b c-d"])
def test_addition_and_subtraction(selection):
    tree = select.Tree.from_selection(selection)
    assert not tree.is_operation
    assert len(tree) == 2
    expected = """graph LR
    _0_[+] --> a
    _0_[+] --> b
    _1_[-] --> c
    _1_[-] --> d"""
    assert tree.to_mermaid() == expected


@pytest.mark.parametrize("selection", ["foo+bar-baz", "foo + bar - baz"])
def test_longer_equation(selection):
    tree = select.Tree.from_selection(selection)
    assert len(tree) == 1
    expected = """graph LR
    _0_[+] --> foo
    _0_[+] --> _1_[-]
    _1_[-] --> bar
    _1_[-] --> baz"""
    assert tree.to_mermaid() == expected


@pytest.mark.parametrize("selection", ["-a", "- a", "+a"])
def test_unary_operator(selection):
    tree = select.Tree.from_selection(selection)
    assert len(tree) == 1
    expected = f"""graph LR
    _0_[{selection[0]}] --> a"""
    assert tree.to_mermaid() == expected


@pytest.mark.parametrize("selection", ["a, -b", "a,-b"])
def test_unary_operator_after_split(selection):
    tree = select.Tree.from_selection(selection)
    assert len(tree) == 2
    expected = """graph LR
    a
    _0_[-] --> b"""
    assert tree.to_mermaid() == expected


def test_operator_and_parenthesis():
    tree = select.Tree.from_selection("A(x + y)")
    assert len(tree) == 1
    expected = """graph LR
    A --> _0_[+]
    _0_[+] --> x
    _0_[+] --> y"""
    assert tree.to_mermaid() == expected


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


@pytest.mark.parametrize("selection", [":1", "1:", "1:,2", "a~", "a~,b", "~a"])
def test_broken_group_raises_error(selection):
    with pytest.raises(exception.IncorrectUsage):
        select.Tree.from_selection(selection)


@pytest.mark.parametrize("selection", ["(", "A(", "A,(", ")", "A)"])
def test_broken_parenthesis(selection):
    with pytest.raises(exception.IncorrectUsage):
        select.Tree.from_selection(selection)


@pytest.mark.parametrize("selection", ["a+", "b-", "a-,b"])
def test_missing_operand(selection):
    with pytest.raises(exception.IncorrectUsage):
        select.Tree.from_selection(selection)
