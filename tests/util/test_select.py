# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import pytest

from py4vasp import exception
from py4vasp._util import select


def test_empty_tree():
    assert selections(None) == ((),)
    assert graph(None) == "graph LR"


def test_one_level():
    selection = "foo bar baz"
    assert selections(selection) == (("foo",), ("bar",), ("baz",))
    expected = """graph LR
    foo
    bar
    baz"""
    assert graph(selection) == expected


def test_multiple_level():
    selection = "foo(bar(baz))"
    assert selections(selection) == (("foo", "bar", "baz"),)
    expected = """graph LR
    foo --> bar
    bar --> baz"""
    assert graph(selection) == expected


@pytest.mark.parametrize("selection", ["foo(bar baz)", "  foo   (  bar,   baz  )"])
def test_mixed_selection(selection):
    assert selections(selection) == (("foo", "bar"), ("foo", "baz"))
    expected = """graph LR
    foo --> bar
    foo --> baz"""
    assert graph(selection) == expected


def test_comma_as_separator():
    selection = "foo, bar(1, 2)"
    assert selections(selection) == (("foo",), ("bar", "1"), ("bar", "2"))
    expected = """graph LR
    foo
    bar --> 1
    bar --> 2"""
    assert graph(selection) == expected


def test_no_whitespace():
    selection = "foo(bar)baz"
    assert selections(selection) == (("foo", "bar"), ("baz",))
    expected = """graph LR
    foo --> bar
    baz"""
    assert graph(selection) == expected


def test_ranges():
    selection = "foo(1 : 3) 2 : 6 baz"
    range1 = select.Group(["1", "3"], ":")
    range2 = select.Group(["2", "6"], ":")
    assert selections(selection) == (("foo", range1), (range2,), ("baz",))
    expected = """graph LR
    foo --> 1:3
    2:6
    baz"""
    assert graph(selection) == expected


def test_pair_selection():
    selection = "foo  ~  bar, baz~foo"
    pair1 = select.Group(["foo", "bar"], "~")
    pair2 = select.Group(["baz", "foo"], "~")
    assert selections(selection) == ((pair1,), (pair2,))
    expected = """graph LR
    foo~bar
    baz~foo"""
    assert graph(selection) == expected


@pytest.mark.parametrize("selection", ["a + b, c - d", "a+b c-d"])
def test_addition_and_subtraction(selection):
    operation1 = select.Operation(selections("a")[0], "+", selections("b")[0])
    operation2 = select.Operation(selections("c")[0], "-", selections("d")[0])
    assert selections(selection) == ((operation1,), (operation2,))
    expected = """graph LR
    _0_[+] --> a
    _0_[+] --> b
    _1_[-] --> c
    _1_[-] --> d"""
    assert graph(selection) == expected


@pytest.mark.parametrize("selection", ["foo+bar-baz", "foo + bar - baz"])
def test_longer_equation(selection):
    operation = select.Operation(selections("foo")[0], "+", selections("bar - baz")[0])
    assert selections(selection) == ((operation,),)
    expected = """graph LR
    _0_[+] --> foo
    _0_[+] --> _1_[-]
    _1_[-] --> bar
    _1_[-] --> baz"""
    assert graph(selection) == expected


@pytest.mark.parametrize("selection", ["-a", "- a", "+a"])
def test_unary_operator(selection):
    operation = select.Operation((), selection[0], selections("a")[0])
    assert selections(selection) == ((operation,),)
    expected = f"""graph LR
    _0_[{selection[0]}] --> a"""
    assert graph(selection) == expected


@pytest.mark.parametrize("selection", ["a, -b", "a,-b"])
def test_unary_operator_after_split(selection):
    operation = select.Operation((), "-", selections("b")[0])
    assert selections(selection) == (("a",), (operation,))
    expected = """graph LR
    a
    _0_[-] --> b"""
    assert graph(selection) == expected


@pytest.mark.parametrize("selection", ["A(x + y)", "A ( x+y )"])
def test_operator_in_parenthesis(selection):
    operation = select.Operation(selections("x")[0], "+", selections("y")[0])
    assert selections(selection) == (("A", operation),)
    expected = """graph LR
    A --> _0_[+]
    _0_[+] --> x
    _0_[+] --> y"""
    assert graph(selection) == expected


def test_adding_two_parenthesis():
    selection = "A(x) - B(y)"
    operation = select.Operation(selections("A(x)")[0], "-", selections("B(y)")[0])
    assert selections(selection) == ((operation,),)
    expected = """graph LR
    _0_[-] --> A
    A --> x
    _0_[-] --> B
    B --> y"""
    assert graph(selection) == expected


def test_complex_tree():
    selection = "A(B(1:3), C~D(E F)) G(H, J) K"
    expected_selections = (
        ("A", "B", select.Group(["1", "3"], ":")),
        ("A", select.Group(["C", "D"], "~"), "E"),
        ("A", select.Group(["C", "D"], "~"), "F"),
        ("G", "H"),
        ("G", "J"),
        ("K",),
    )
    expected_graph = """graph LR
    A --> B
    B --> 1:3
    A --> C~D
    C~D --> E
    C~D --> F
    G --> H
    G --> J
    K"""
    assert selections(selection) == expected_selections
    assert graph(selection) == expected_graph


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


def selections(selection):
    tree = select.Tree.from_selection(selection)
    return tuple(tree.selections())


def graph(selection):
    tree = select.Tree.from_selection(selection)
    return tree.to_mermaid()
