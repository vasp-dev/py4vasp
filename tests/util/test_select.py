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
    operation1 = select.Operation(("a",), "+", ("b",))
    operation2 = select.Operation(("c",), "-", ("d",))
    assert selections(selection) == ((operation1,), (operation2,))
    expected = """graph LR
    _0_[+] --> a
    _0_[+] --> b
    _1_[-] --> c
    _1_[-] --> d"""
    assert graph(selection) == expected


@pytest.mark.parametrize("selection", ["foo+bar-baz", "foo + bar - baz"])
def test_longer_equation(selection):
    operation = select.Operation(("foo",), "+", selections("bar - baz")[0])
    assert selections(selection) == ((operation,),)
    expected = """graph LR
    _0_[+] --> foo
    _0_[+] --> _1_[-]
    _1_[-] --> bar
    _1_[-] --> baz"""
    assert graph(selection) == expected


@pytest.mark.parametrize("selection", ["-a", "- a", "+a"])
def test_unary_operator(selection):
    operation = select.Operation((), selection[0], ("a",))
    assert selections(selection) == ((operation,),)
    expected = f"""graph LR
    _0_[{selection[0]}] --> a"""
    assert graph(selection) == expected


@pytest.mark.parametrize("selection", ["a, -b", "a,-b"])
def test_unary_operator_after_split(selection):
    operation = select.Operation((), "-", ("b",))
    assert selections(selection) == (("a",), (operation,))
    expected = """graph LR
    a
    _0_[-] --> b"""
    assert graph(selection) == expected


@pytest.mark.parametrize("selection", ["A(x + y)", "A ( x+y )"])
def test_operator_in_parenthesis(selection):
    operation = select.Operation(("x",), "+", ("y",))
    assert selections(selection) == (("A", operation),)
    expected = """graph LR
    A --> _0_[+]
    _0_[+] --> x
    _0_[+] --> y"""
    assert graph(selection) == expected


def test_adding_two_parenthesis():
    selection = "A(x) - B(y)"
    operation = select.Operation(("A", "x"), "-", ("B", "y"))
    assert selections(selection) == ((operation,),)
    expected = """graph LR
    _0_[-] --> A
    A --> x
    _0_[-] --> B
    B --> y"""
    assert graph(selection) == expected


def test_nested_operations():
    selection = "A(x y) + B(u v)"
    Ax_Bu = select.Operation(("A", "x"), "+", ("B", "u"))
    Ax_Bv = select.Operation(("A", "x"), "+", ("B", "v"))
    Ay_Bu = select.Operation(("A", "y"), "+", ("B", "u"))
    Ay_Bv = select.Operation(("A", "y"), "+", ("B", "v"))
    assert selections(selection) == ((Ax_Bu,), (Ax_Bv,), (Ay_Bu,), (Ay_Bv,))
    expected = """graph LR
    _0_[+] --> A
    A --> x
    A --> y
    _0_[+] --> B
    B --> u
    B --> v"""
    assert graph(selection) == expected


def test_complex_nesting():
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


def test_complex_operation():
    selection = "A(x + y(z)) + B(1:3 u - v) - C~D"
    first_operand = selections("A(x+y(z))")[0]
    second_operands = selections("B(1:3 u-v)")
    third_operand = selections("C~D")[0]
    inner_operation1 = select.Operation(second_operands[0], "-", third_operand)
    inner_operation2 = select.Operation(second_operands[1], "-", third_operand)
    outer_operation1 = select.Operation(first_operand, "+", (inner_operation1,))
    outer_operation2 = select.Operation(first_operand, "+", (inner_operation2,))
    assert selections(selection) == ((outer_operation1,), (outer_operation2,))
    expected = """graph LR
    _1_[+] --> A
    A --> _0_[+]
    _0_[+] --> x
    _0_[+] --> y
    y --> z
    _1_[+] --> _3_[-]
    _3_[-] --> B
    B --> 1:3
    B --> _2_[-]
    _2_[-] --> u
    _2_[-] --> v
    _3_[-] --> C~D"""
    assert graph(selection) == expected


@pytest.mark.parametrize(
    "input, output",
    [
        (" ", ""),
        ("A(B(1:3), C~D(E F)) G(H, J)", "A(B(1:3)), A(C~D(E)), A(C~D(F)), G(H), G(J)"),
        (
            "A(x+y(z)) + B(1:3 u-v) - C~D",
            "A(x + y(z)) + B(1:3) - C~D, A(x + y(z)) + B(u - v) - C~D",
        ),
    ],
)
def test_selections_to_string(input, output):
    assert select.selections_to_string(selections(input)) == output
    assert selections(input) == selections(output)


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


@pytest.mark.parametrize("selection", [None, "string"])
def test_default_constructor_raises_error(selection):
    with pytest.raises(exception._Py4VaspInternalError):
        select.Tree(selection)


def selections(selection):
    tree = select.Tree.from_selection(selection)
    return tuple(tree.selections())


def graph(selection):
    tree = select.Tree.from_selection(selection)
    return tree.to_mermaid()
