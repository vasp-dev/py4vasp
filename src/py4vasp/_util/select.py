# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
""" Parse a string to a Tree of selections.

In many cases, a user may want to select a certain subset of quantities to be refined.
Examples include selecting the projected DOS or a particular component of the energy.
To give the user a consistent experience, all routines that provide this functionality
should use this Tree to understand the user input. A typical use-case will look like
this

>>> from py4vasp._util import select
>>> def parse_user_selection(selection):
>>>     tree = select.Tree.from_selection(selection)
>>>     for selection in tree.selections():
>>>         ...     # code to analyze the selection

Each individual selection will be a tuple to allow for nested selections. In the
simplest case where you only want to provide dictionary-like access, you could use the
first component of this tuple.

Features
--------
lists
    A space or a comma separates elements in the selection. A string "a, b c" would
    result in three selections "a", "b", and "c".

nesting
    Occasionally, you want to select from multiple indices e.g. the d DOS of Ti. This
    selection would be achieved by "Ti(d)" which would result in a single selection
    represented by the tuple '("Ti", "d")'. Note that the order of the selection should
    not matter in general, so "d(Ti)" should yield the same result as "Ti(d)".

groups
    Sometimes it can be useful to group multiple elements together. Currently, it is
    supported to connect two elements with a colon to form a range ("1:3") or with
    a tilde to form a pair ("A~B").

operations
    Adding or subtracting two quantities is supported. Currently this will interpret
    the operations left to right, so multiplication and division is not available to
    make sure the equation is correct.

For the common case, where you want to use the user selection to specify the index of
an array, the `py4vasp._util.index.Selector` defines an interface compatible with the
features of the Tree.
"""
import dataclasses
import itertools

from py4vasp import exception
from py4vasp._util import check

range_separator = ":"
pair_separator = "~"
group_separators = (range_separator, pair_separator)
operators = ("+", "-")
all = "__all__"
end_of_text = chr(3)


class Tree:
    "Contains the whole tree or a subsection of the tree parsed from the user input."

    def __init__(self, parent=None, *, _internal=False):
        "For internal use, you should not call this. Use `from_selection` instead."
        _raise_error_if_not_internal_call(_internal)
        self._new_selection = True
        self._space_parsed = False
        self._is_operation = False
        self._parent = parent
        self._children = []
        self._content = ""
        if not parent:
            self._counter = itertools.count()

    @classmethod
    def from_selection(cls, selection):
        """Parse the user selection into a Tree object.

        Parameters
        ----------
        selection : str
            User provided string defining some selected quantities.
        """
        tree = cls(_internal=True)
        selection = selection or ""
        message = f"Selection must be a string. The passed argument {selection} is not allowed."
        check.raise_error_if_not_string(selection, message)
        tree._parse_selection_character_by_character(selection)
        return tree

    @property
    def nodes(self):
        return self._children

    def __str__(self):
        return str(self._content)

    def selections(self, selected=(), filter={}):
        """Core routine generating all user selections parsed.

        This will generate one selection at a time so it should be used in a loop or
        converted to a list.

        Parameters
        ----------
        selected : tuple
            Prior selections obtained from a different source. These selections will
            be added to any additional selection parsed from the user input. If not
            set, if defaults to giving just the user selections.
        filter : set
            Remove any element found in the set from the resulting selection.

        Yields
        ------
        tuple
            Each selection corresponds to one path from the root of the tree to one of
            its leaves.
        """
        if self._content and self._content not in filter:
            content = (self._content,)
        else:
            content = ()
        if not self._children:
            yield selected + content
        elif self._is_operation:
            yield from self._operation_selections(selected, filter)
        else:
            for child in self._children:
                yield from child.selections(selected + content, filter)

    def _operation_selections(self, selected, filter):
        left_operands = self._get_operands(self._children[0], filter)
        right_operands = self._get_operands(self._children[1], filter)
        for left_op, right_op in itertools.product(left_operands, right_operands):
            yield *selected, Operation(left_op, self._content.operator, right_op)

    def _get_operands(self, child, filter):
        for operand in child.selections(filter=filter):
            child._raise_error_if_content_and_operator_are_incompatible(operand)
            yield operand

    def to_mermaid(self):
        "Helper routine to visualize the Tree using Mermaid"
        return "\n".join(self._to_mermaid(root=True))

    def _to_mermaid(self, root=False):
        if root:
            yield "graph LR"
        if self._content:
            if str(self._parent):
                yield f"    {self._parent} --> {self}"
            elif not self._children:
                yield f"    {self}"
        for child in self._children:
            yield from child._to_mermaid()

    def _parse_selection_character_by_character(self, selection):
        active_node = self
        try:
            for ii, character in enumerate(selection):
                active_node = active_node._parse_character(character)
            active_node._parse_character(end_of_text)
        except exception._Py4VaspInternalError as error:
            _raise_error_if_parsing_failed(error, selection, ii)

    def _parse_character(self, character):
        if character == ",":
            return self._parse_new_selection()
        elif character == " ":
            return self._parse_space()
        elif character in group_separators:
            return self._parse_group(character)
        elif character == "(":
            return self._parse_open_parenthesis()
        elif character == ")":
            return self._parse_close_parenthesis()
        elif character in operators:
            return self._parse_operator(character)
        elif character == end_of_text:
            return self._parse_end_of_text()
        else:
            return self._store_character_in_tree(character)

    def _parse_new_selection(self):
        self._new_selection = True
        self._raise_error_if_group_misses_right_hand_side()
        self._raise_error_if_operation_misses_right_hand_side()
        return self._finalize_operation()

    def _finalize_operation(self):
        if self._new_child_needed() and self._is_operation and len(self._children) == 2:
            return self._parent._parse_new_selection()
        return self

    def _parse_space(self):
        self._space_parsed = True
        return self

    def _parse_group(self, separator):
        self._raise_error_if_group_misses_left_hand_side(separator)
        self._ignore_space = True
        self._children[-1]._transform_to_group(separator)
        return self

    def _transform_to_group(self, separator):
        self._content = Group([self._content, ""], separator)

    def _parse_open_parenthesis(self):
        self._raise_error_if_opening_parenthesis_without_argument()
        return self._children[-1]

    def _parse_close_parenthesis(self):
        self._new_selection = True
        self._raise_error_if_superfluous_closing_parenthesis()
        node = self._finalize_operation()
        return node._parent._parse_space()

    def _parse_operator(self, operator):
        self._add_child_if_needed(ignore_space=True)
        self._children[-1]._transform_to_operation(operator)
        return self._children[-1]

    def _transform_to_operation(self, operator):
        self._is_operation = True
        node = Tree(self, _internal=True)
        node._content = self._content
        node._children = self._children
        for child in node._children:
            child._parent = node
        self._content = _Operator(operator, self._next_id())
        self._children = [node]

    def _next_id(self):
        if self._parent:
            return self._parent._next_id()
        else:
            return next(self._counter)

    def _parse_end_of_text(self):
        self._raise_error_if_closing_parenthesis_missing()
        self._raise_error_if_group_misses_right_hand_side()
        self._raise_error_if_operation_misses_right_hand_side()
        return self

    def _store_character_in_tree(self, character):
        node = self._finalize_operation()
        ignore_space = node._is_operation or node._child_is_open_group()
        node._add_child_if_needed(ignore_space)
        node._space_parsed = False
        node._children[-1]._content += character
        return node

    def _child_is_open_group(self):
        if len(self._children) == 0:
            return False
        content = self._children[-1]._content
        return isinstance(content, Group) and not content.group[1]

    def _add_child_if_needed(self, ignore_space):
        if not self._new_child_needed(ignore_space):
            return
        self._children.append(Tree(self, _internal=True))
        self._new_selection = False

    def _new_child_needed(self, ignore_space=False):
        return self._new_selection or (self._space_parsed and not ignore_space)

    def _raise_error_if_group_misses_left_hand_side(self, separator):
        if len(self._children) > 0:
            return
        self._raise_group_error_message("left", separator)

    def _raise_error_if_group_misses_right_hand_side(self):
        if len(self._children) == 0:
            return
        content = self._children[-1]._content
        if not isinstance(content, Group) or content.group[1]:
            return
        self._raise_group_error_message("right", content.separator)

    def _raise_group_error_message(self, missing_side, separator):
        group = "range" if separator == range_separator else "pair"
        message = f"The {missing_side} argument of {group} is missing."
        raise exception._Py4VaspInternalError(message)

    def _raise_error_if_opening_parenthesis_without_argument(self):
        if len(self._children) > 0:
            return
        message = "Opening parenthesis '(' must relate to a previous argument."
        raise exception._Py4VaspInternalError(message)

    def _raise_error_if_superfluous_closing_parenthesis(self):
        if self._parent:
            return
        message = "Closing parenthesis ')' must follow an opening one."
        raise exception._Py4VaspInternalError(message)

    def _raise_error_if_closing_parenthesis_missing(self):
        if not self._parent or self._is_operation:
            return
        message = "An opening parenthesis was not followed by a closing one."
        raise exception._Py4VaspInternalError(message)

    def _raise_error_if_operation_misses_right_hand_side(self):
        if not self._is_operation or len(self._children) == 2:
            return
        message = f"The operator {self._content} is not followed by an element."
        raise exception._Py4VaspInternalError(message)

    def _raise_error_if_content_and_operator_are_incompatible(self, operand):
        if bool(self._content) == bool(operand):
            return
        message = f"The operand `{operand}` has a qualitatively different behavior then the content `{self}`. This may occur when a filter would replace the last element."
        raise exception.IncorrectUsage(message)


@dataclasses.dataclass
class Group:
    "A user selection where multiple elements should be treated together."
    group: list
    "The individual members of the group."
    separator: str
    "The string separating the members of the group."
    __str__ = lambda self: self.separator.join(self.group)
    __hash__ = lambda self: hash(str(self))

    def __iadd__(self, character):
        self.group[-1] += character
        return self


@dataclasses.dataclass
class _Operator:
    operator: str
    _id: int
    __str__ = lambda self: f"_{self._id}_[{self.operator}]"
    __hash__ = lambda self: hash(self.operator)


@dataclasses.dataclass
class Operation:
    "A mathematical operation like addition and subtraction."
    left_operand: tuple
    "The selection on the left-hand side of the operation."
    operator: str
    "A character identifying the operation."
    right_operand: tuple
    "The selection on the right-hand side of the operation."

    def __str__(self):
        left_op = _selection_to_string(self.left_operand)
        right_op = _selection_to_string(self.right_operand)
        return f"{left_op} {self.operator} {right_op}"

    def unary(self):
        return self.left_operand == ()


def _raise_error_if_parsing_failed(error, selection, ii):
    message = f"""Error when parsing the selection string
  {selection}
  {" " * ii}^
{error}"""
    raise exception.IncorrectUsage(message)


def _raise_error_if_not_internal_call(internal):
    if internal:
        return
    message = """
Tree was initialized using the default constructor; this is most likely not what you
want. In most cases, you should use `Tree.from_selection` to initialize instead. The
default constructor is used internally to construct the leaves of the Tree. In the rare
case, where you do want to use the internal constructor, please pass the keyword
argument `_internal = True`."""
    raise exception._Py4VaspInternalError(message)


def selections_to_string(selections):
    "This routine is intended to convert selections back to string that would regenerate a tree."
    return ", ".join(_selection_to_string(selection) for selection in selections)


def _selection_to_string(selection):
    parts = [str(part) for part in selection]
    return "(".join(parts) + ")" * (len(parts) - 1)


def contains(selection, choice, ignore_case=False):
    return any(_part_contains(part, choice, ignore_case) for part in selection)


def _part_contains(part, choice, ignore_case):
    if isinstance(part, Group):
        return _choice_in_group(part.group, choice, ignore_case)
    if isinstance(part, Operation):
        return _choice_in_operation(part, choice, ignore_case)
    return _part_is_choice(part, choice, ignore_case)


def _choice_in_group(group, choice, ignore_case):
    if ignore_case:
        return any(choice.lower() == element.lower() for element in group)
    else:
        return choice in group


def _choice_in_operation(part, choice, ignore_case):
    in_left_op = contains(part.left_operand, choice, ignore_case)
    in_right_op = contains(part.right_operand, choice, ignore_case)
    return in_left_op or in_right_op


def _part_is_choice(part, choice, ignore_case):
    if ignore_case:
        return part.lower() == choice.lower()
    else:
        return part == choice
