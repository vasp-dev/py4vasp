# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
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


class SelectionParserError(Exception):
    """This is created when something goes wrong parsing a character"""


@dataclasses.dataclass
class Group:
    group: list
    separator: str
    __str__ = lambda self: self.separator.join(self.group)

    def __iadd__(self, character):
        self.group[-1] += character
        return self


@dataclasses.dataclass
class _Operator:
    operator: str
    _id: int
    __str__ = lambda self: f"_{self._id}_[{self.operator}]"


@dataclasses.dataclass
class Operation:
    left_operand: str
    operator: str
    right_operand: str

    def __str__(self):
        parts = [
            selections_to_string(self.left_operand),
            self.operator,
            selections_to_string(self.right_operand),
        ]
        return " ".join(parts)


class Tree:
    def __init__(self, parent=None):
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
        tree = cls()
        selection = selection or ""
        message = f"Selection must be a string. The passed argument {selection} is not allowed."
        check.raise_error_if_not_string(selection, message)
        _parse_selection_character_by_character(tree, selection)
        return tree

    @property
    def nodes(self):
        return self._children

    @property
    def content(self):
        return self._content

    @property
    def is_operation(self):
        return self._is_operation

    def __str__(self):
        return str(self._content)

    def selections(self):
        content = (self._content,) if self._content else ()
        if len(self._children) == 0:
            yield content
        elif self._is_operation:
            left_operand = tuple(self._children[0].selections())
            operator = self._content.operator
            right_operand = tuple(self._children[1].selections())
            yield (Operation(left_operand, operator, right_operand),)
        else:
            for child in self._children:
                for selection in child.selections():
                    yield content + selection

    def _empty_tree(self):
        return self._parent is None and not self._children

    def to_mermaid(self):
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

    def parse_character(self, character):
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
        node = Tree(self)
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
        self._children.append(Tree(self))
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
        content = self._children[-1].content
        if not isinstance(content, Group) or content.group[1]:
            return
        self._raise_group_error_message("right", content.separator)

    def _raise_group_error_message(self, missing_side, separator):
        group = "range" if separator == range_separator else "pair"
        message = f"The {missing_side} argument of {group} is missing."
        raise SelectionParserError(message)

    def _raise_error_if_opening_parenthesis_without_argument(self):
        if len(self._children) > 0:
            return
        message = "Opening parenthesis '(' must relate to a previous argument."
        raise SelectionParserError(message)

    def _raise_error_if_superfluous_closing_parenthesis(self):
        if self._parent:
            return
        message = "Closing parenthesis ')' must follow an opening one."
        raise SelectionParserError(message)

    def _raise_error_if_closing_parenthesis_missing(self):
        if not self._parent or self._is_operation:
            return
        message = "An opening parenthesis was not followed by a closing one."
        raise SelectionParserError(message)

    def _raise_error_if_operation_misses_right_hand_side(self):
        if not self._is_operation or len(self._children) == 2:
            return
        message = f"The operator {self._content} is not followed by an element."
        raise SelectionParserError(message)


def _parse_selection_character_by_character(tree, selection):
    active_node = tree
    try:
        for ii, character in enumerate(selection):
            active_node = active_node.parse_character(character)
        active_node.parse_character(end_of_text)
    except SelectionParserError as error:
        _raise_error_if_parsing_failed(error, selection, ii)


def _raise_error_if_parsing_failed(error, selection, ii):
    message = f"""Error when parsing the selection string
  {selection}
  {" " * ii}^
{error}"""
    raise exception.IncorrectUsage(message)


def selections_to_string(selections):
    "This routine is intended to convert selections back to string that would regenerate a tree."
    return ", ".join(_selection_to_string(selection) for selection in selections)


def _selection_to_string(selection):
    parts = [str(part) for part in selection]
    return "(".join(parts) + ")" * (len(parts) - 1)
