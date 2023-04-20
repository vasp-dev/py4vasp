# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import dataclasses

from py4vasp._util import check

range_separator = ":"
pair_separator = "~"
group_separators = (range_separator, pair_separator)
operators = ("+", "-")
all = "__all__"


@dataclasses.dataclass
class Group:
    group: list
    separator: str
    __str__ = lambda self: self.separator.join(self.group)

    def __iadd__(self, character):
        self.group[-1] += character
        return self


class Tree:
    def __init__(self, parent=None):
        self._new_child = True
        self._ignore_separator = True
        self._is_operation = False
        self._operation_complete = False
        self._parent = parent
        self._children = []
        self._content = ""

    @classmethod
    def from_selection(cls, selection):
        tree = cls()
        active_node = tree
        selection = selection or ""
        message = f"Selection must be a string. The passed argument {selection} is not allowed."
        check.raise_error_if_not_string(selection, message)
        for character in selection:
            active_node = active_node.parse_character(character)
        return tree

    @property
    def nodes(self):
        return self._children

    @property
    def content(self):
        return self._content

    def parse_character(self, character):
        if character in (" ", ","):
            return self._parse_separator(character)
        elif character in group_separators:
            return self._parse_group(character)
        elif character in operators:
            return self._parse_operator(character)
        elif character == "(":
            return self._children[-1]
        elif character == ")":
            return self._parent._parse_separator(character)
        elif self._operation_complete:
            node = self._finalize_operation()
            return node._store_content_in_child(character)
        else:
            return self._store_content_in_child(character)

    def _parse_separator(self, character):
        if not self._ignore_separator:
            self._new_child = True
        if character == " ":
            self._operation_complete = self._is_operation and len(self._children) == 2
            return self
        return self._finalize_operation()

    def _parse_group(self, separator):
        self._ignore_separator = True
        self._new_child = False
        self._children[-1]._transform_to_group(separator)
        return self

    def _transform_to_group(self, separator):
        self._content = Group([self._content, ""], separator)

    def _parse_operator(self, operator):
        self._ignore_separator = True
        self._new_child = False
        self._children[-1]._transform_to_operation(operator)
        return self._children[-1]

    def _transform_to_operation(self, operator):
        self._is_operation = True
        self._operation_complete = False
        self._children.append(Tree(self))
        self._children[-1]._content = self._content
        self._content = operator

    def _finalize_operation(self):
        if not self._is_operation:
            return self
        if self._parent:
            self._parent._new_child = True
            return self._parent._parse_separator(",")
        else:
            return self

    def _store_content_in_child(self, character):
        self._ignore_separator = False
        self._add_child_if_necessary()
        self._children[-1]._content += character
        return self

    def _add_child_if_necessary(self):
        if self._new_child:
            self._children.append(Tree(self))
            self._new_child = False

    def __str__(self):
        return str(self._content)

    def __len__(self):
        if self._empty_tree():
            return 0
        elif len(self._children) == 0 or self._is_operation:
            return 1
        else:
            return sum(len(child) for child in self._children)

    @property
    def is_operation(self):
        return self._is_operation

    def selections(self):
        content = (self._content,) if self._parent else ()
        if len(self._children) == 0:
            yield content
        elif self._is_operation:
            yield content + (self._children[0].content, self._children[1].content)
        else:
            for child in self._children:
                for selection in child.selections():
                    yield content + selection

    def _empty_tree(self):
        return self._parent is None and not self._children


def selections_to_string(selections):
    "This routine is intended to convert selections back to string that would regenerate a tree."
    return ", ".join(_selection_to_string(selection) for selection in selections)


def _selection_to_string(selection):
    parts = [str(part) for part in selection]
    return "(".join(parts) + ")" * (len(parts) - 1)
