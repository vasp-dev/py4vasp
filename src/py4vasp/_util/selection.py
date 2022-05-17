# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import dataclasses


range_separator = ":"
pair_separator = "~"
group_separators = (range_separator, pair_separator)
all = "__all__"


@dataclasses.dataclass
class SelectionGroup:
    group: list
    separator: str
    __str__ = lambda self: self.separator.join(self.group)

    def __iadd__(self, character):
        self.group[-1] += character
        return self


class SelectionTree:
    def __init__(self, parent=None):
        self._new_child = True
        self._ignore_separator = True
        self._is_group = False
        self._parent = parent
        self._children = []
        self._content = ""

    @classmethod
    def from_selection(cls, selection):
        tree = cls()
        active_node = tree
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
            return self._parse_separator()
        elif character in group_separators:
            return self._parse_group(character)
        elif character == "(":
            return self._children[-1]
        elif character == ")":
            return self._parent._parse_separator()
        else:
            return self._store_content_in_child(character)

    def _parse_separator(self):
        if not self._ignore_separator:
            self._new_child = True
        return self

    def _parse_group(self, separator):
        self._ignore_separator = True
        self._new_child = False
        self._children[-1]._transform_to_group(separator)
        return self

    def _transform_to_group(self, separator):
        self._content = SelectionGroup([self._content, ""], separator)

    def _store_content_in_child(self, character):
        self._ignore_separator = False
        self._add_child_if_necessary()
        self._children[-1]._content += character
        return self

    def _add_child_if_necessary(self):
        if self._new_child:
            self._children.append(SelectionTree(self))
            self._new_child = False

    def __str__(self):
        return str(self._content)
