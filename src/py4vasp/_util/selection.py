# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)


range_separator = ":"
all = "__all__"


class SelectionTree:
    def __init__(self, parent=None):
        self._new_child = True
        self._is_range = False
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

    def parse_character(self, character):
        if character in (" ", ","):
            return self._parse_separator()
        elif character == range_separator:
            return self._parse_range(character)
        elif character == "(":
            return self._children[-1]
        elif character == ")":
            return self._parent._parse_separator()
        else:
            return self._store_content_in_child(character)

    def _parse_separator(self):
        self._new_child = not self._is_range
        return self

    def _parse_range(self, character):
        self._is_range = True
        self._new_child = False
        return self._store_content_in_child(character)

    def _store_content_in_child(self, character):
        self._add_child_if_necessary()
        self._children[-1]._content += character
        return self

    def _add_child_if_necessary(self):
        if self._new_child:
            self._children.append(SelectionTree(self))
            self._new_child = False

    def __str__(self):
        return self._content
