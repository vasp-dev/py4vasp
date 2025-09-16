# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import numpy as np

from py4vasp import exception
from py4vasp._calculation.electron_phonon_chemical_potential import (
    ElectronPhononChemicalPotential,
)
from py4vasp._util import check, convert, select, suggest

ALIAS = {
    "selfen_delta": "delta",
    "scattering_approx": "scattering_approximation",
}
NOT_FOUND = "not found"


class ElectronPhononAccumulator:
    "Helper class to create instances of electron phonon calculations"

    def __init__(self, parent, raw_data):
        self._parent = parent
        self._raw_data = raw_data
        self._name = convert.quantity_name(parent.__class__.__name__)

    def __str__(self):
        num_instances = len(self._parent)
        selection_options = self.selections()
        options_str = "\n".join(
            f"    {key}: {value}" for key, value in selection_options.items()
        )
        name = self._name.replace("electron_phonon", "Electron-phonon")
        name = name.replace("_", " ")
        return f"{name} with {num_instances} instance(s):\n{options_str}"

    def to_dict(self):
        return {"naccumulators": len(self._parent)}

    def selections(self, base_selections={}):
        mu_tag, mu_val = self.chemical_potential_mu_tag()
        return {
            **base_selections,
            mu_tag: np.unique(mu_val),
            "nbands_sum": np.unique(self._raw_data.nbands_sum),
            "selfen_delta": np.unique(self._raw_data.delta),
            "scattering_approx": np.unique(self._raw_data.scattering_approximation),
        }

    def _chemical_potential(self):
        new_chemical_potential = ElectronPhononChemicalPotential.from_data
        return new_chemical_potential(self._raw_data.chemical_potential)

    def chemical_potential_mu_tag(self):
        return self._chemical_potential().mu_tag()

    def chemical_potential_label(self):
        return self._chemical_potential().label()

    def select_indices(self, selection, *args_filters, **kwargs_filters):
        tree = select.Tree.from_selection(selection)
        return {
            index_
            for selection in tree.selections(filter=set(args_filters))
            for index_ in self._filter_indices(selection, kwargs_filters)
        }

    def _filter_indices(self, selection, filters):
        remaining_indices = range(len(self._raw_data.valid_indices))
        for key, value in filters.items():
            remaining_indices = self._filter_assignment(remaining_indices, key, value)
        for assignment in selection:
            self._raise_error_if_assignment_format_incorrect(assignment)
            remaining_indices = self._filter_assignment(
                remaining_indices, assignment.left_operand, assignment.right_operand
            )
            remaining_indices = list(remaining_indices)
        yield from remaining_indices

    def _raise_error_if_assignment_format_incorrect(self, assignment):
        if not isinstance(assignment, select.Assignment):
            message = f'\
The selection {assignment} is not formatted correctly. It should be formatted like \
"key=value". Please check the "selections" method for available options.'
            raise exception.IncorrectUsage(message)

    def _filter_assignment(self, remaining_indices, key, value):
        for index_ in remaining_indices:
            if self._match_key_value(index_, key, str(value)):
                yield index_

    def _match_key_value(self, index_, key, value):
        instance_value = self.get_data(key, index_)
        try:
            value = float(value)
        except ValueError:
            return instance_value == value
        return np.isclose(instance_value, float(value), rtol=1e-8, atol=0)

    def get_data(self, name, index):
        name = ALIAS.get(name, name)
        dataset = getattr(self._raw_data, name, NOT_FOUND)
        if dataset is NOT_FOUND:
            mu_tag, mu_val = self.chemical_potential_mu_tag()
            self._raise_error_if_not_present(name, expected_name=mu_tag)
            return mu_val[self._raw_data.id_index[index, 2] - 1]
        if check.is_none(dataset):
            return None
        return np.array(dataset[index])

    def _raise_error_if_not_present(self, name, expected_name):
        if name != expected_name:
            valid_names = set(self._parent.selections().keys())
            valid_names.remove(self._name)
            did_you_mean = suggest.did_you_mean(name, valid_names)
            available_selections = '", "'.join(valid_names)
            message = f'\
The selection "{name}" is not a valid choice. {did_you_mean}Please check the \
available selections: "{available_selections}".'
            raise exception.IncorrectUsage(message)
