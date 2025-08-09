# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import numpy as np

from py4vasp import exception
from py4vasp._calculation.electron_phonon_chemical_potential import (
    ElectronPhononChemicalPotential,
)
from py4vasp._util import convert, select, suggest

ALIAS = {
    "selfen_delta": "delta",
    "scattering_approx": "scattering_approximation",
}


class ElectronPhononAccumulator:
    "Helper class to create instances of electron phonon calculations"

    def __init__(self, accumulator, raw_data):
        self._accumulator = accumulator
        self._raw_data = raw_data
        self._name = convert.quantity_name(accumulator.__class__.__name__)

    def __str__(self):
        num_instances = len(self._accumulator)
        selection_options = self._accumulator.selections()
        selection_options.pop(self._name, None)
        options_str = "\n".join(
            f"    {key}: {value}" for key, value in selection_options.items()
        )
        name = self._name.replace("electron_phonon", "Electron-phonon")
        name = name.replace("_", " ")
        return f"{name} with {num_instances} instance(s):\n{options_str}"

    def to_dict(self):
        return {"naccumulators": len(self._accumulator)}

    def selections(self, base_selections):
        mu_tag, mu_val = self.chemical_potential_mu_tag()
        return {
            **base_selections,
            mu_tag: np.unique(mu_val),
            "nbands_sum": np.unique(self._raw_data.nbands_sum),
            "selfen_delta": np.unique(self._raw_data.delta),
            "scattering_approx": np.unique(self._raw_data.scattering_approximation),
        }

    def chemical_potential_mu_tag(self):
        chemical_potential = ElectronPhononChemicalPotential.from_data(
            self._raw_data.chemical_potential
        )
        return chemical_potential.mu_tag()

    def select_indices(self, selection, **filters):
        tree = select.Tree.from_selection(selection)
        return {
            index_
            for selection in tree.selections()
            for index_ in self._filter_indices(selection, filters)
        }

    def _filter_indices(self, selection, filters):
        remaining_indices = range(len(self._raw_data.valid_indices))
        for key, value in filters.items():
            remaining_indices = self._filter_group(remaining_indices, key, value)
        for group in selection:
            self._raise_error_if_group_format_incorrect(group)
            assert len(group.group) == 2
            remaining_indices = self._filter_group(remaining_indices, *group.group)
            remaining_indices = list(remaining_indices)
        yield from remaining_indices

    def _raise_error_if_group_format_incorrect(self, group):
        if not isinstance(group, select.Group) or group.separator != "=":
            message = f'\
The selection {group} is not formatted correctly. It should be formatted like \
"key=value". Please check the "selections" method for available options.'
            raise exception.IncorrectUsage(message)

    def _filter_group(self, remaining_indices, key, value):
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
        dataset = getattr(self._raw_data, name, None)
        if dataset is not None:
            return np.array(dataset[index])
        mu_tag, mu_val = self.chemical_potential_mu_tag()
        self._raise_error_if_not_present(name, expected_name=mu_tag)
        return mu_val[self._raw_data.id_index[index, 2] - 1]

    def _raise_error_if_not_present(self, name, expected_name):
        if name != expected_name:
            valid_names = set(self._accumulator.selections().keys())
            valid_names.remove(self._name)
            did_you_mean = suggest.did_you_mean(name, valid_names)
            available_selections = '", "'.join(valid_names)
            message = f'\
The selection "{name}" is not a valid choice. {did_you_mean}Please check the \
available selections: "{available_selections}".'
            raise exception.IncorrectUsage(message)
