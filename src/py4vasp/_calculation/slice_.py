# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import copy

import numpy as np

from py4vasp import exception
from py4vasp._raw.data_wrapper import VaspData


def examples(instance_name, function_name=None, step="step"):
    if function_name is None:
        function_name = "read"
        access = "a method of this class"
        depend_on = f"the {step}s"
    else:
        access = "this method"
        depend_on = f"the {step}s of the class"
    return f"""
Examples
--------
If you access {access}, the result will depend on {depend_on} that
you selected with the [] operator. Without selected steps the results from the
final {step} will be used.

>>> calculation.{instance_name}.{function_name}()

To select the results for all {step}s, you don't specify the array boundaries.

>>> calculation.{instance_name}[:].{function_name}()

You can also select specific {step}s or a subset of {step}s as follows

>>> calculation.{instance_name}[5].{function_name}()
>>> calculation.{instance_name}[1:6].{function_name}()""".strip()


class Mixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._set_steps_and_slice()
        self._original = True

    def __getitem__(self, steps):
        self._raise_error_if_not_original()
        new = copy.copy(self)
        new._original = False
        return new._set_steps_and_slice(steps)

    def _set_steps_and_slice(self, steps=None):
        steps = self._default_steps() if steps is None else steps
        self._steps = steps
        self._is_slice = isinstance(steps, slice)
        if self._is_slice:
            self._slice = steps
        elif steps == -1:
            self._slice = slice(-1, None)
        else:
            self._slice = _create_slice_for_current_step_if_possible(steps)
        return self

    # override this method to implement a different default step choice
    def _default_steps(self):
        return -1

    @property
    def _last_step_in_slice(self):
        return (self._slice.stop or 0) - 1

    def _raise_error_if_not_original(self):
        if not self._original:
            message = "Taking nested slices is not implemented. Please derive all slices from the original Refinery."
            raise exception.NotImplemented(message)


def _create_slice_for_current_step_if_possible(steps):
    try:
        return slice(steps, steps + 1)
    except TypeError as error:
        message = f"Error creating slice [{steps}:{steps} + 1], please check the access operator argument."
        raise exception.IncorrectUsage(message) from None


def check_single_step(steps):
    "Return the step as int; only a single step can be selected from a history."
    if isinstance(steps, (int, np.integer)) and not isinstance(steps, bool):
        return int(steps)
    message = f"Select a single step with an integer, {steps!r} is not supported."
    raise exception.IncorrectUsage(message)


def take_step(history, step):
    """Return one step of a history, i.e. data whose first dimension counts the steps,
    as raw data (a scalar if the step holds a single value)."""
    number_steps = len(history)
    if not -number_steps <= step < number_steps:
        message = (
            f"The step {step} is out of range, the data contains {number_steps} steps."
        )
        raise exception.IncorrectUsage(message)
    result = VaspData(np.asarray(history[step]))
    return result[()] if result.ndim == 0 else result
