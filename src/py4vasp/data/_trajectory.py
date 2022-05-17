# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import copy
from py4vasp.data._base import DataBase
import py4vasp.exceptions as exception


def trajectory_examples(instance_name, function_name=None, step="step"):
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
you selected with the [] operator. Without any selection the results from the
final {step} will be used.

>>> calc.{instance_name}.{function_name}()

To select the results for all {step}s, you don't specify the array boundaries.

>>> calc.{instance_name}[:].{function_name}()

You can also select specific {step}s or a subset of {step}s as follows

>>> calc.{instance_name}[5].{function_name}()
>>> calc.{instance_name}[1:6].{function_name}()""".strip()


class DataTrajectory(DataBase):
    def _initialize(self):
        return self._set_steps_and_slice(-1)

    def __getitem__(self, steps):
        new = copy.copy(self)
        return new._set_steps_and_slice(steps)

    def _set_steps_and_slice(self, steps):
        self._steps = steps
        self._is_slice = isinstance(steps, slice)
        if self._is_slice:
            self._slice = steps
        elif steps == -1:
            self._slice = slice(-1, None)
        else:
            self._slice = _create_slice_for_current_step_if_possible(steps)
        return self

    @property
    def _last_step_in_slice(self):
        return (self._slice.stop or 0) - 1


def _create_slice_for_current_step_if_possible(steps):
    try:
        return slice(steps, steps + 1)
    except TypeError as error:
        message = f"Error creating slice [{steps}:{steps} + 1], please check the access operator argument."
        raise exception.IncorrectUsage(message) from error
