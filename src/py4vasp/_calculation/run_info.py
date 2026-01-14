# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
from py4vasp._calculation import base, structure


class RunInfo(base.Refinery):
    "Contains information about the VASP run."

    @base.data_access
    def to_dict(self):
        "Convert the run information to a dictionary."
        positions = self._structure().positions()
        number_steps = 1 if positions.ndim==2 else positions.shape[0]
        return {
            "vasp_version": self._raw_data.runtime_data.vasp_version,
            "number_steps": number_steps,
        }
    
    @base.data_access
    def _to_database(self, *args, **kwargs):
        return {
            "run_info": self.to_dict(),
        }
    
    def _structure(self):
        return structure.Structure.from_data(self._raw_data.structure)