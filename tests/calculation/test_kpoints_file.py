# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import pytest

from py4vasp._calculation import _kpoints_file


@pytest.mark.parametrize(
    "name, expected",
    [
        ("GAMMA", "Γ"),
        ("SIGMA_0", "Σ₀"),
        ("DELTA_0", "Δ₀"),
        ("LAMBDA_0", "Λ₀"),
        ("K_2", "K₂"),
        ("H_12", "H₁₂"),
        ("X", "X"),
        # seekpath only labels its special points with the Greek letters above and a
        # numeric subscript; anything else is passed on as it is
        ("UNKNOWN", "UNKNOWN"),
        ("H_a", "H_a"),
    ],
)
def test_label_to_unicode(name, expected):
    assert _kpoints_file.label_to_unicode(name) == expected
