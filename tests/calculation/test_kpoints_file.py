# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import types

import numpy as np
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


@pytest.fixture
def three_segments():
    path = types.SimpleNamespace()
    path.coordinates = np.array(
        [
            [[0, 0, 0], [0.5, 0.5, 0]],
            [[0.5, 0.5, 0], [0.5, 0.75, 0.25]],
            [[0.5, 0.75, 0.25], [0, 0, 0]],
        ]
    )
    path.labels = [("Γ", "X"), ("X", "W"), ("W", "Γ")]
    path.comment = "k points along high symmetry lines"
    path.number_points = 40
    return path


def test_line_mode(three_segments):
    expected = """\
k points along high symmetry lines
40
line mode
reciprocal
  0.00000000   0.00000000   0.00000000  Γ
  0.50000000   0.50000000   0.00000000  X

  0.50000000   0.50000000   0.00000000  X
  0.50000000   0.75000000   0.25000000  W

  0.50000000   0.75000000   0.25000000  W
  0.00000000   0.00000000   0.00000000  Γ"""
    actual = _kpoints_file.line_mode(
        three_segments.coordinates,
        three_segments.labels,
        three_segments.number_points,
        three_segments.comment,
    )
    assert actual == expected


def test_line_mode_does_not_comment_labels(three_segments):
    # Many tools hide the labels behind a comment character; then VASP does not read
    # them. VASP expects the label as the fourth field of the k-point line.
    text = _kpoints_file.line_mode(
        three_segments.coordinates,
        three_segments.labels,
        three_segments.number_points,
        three_segments.comment,
    )
    assert "!" not in text
    assert "#" not in text
    kpoint_lines = [line for line in text.splitlines()[4:] if line.strip()]
    assert [line.split()[3] for line in kpoint_lines] == list("ΓXXWWΓ")
