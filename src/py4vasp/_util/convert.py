import numpy as np


def text_to_string(text):
    "Text can be either bytes or string"
    try:
        return text.decode()
    except (UnicodeDecodeError, AttributeError):
        return text


def to_complex(array):
    assert array.dtype == np.float64
    assert array.shape[-1] == 2
    return array.view(np.complex128).reshape(array.shape[:-1])
