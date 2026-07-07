# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
import contextlib
from typing import Optional


def record_encountered_error(
    encountered_errors: Optional[dict[str, list[str]]],
    key: str,
    error: Exception,
    context: Optional[str] = None,
):
    """Store a concise error message for later inspection by database callers."""
    if encountered_errors is None or key is None:
        return
    message = f"{type(error).__name__}: {error}"
    if context:
        message = f"{context} | {message}"
    encountered_errors.setdefault(key, []).append(message)


@contextlib.contextmanager
def suppress_and_record(
    encountered_errors: Optional[dict[str, list[str]]],
    key: str,
    *exceptions,
    context: Optional[str] = None,
):
    """Like contextlib.suppress, but also records the suppressed error message."""
    try:
        yield
    except exceptions as error:
        record_encountered_error(encountered_errors, key, error, context=context)
