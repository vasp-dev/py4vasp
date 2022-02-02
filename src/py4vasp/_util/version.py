# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
from py4vasp.raw import RawVersion
import py4vasp.exceptions as exception
import functools

minimal_vasp_version = RawVersion(6, 2)
current_vasp_version = RawVersion(6, 2, 1)


def require(version, err_msg=None):
    def decorator_require(func):
        @functools.wraps(func)
        def func_with_requirement_test(raw_data, *args, **kwargs):
            my_version = raw_data.version
            if my_version >= version:
                return func(raw_data, *args, **kwargs)
            else:
                error_message = err_msg
                if error_message is None:
                    error_message = (
                        f"You called {func.__qualname__} which is not compatible with "
                        f"the {my_version.major}.{my_version.minor}.{my_version.patch}"
                        " version of Vasp you are using. Please use at least version "
                        f"{version.major}.{version.minor}.{version.patch} for this "
                        "feature."
                    )
                raise exception.OutdatedVaspVersion(error_message)

        return func_with_requirement_test

    return decorator_require
