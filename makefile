# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
.PHONY: test format

TEST ?= tests

test:
	poetry run pytest $(TEST)


format:
	poetry run black .