# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
.PHONY: test format style import coverage

TEST ?= tests

test:
	poetry run pytest $(TEST)

coverage:
	poetry run pytest --cov=py4vasp --cov-report html

format: import style

style:
	poetry run black .

import:
	poetry run isort .
