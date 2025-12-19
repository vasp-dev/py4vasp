# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
.PHONY: test format style import coverage doc clean

TEST ?= tests

test:
	pytest $(TEST)

coverage:
	pytest --cov=py4vasp --cov-report html

format: import style

style:
	black .

import:
	isort .

doc:
	make -C docs hugo

clean:
	make -C docs clean
