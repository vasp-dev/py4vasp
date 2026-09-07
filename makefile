# Copyright © VASP Software GmbH,
# Licensed under the Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0)
.PHONY: test test-core format style import coverage build doc clean release

TEST ?= tests packages

test:
	pytest $(TEST)

# The py4vasp-core installation: numpy and h5py only, everything else must degrade.
# The tests under packages/ skip themselves, see their conftest.py.
test-core:
	uv sync --package py4vasp-core --no-default-groups --group test
	uv run --no-sync pytest
	uv sync --all-packages --all-extras --no-extra mdtraj

coverage:
	pytest --cov=py4vasp --cov=vasp --cov-report html

format: import style

style:
	black .

import:
	isort .

build:
	uv build --all-packages

doc:
	make -C docs hugo

clean:
	make -C docs clean
	rm -rf dist

# The version lives in src/py4vasp/__init__.py; the two wrapper distributions have to
# repeat it because a metadata-only package has no module to read it from. Keep them in
# step here -- tests/test_version.py fails if they drift.
release:
ifndef VERSION
	$(error Usage: make release VERSION=0.12.0)
endif
	uv run --no-sync python scripts/set_version.py $(VERSION)
	uv lock
	uv run --no-sync pytest tests/test_version.py
	@echo "Now review the diff, reset __DB_SCHEMA__ in src/py4vasp/_raw/models.py if this"
	@echo "is a new minor series, and regenerate the schema snapshot with"
	@echo "  pytest tests/raw/test_schema_version.py --update-schema-snapshot"
