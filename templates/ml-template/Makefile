PYTHON ?= python3
UV ?= uv
PYTEST ?= pytest

.PHONY: help install train evaluate predict check test clean

help:
	@echo "ML template commands"
	@echo "  make install"
	@echo "  make train"
	@echo "  make evaluate"
	@echo "  make predict"
	@echo "  make check"
	@echo "  make test"
	@echo "  make clean"

install:
	$(UV) sync

train:
	$(PYTHON) scripts/train.py

evaluate:
	$(PYTHON) scripts/evaluate.py

predict:
	$(PYTHON) scripts/predict.py

# Compile the entry-point scripts to catch obvious syntax regressions early.
check:
	$(PYTHON) -m compileall scripts

test:
	$(PYTEST)

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} +
