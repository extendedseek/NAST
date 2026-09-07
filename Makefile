.PHONY: install install-dev test lint format smoke validate clean

install:
	python -m pip install -e .

install-dev:
	python -m pip install -e ".[dev]"

test:
	python -m unittest discover -s tests -v

lint:
	ruff check src tests scripts

format:
	ruff format src tests scripts

smoke:
	python scripts/smoke_selector.py

validate:
	PYTHONPATH=src python -m nast validate-config --config configs/smoke.yaml

clean:
	python -c "import pathlib, shutil; [shutil.rmtree(p) for p in pathlib.Path('.').rglob('__pycache__') if p.is_dir()]"
