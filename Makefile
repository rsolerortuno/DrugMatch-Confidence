.PHONY: install test lint typecheck check app build clean

install:
	python -m pip install -e ".[app,dev]"

test:
	pytest -q

lint:
	ruff check src tests app scripts

typecheck:
	mypy src

check:
	python -m compileall -q src app scripts tests
	ruff check src tests app scripts
	mypy src
	pytest -q
	python -m drugmatch --help

app:
	streamlit run app/streamlit_app.py

build:
	python -m build

clean:
	rm -rf build dist .pytest_cache .mypy_cache .ruff_cache *.egg-info src/*.egg-info
