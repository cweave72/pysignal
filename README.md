# pysignal

A collection of signal processing utilities.

## Getting Started

Clone: `git clone https://github.com/cweave72/pysignal.git`

Build the virtual env (requires `uv`):
```
cd pysignal
. init_venv.sh
```

Run the quick test:
```
(pysignal) python test.py
```

## Running tests

```
(pysignal) pytest -vv pysignal/tests
```

## Notebooks

Running a marimo notebook (Ex: `pysignal/notebooks/resampler_demo.py`):

1. Activate the virtual environment for the repo.
```
. init_venv
```

2. Run one of the `marimo edit` commands below:
```
# Open marimo browser interface (see all available notebooks).
(pysignal) $ marimo edit

# Open a specific notebook.
(pysignal) $ marimo edit <path to notebook>
```
