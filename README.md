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

Running a marimo notebook (Ex: `notebooks/resampler_demo`):

1. Activate the virtual environment for this repo.
```
. init_venv
```

2. Run marimo edit:
```
(pysignal) $ marimo edit notebooks/resampler_demo/notebook.py
```

Go to your browser to see the notebook.
