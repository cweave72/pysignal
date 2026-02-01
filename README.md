# pysignal

A Python library for digital signal processing and communications, with a focus on filter design, modulation, and spectral analysis.

## Features

| Module | Description |
|--------|-------------|
| `filter` | Digital filter design, polyphase resamplers |
| `comms` | QPSK modulation, Nyquist pulse shaping, noise generation |
| `plottools` | Spectrum, eye diagram, and time-domain plotting |
| `spectrum` | Spectral analysis utilities |
| `utils` | Common DSP utilities (upsampling, bit manipulation, etc.) |

## Getting Started

**Requirements:** Python 3.10+, [uv](https://docs.astral.sh/uv/)

```bash
git clone https://github.com/cweave72/pysignal.git
cd pysignal
source init_venv.sh
```

## Quick Example

```python
from pysignal.filter.filter import Filter
from pysignal import setup_logging
import logging

setup_logging(rootlogger=logging.getLogger(), level='info')

# Design a lowpass filter
filt = Filter(fpass=1, fstop=1.7, fs=10, stopdB=60, wl=16, fl=15)
filt.design()
filt.plotResponse(quantized=True)
```

## Running Tests

```bash
pytest tests/ -v
```

## Notebooks

Interactive [Marimo](https://marimo.io/) notebooks are available in `notebooks/`:

```bash
# Open a specific notebook
marimo edit notebooks/resampler_demo.py

# Or browse all notebooks
marimo edit
```

## Project Structure

```
pysignal/
├── pysignal/      # Source package
├── tests/         # Test suite
├── notebooks/     # Marimo notebooks
└── pyproject.toml
```
