"""spectrum.py

A collection of spectral calculation functions.
"""

import numpy as np
import scipy.signal as signal

import pysignal.utils as utils

import logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def computeSpec(x, fs=1, Nfft=None, wind=None):
    """Computes the spectrum of the input signal x
    Returns the tuple (X, f) (non fftshift'ed arrays).
    """

    x = np.array(x, dtype=complex)
    fs = float(fs)

    if Nfft is None:
        Nfft = len(x)

    if wind is not None:
        w = signal.get_window(wind, len(x))
        X = np.fft.fft(x*w, Nfft)
    else:
        X = np.fft.fft(x, Nfft)

    # Get the frequency axis.
    f = np.linspace(-fs/2, fs/2, Nfft, endpoint=False)

    return X, f


def computeAvgSpec(x, fs=1, Nfft=None, wind='blackman'):
    """Computes the averaged spectra of the signal x.
    """

    N = len(x)
    logger.debug(f"N={N}")
    # Ensure length is even.
    N -= (N % 2)

    if Nfft is None:
        # If Nfft is not specified, find maximum Nfft which still guarantees 8
        # averages (assume overlap of Nfft/2).
        minNumAvgs = 8
        M = 16
        while M >= 4:
            numAvgs = 2*(N/(2**M)) - 1
            if numAvgs > minNumAvgs:
                break
            M -= 1

        Nfft = 2**M
        logger.debug(f"Using Nfft={Nfft}, {numAvgs} averages")
    else:
        logger.debug(f"Using Nfft={Nfft}")

    Xavg = np.zeros(Nfft, dtype=x.dtype)
    k = 0
    for chunk in utils.chunker(x, Nfft, Nfft//2):
        if len(chunk) == Nfft:
            X, f = computeSpec(chunk, fs=fs, Nfft=Nfft, wind=wind)
            Xavg += X
            k += 1
        else:
            break

    Xavg /= k

    logger.debug(f"Computed {k} averages.")
    return Xavg, f
