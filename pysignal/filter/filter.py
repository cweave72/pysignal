"""
Module which defines a filter object and operations for designing and
plotting response of FIR filters.
"""
import logging
import math
import textwrap
import numpy as np
from scipy import signal

import pysignal.utils as utils
import pysignal.spectrum as spec
import pysignal.plottools as plottools

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class Filter(object):
    " Container class for filter specification and design. "

    def __init__(self, *args, **kwargs):

        self.fpass  = kwargs.pop('fpass', None)
        self.fstop  = kwargs.pop('fstop', None)
        self.stopdB = kwargs.pop('stopdB', None)
        self.wl     = kwargs.pop('wl', None)
        self.fl     = kwargs.pop('fl', None)
        self.N      = kwargs.pop('N', None)
        self.fs     = float(kwargs.pop('fs', 1.))
        self.w      = kwargs.pop('w', [1, 10])

        self.b = None

    def estimateN(self):
        "Estimates the number of taps required to meet the design specs."
        delta = float(self.fstop) - (self.fpass)
        return int(math.ceil((self.fs/delta) * self.stopdB/22.0))

    def quantize(self, wl=None, fl=None):
        "Quantizes the filter coefficients."
        if self.b is None:
            logger.error("Run Filter.design() first.")
            return

        if wl is not None:
            self.wl = wl
            self.fl = fl

        self.bq = utils.quantize(self.b, self.wl, self.fl)
        self.coef = utils.quantize(self.b, self.wl, self.fl, out='scaledint')

    def design(self, numTaps=None, filttype='bandpass'):
        "Designs the filter using remez algorithm."

        # Build freq and gain vectors.
        self.f = [0.0, float(self.fpass), float(self.fstop), self.fs/2.0]
        logger.debug(f"f={self.f}")
        self.a = [1, 0]

        # Get estimate for N if not already set.
        if numTaps is not None:
            self.N = numTaps
        elif self.N is None:
            self.N = self.estimateN()

        try:
            self.b = signal.remez(self.N, self.f, self.a, weight=self.w,
                                  fs=self.fs, type=filttype)
        except Exception as e:
            logger.exception("Error designing filter: %s" % str(e))
            return

        # Normalize so the largest coefficient is 1.
        self.b /= np.amax(self.b)

        # Quantize filter if fixed-point parameters have been specified.
        if (self.wl is not None) and (self.fl is not None):
            self.quantize()

    def plotResponse(self, quantized=False, **kwargs):
        "Computes the frequency response of the filter."

        if self.b is None:
            self.design()

        if not quantized:
            X, f = spec.computeSpec(self.b, self.fs, Nfft=10*len(self.b))
        else:
            X, f = spec.computeSpec(self.bq, self.fs, Nfft=10*len(self.b))

        Xnorm = np.absolute(X/np.amax(X))**2
        self.H = 10.*np.log10(Xnorm)

        kwargs.setdefault('title', 'Filter Response (N=%u)' % self.N)
        kwargs.setdefault('xlabel', 'Freq')
        kwargs.setdefault('ylabel', 'Magnitude dB')
        kwargs.setdefault('axis', [-self.fs/2, self.fs/2, -100, 3])
        kwargs.setdefault('marker', '')

        fig, ax = plottools.plot(f, self.H, **kwargs)
        ymin, ymax = ax.get_ylim()
        xmin, xmax = ax.get_xlim()
        ax.vlines([-self.fpass, -self.fstop, self.fpass, self.fstop],
                  ymin, ymax, linestyles='dashed', color='red')
        ax.hlines(-self.stopdB, xmin, xmax, linestyles='dashed', color='green') 
        
    def __str__(self):
        txt = '''\
        Filter(fpass=%2.1f; fstop=%2.1f, stopdB=%2.1f, Fs=%2.1f, wl=%u, fl=%u, N=%u)
        ''' % (
            self.fpass,
            self.fstop,
            self.stopdB,
            self.fs,
            self.wl,
            self.fl,
            self.N)
        return textwrap.dedent(txt)


if __name__ == "__main__":
    filt = Filter(fpass=0.2, fstop=0.3, stopdB=80, wl=16, fl=15)
    filt.setWeight(1, 100)
    filt.plotResponse(quant=True) 
    logger.debug(f"filt.b={filt.b}")
    logger.debug(f"filt={filt}")
