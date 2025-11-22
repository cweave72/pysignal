"""
Module which defines a filter object and operations for designing and
plotting response of FIR filters.
"""
import logging
import textwrap
import numpy as np
from scipy import signal

import matplotlib
import matplotlib.pyplot as plt

import pysignal.utils as utils
import pysignal.spectrum as spec
import pysignal.plottools as plottools

from pysignal import setup_logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

matplotlib.use('TkAgg')


def myfrf(fpass, fstop, ripple_dB=0.1, atten_dB=60, fs=1, steps=50):
    """Generates the frequency, desired attenuation and weight arrays to create
    a 1/f stopband response.

    Parameters:
    fpass     : Passband frequency edge.
    fstop     : Stopband frequency edge.
    ripple_dB : Passband ripple, dB
    atten_dB  : Stopband attenuation, dB
    fs        : filter input sample rate
    steps     : Number of steps in resulting frequency grid.

    Returns: (freqs, amplitudes, weights) for call to remez.
    Reference: harris: Multirate Signal Processing, pp 64-65
    """
    f1 = fpass
    f2 = fstop
    f3 = fs/2.0
    
    d1 = 10**(ripple_dB/20) - 1
    d2 = 10**(-atten_dB/20)

    # Compute ratio of passband ripple to stopband rejection.
    w = d1/d2

    # Ending stopband weight.
    K = f3/f2*w

    # Set the guard between frequency steps to be a percentage of the grid
    # step-size.
    guard_pct = 0.15
    guard = guard_pct * (f3 - f2)/steps

    #logger.debug(f"fs={fs}")
    #logger.debug(f"(fpass) f1={f1}")
    #logger.debug(f"(fstop) f2={f2}")
    #logger.debug(f"(fnyq)  f3={f3}")
    #logger.debug(f"(ripple lin) d1={d1}")
    #logger.debug(f"(atten lin) d2={d2}")
    #logger.debug(f"(d1/d2) w={w}")
    #logger.debug(f"guard={guard}")
    #logger.debug(f"K={K}")

    #Create weight vector for stop band.
    x = np.linspace(f2, f3, steps)
    #Create a line passing through 2 pts (f2,w), (f3,K)
    ww = (w-K)/(f2-f3)*(x - f3) + K
    ww = ww.flatten()
    ww = np.insert(ww, 0, 1)
    ww = ww[0:-1]

    #Create freq vector.
    wf = np.linspace(f2, fs/2.0, steps)
    wf = np.reshape(wf, (steps, -1))
    wff = np.concatenate((wf[0:-1], wf[1:]-guard), axis=1)
    wff[-1, 1] = wff[-1,1] + guard
    ff = wff.flatten()
    ff = np.insert(ff, 0, 0)
    ff = np.insert(ff, 1, f1)

    #Create amplitude vector
    aa = np.zeros(steps)
    aa[0] = 1

    return (ff, aa, ww)


class Fir:
    " Container class for FIR filter specification and design. "

    def __init__(self, **kwargs):

        self.fpass  = kwargs.pop('fpass', None)
        self.fstop  = kwargs.pop('fstop', None)
        self.rippledB = kwargs.pop('rippledB', 0.1)
        self.stopdB = kwargs.pop('stopdB', 60)
        self.wl     = kwargs.pop('wl', 16)
        self.fl     = kwargs.pop('fl', 15)
        self.N      = kwargs.pop('N', None)
        self.fs     = float(kwargs.pop('fs', 1.))

        self.b = None

    def estimate_N(self):
        "Estimates the number of taps required to meet the design specs."
        delta = float(self.fstop) - (self.fpass)
        return int(np.ceil((self.fs/delta) * self.stopdB/22.0))

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

    def design(self, N=None, stop_1overf=True, grid_density=16, debug_log=False):
        """Designs the filter using remez algorithm.
        """
        # Get estimate for N if not already set.
        if N is not None:
            self.N = N
        elif self.N is None:
            self.N = self.estimate_N()

        if stop_1overf:
            self.f, self.a, self.w = myfrf(self.fpass,
                                           self.fstop,
                                           self.rippledB,
                                           self.stopdB,
                                           fs=self.fs)
            if debug_log:
                logger.debug(f"myfrf: f={self.f}")
                logger.debug(f"myfrf: a={self.a}")
                logger.debug(f"myfrf: w={self.w}")
        else:
            # Build freq and gain vectors.
            self.f = [0.0, float(self.fpass), float(self.fstop), self.fs/2.0]
            self.a = [1, 0]
            d1 = 10**(self.rippledB/20) - 1
            d2 = 10**(-self.stopdB/20)
            # Compute ratio of passband ripple to stopband rejection.
            self.w = [1, int(d1/d2)]

        try:
            logger.debug(f"Designing filter with {self.N} taps (grid_density={grid_density})")
            self.b = signal.remez(self.N,
                                  self.f,
                                  self.a,
                                  weight=self.w,
                                  fs=self.fs,
                                  grid_density=grid_density)
        except Exception as e:
            logger.exception("Error designing filter: %s" % str(e))
            raise e

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

        kwargs.setdefault('title', f'Filter Response (N={self.N})')
        kwargs.setdefault('xlabel', 'Freq')
        kwargs.setdefault('ylabel', 'Magnitude dB')
        kwargs.setdefault('marker', '')
        axis = kwargs.pop('axis', None)
        yaxis = kwargs.pop('yaxis', [-100, 3])
        xaxis = kwargs.pop('xaxis', [-self.fs/2, self.fs/2])

        if axis is not None:
            kwargs.setdefault('axis', axis)
        else:
            kwargs.setdefault('axis', xaxis + yaxis)

        fig, ax = plottools.plot(f, np.fft.fftshift(self.H), **kwargs)
        ymin, ymax = ax.get_ylim()
        xmin, xmax = ax.get_xlim()
        ax.vlines([-self.fpass, -self.fstop, self.fpass, self.fstop],
                  ymin, ymax, linestyles='dashed', color='red')
        ax.hlines(-self.stopdB, xmin, xmax, linestyles='dashed', color='green') 

        return fig, ax

    def plotStem(self, quantized=False, title=None):
        """Plots the impulse response as a stem plot.
        """
        x = np.arange(0, self.N)
        # Create new figure and axis.
        fig, ax = plt.subplots()

        if title is None:
            title = f"{self.__class__.__name__} Impulse response N={self.N}."

        b = self.b
        if quantized:
            b = self.bq

        ax.stem(x, b)
        ax.set_title(title)
        ax.grid('on')

        return fig, ax
        
    def __str__(self):
        ret = (f"{self.__class__.__name__}("
               f"fpass={self.fpass:.3f}, fstop={self.fstop:.3f}, "
               f"rippledB={self.rippledB}, "
               f"stopdB={self.stopdB}, Fs={self.fs:.2f}, wl={self.wl}, "
               f"fl={self.fl}, N={self.N})")
        return ret


if __name__ == "__main__":
    setup_logging(logging.getLogger(), level='debug')

    filt = Fir(fpass=0.2, fstop=0.21, rippledB=0.1, stopdB=80, wl=16, fl=15)
    filt.design(stop_1overf=True)
    filt.plotResponse(quantized=False, yaxis=[-120, 5]) 

    logger.info(f"filt={filt}")
    logger.info(f"({len(filt.b)}) filt.b={filt.b}")

    plt.ion()
    plt.show()
    utils.enterShell()
