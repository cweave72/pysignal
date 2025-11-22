import logging
import numpy as np
from scipy import signal

import matplotlib
import matplotlib.pyplot as plt

import pysignal.utils as utils
import pysignal.spectrum as spec
import pysignal.plottools as plottools

from pysignal.filter.filter import Fir
from pysignal import setup_logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

matplotlib.use('TkAgg')


class Resampler(Fir):
    """Resampler class.
    """

    def __init__(self, num_paths: int, **kwargs):
        """Initializer for Resampler.
        Parameters:
            num_paths : Number of polyphase paths.

        kwargs (from filter.Fir):
            self.fpass  = kwargs.pop('fpass', None)
            self.fstop  = kwargs.pop('fstop', None)
            self.rippledB = kwargs.pop('rippledB', 0.1)
            self.stopdB = kwargs.pop('stopdB', 60)
            self.wl     = kwargs.pop('wl', 16)
            self.fl     = kwargs.pop('fl', 15)
            self.N      = kwargs.pop('N', None)
            self.fs     = float(kwargs.pop('fs', 1.))
        """
        super().__init__(**kwargs)
        self.P = num_paths

    def estimate_N(self):
        "Estimates the number of taps required to meet the design specs."
        delta = float(self.fstop) - (self.fpass)
        return int(np.ceil(((self.fs*self.P)/delta) * self.stopdB/22.0))

    def design(self, N=None, **kwargs):
        """Design the resampling prototype filter.
        kwargs (passed to filter.Fir.design method):
            stop_1overf: bool : Apply 1/f to stopband response (default True).
            grid_density: int : Remez grid density (defaults to 16)
        """
        if N is not None:
            self.N = N
        else:
            N = self.N

        if N is None:
            self.N = self.estimate_N()

        super().design(N=self.N, **kwargs)

        self.b = self.b/np.amax(self.b)

        # Compute derivative filter coefficients.
        db = np.convolve(self.b, np.array([1, 0, -1])/2)
        self.db = db[1:-1]

    def filter(
        self,
        x,
        fs_in,
        fs_out,
        path_init=0,
        rational_mode=False,
        use_quantized=False
    ):
        """Resamples the data vector x to new sample rate f_out.
        fs_out = (P/Q)*fs_in
        """
        # Ensure the filter is designed.
        if self.b is None:
            self.design()

        b = self.b
        db = self.db
        if use_quantized:
            self.quantize()
            b = self.bq

        taps_per_path = self.N//self.P

        # Perform polyphase partition of filter coefs.
        # P rows x P/N cols
        path_coefs = b.reshape(taps_per_path, self.P).T
        dpath_coefs = db.reshape(taps_per_path, self.P).T

        # Compute Q for the fs_in/fs_out ratio.
        Q = (fs_in/fs_out)*self.P
        logger.debug(f"Q={Q}")

        # Number of expected output samples (1:P/Q) interpolation factor.
        Nout = int(len(x)*(self.P/Q))
        y = np.zeros(Nout, dtype=complex)
        idx_out = 0

        reg = np.zeros(taps_per_path, dtype=complex)
        idx_in = 0

        accum = float(int(path_init))
        logger.debug(f"accum={accum}")

        while idx_in < len(x):

            # Roll the state register and deliver new input sample.
            reg = np.roll(reg, shift=1)
            reg[0] = x[idx_in]
            #logger.debug(f"reg={reg}")
            idx_in += 1

            # Compute output samples until accumulator overflow.
            while accum < self.P:
                path_index = int(accum)
                frac_part = accum - path_index
                #logger.debug(f"{idx_in} accum={accum}, path_idx={path_index} "
                #             f"path={path_coefs[path_index, :]}")
                inner = np.inner(reg, path_coefs[path_index, :])

                lin_correction = 0
                # If full arbitrary mode, compute linear correction from
                # derivative filter result.
                if not rational_mode:
                    deriv = np.inner(reg, dpath_coefs[path_index, :])
                    lin_correction = deriv * frac_part

                y[idx_out] = inner + lin_correction
                idx_out += 1

                accum += Q

            # Wrap accumulator modulo P.
            # Note: If the accumulator is still > P, the next output sample
            # will be skipped.
            accum -= self.P
                    
        return y


if __name__ == "__main__":
    from pysignal.comms import ShapingFilter

    setup_logging(logging.getLogger(), level='debug')

    shaped = ShapingFilter(0.2, 2, sqrt=False)
    shaped.plot_response()

    filt = Resampler(num_paths=32,
                     fpass=0.4, fstop=.75,
                     stopdB=66,
                     fs=64,
                     N=512,
                     w=[1, 10],
                     wl=16, fl=15)

    filt.design(stop_1overf=False, grid_density=16)
    filt.plotResponse(quantized=False, axis=[-2, 2, -80, 5]) 

    logger.info(f"filt={filt}")
    #logger.info(f"({len(filt.b)}) filt.b={filt.b}")

    filt.filter(np.zeros(100), 1, 2)

    plt.ion()
    plt.show()
    utils.enterShell()
