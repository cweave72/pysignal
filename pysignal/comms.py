import logging
import numpy as np
import scipy.signal as signal
import random
import commpy.filters as commsfilters

from typing import List

import pysignal.utils as utils
import pysignal.plottools as plottools

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class ShapingFilter:
    """Generates a shaping filter object.
    """

    def __init__(self, alpha, M, Nsymb=12, sqrt=False, wl=None, fl=None):
        """
        :alpha: Excess bandwidth factor.
        :M: Upsample factor
        :Nsymb: Number of symbols in the filter.
        :sqrt: Generate a root-raised cosine filter.
        """
        self.alpha = alpha
        self.M = M
        self.sqrt = sqrt
        self.Nsymb = Nsymb

        self.b = nyquistFilt(alpha, M, Nsymb, sqrt)
        self.b = self.b/np.amax(self.b)
        logger.debug(f"b={self.b}")

        if wl is not None:
            self.bq = utils.quantize(self.b, wl, fl)
            logger.debug(f"bq={self.bq}")

    def plot_response(self, use_quantized=False):
        """Plots response of shaping filter.
        Returns (fig, ax)
        """
        b = self.bq if use_quantized else self.b
        fig, ax = plottools.plotSpec(
            b,
            Nfft=1024,
            fs=self.M,
            title=f'Shaping Filter: alpha={self.alpha}, N={len(b)}, sqrt={self.sqrt}',
            xlabel='freq normalized to symbol rate',
            ylabel='dB',
            axis=[-self.M/2, self.M/2, -80, 5])
        return fig, ax

    def process(self, mapped_symbs: np.ndarray, use_quantized=False):
        """Process provided symbols through the shaping filter.
        """
        b = self.bq if use_quantized else self.b
        up = utils.upsample(mapped_symbs, self.M)
        y = signal.lfilter(b, 1, up)
        return y


class SymbolMapper:
    """Maps input bits to symbols based on provided mapping.
    """

    def __init__(self, mapping: dict, scale_factor):
        """Initializes the mapper.
        Example QPSK mapper:
        mapping = {
            0:  1 + 1j,
            1: -1 + 1j,
            2: -1 - 1j,
            3:  1 - 1j
        }
        scale = 1/np.sqrt(2)

        """
        self.mapping = mapping
        self.scale = scale_factor
        self.bps = np.log2(len(mapping))

    def map_bytes(self, inp: bytes, msb_first=True):
        """Maps input bytes to symbols.
        Returns complex mapped symbols.
        """
        self.symbs_int = []
        bits = utils.bytes_to_bits(inp, msb_first=msb_first)
        Nsymbs = len(bits) / self.bps
        self.symbs_iq = np.zeros(Nsymbs, dtype=complex)

        for k, symbol_bits in enumerate(utils.chunker(bits, self.bps)):
            symb_int = utils.bits_to_int(symbol_bits)
            self.symbs_int.append(symb_int)
            mapped = self.mapping.get(symb_int)
            self.symbs_iq[k] = mapped * self.scale

        return self.symbs_iq

    def map_symbs(self, inp: list):
        """Maps input symbols to iq.
        Returns complex mapped symbols.
        """
        self.symbs_iq = np.zeros(len(inp), dtype=complex)
        for k, s in enumerate(inp):
            mapped = self.mapping.get(s)
            self.symbs_iq[k] = mapped * self.scale

        return self.symbs_iq

    def __str__(self):
        ret = (f"SymbolMapper(mapping={self.mapping}, "
               f"scale={self.scale}, bps={self.bps}")
        return ret


class QpskMapper(SymbolMapper):

    def_scale = 1/np.sqrt(2)
    def_mapping = {
        0:  1 + 1j,
        1: -1 + 1j,
        2: -1 - 1j,
        3:  1 - 1j
    }

    def __init__(self, mapping=None):
        def_mapping = self.def_mapping
        def_scale = self.def_scale

        if mapping is not None:
            def_mapping = mapping

        super().__init__(mapping=def_mapping, scale_factor=def_scale)

    def __str__(self):
        ret = (f"QpskMapper(mapping={self.mapping}, "
               f"scale={self.scale}, bps={self.bps}")
        return ret


def genNormalizedAwgn(normPwr_dB, size, cmplx=True):
    """ Generates normalized AWGN sequence with power level normPwr_dB relative
    to a unit-power signal.
    """
    pwrLin = np.power(10, float(normPwr_dB)/10)

    if cmplx:
        N = (np.sqrt(pwrLin/2)*np.random.randn(size) + 
             1j*np.sqrt(pwrLin/2)*np.random.randn(size))
    else:
        N = np.sqrt(pwrLin)*np.random.randn(size)

    return N


def getNoisePwr(sig, EsN0_dB, BW):
    """Returns the noise power required to add noise to generate a signal with 
    Es/N0 equal to the value provided in the given bandwidth.
    """
    EsN0_lin = np.power(10, EsN0_dB/10)
    P = np.var(sig)
    Es = P/BW
    N0 = Es/EsN0_lin
    return N0


def addNoise(sig, EsN0_dB, BW, cmplx=True):
    """Adds noise to signal in given BW.
    """
    N0 = getNoisePwr(sig, EsN0_dB, BW)

    size = len(sig)
    if cmplx:
        noise = (np.sqrt(N0/2)*np.random.randn(size) + 
                 1j*np.sqrt(N0/2)*np.random.randn(size))
    else:
        noise = np.sqrt(N0)*np.random.randn(size)

    return sig + noise


def genNoise(Npwr, size, cmplx=True):
    """Generates noise with specified power.
    """
    if cmplx:
        noise = (np.sqrt(Npwr/2)*np.random.randn(size) + 
                 1j*np.sqrt(Npwr/2)*np.random.randn(size))
    else:
        noise = np.sqrt(Npwr)*np.random.randn(size)

    return noise


def nyquistFilt(alpha, M, Nsymb, sqrt=False):
    """Generates a nyquist or root-nyquist (i.e. raised-cosine) filter.
    :alpha: Excess bandwidth factor.
    :M: Upsample factor
    :Nsymb: Number of symbols in the filter.
    :sqrt: Generate a root-raised cosine filter.
    """
    N = M * Nsymb
    if sqrt:
        _, h = commsfilters.rrcosfilter(N, alpha, 1., M)
    else:
        _, h = commsfilters.rcosfilter(N, alpha, 1., M)

    return h


def random_symbols(symb_set: list, num: int, seed=None) -> list:
    """Generate random symbols from the provided symbol set.
    """
    if seed is not None:
        random.seed(seed)

    set_N = len(symb_set)
    # Generate a random list of set indices.
    inds = [random.randint(0, set_N-1) for _ in range(num)]
    return [symb_set[i] for i in inds]
