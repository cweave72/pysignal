import logging
import numpy as np
import commpy.filters as commsfilters

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


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


def nyquistFilt(alpha, M, Nsymb, sqrt=False):

    N = M * Nsymb
    if sqrt:
        _, h = commsfilters.rrcosfilter(N, alpha, 1., M)
    else:
        _, h = commsfilters.rcosfilter(N, alpha, 1., M)

    logger.debug(f"sqrt={sqrt}; h={h}")
    return h
