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


#def nyquistFilt(alpha, M, Nsymb, sqrt=False):
#
#    N = M * Nsymb
#
#    h = np.zeros(N)
#    if (not sqrt):
#        for n in range(N):
#            m = float(n)-((N/2)-1)
#            factor1 = np.sinc(m/M)
#            factor2 = np.cos(alpha*np.pi*m/M)/(1.-np.power(2.*alpha*m/M, 2))
#            h[n] = (1./M)*factor1*factor2
#
#        infs = np.isfinite(h)
#        inf_inds = np.where(infs == False)
#        h[inf_inds] = (np.pi/4)*np.sinc(1./(2*alpha))
#
#    else:
#        for n in range(N):
#            m = float(n)-((N/2)-1)
#            num = (4.*alpha*(m/M)*np.cos(np.pi*(m/M)*(1.+alpha)) + 
#                   np.sin(np.pi*(m/M)*(1.-alpha)))
#            den = (1. - np.power(4.*alpha*m/M, 2)) * np.pi*m/M
#            h[n] = (1./M)*(num/den)
#
#        infs = np.isfinite(h)
#        inf_inds, = np.where(infs == False)
#        
#        if len(inf_inds) == 3:
#            c = inf_inds[1]
#            lr = np.array([inf_inds[0], inf_inds[2]])
#        else:
#            c = inf_inds[0]
#            lr = None
#
#        logger.debug(f"infs={infs}, inf_inds={inf_inds}")
#
#        ## Fill in the 0 element.
#        h[c] = (1./M)*(1. - alpha + 4.*alpha/np.pi)
#        # Fill in the +/- 1/(4*alpha)
#        if lr is not None:
#            scale = (1./M)*alpha/(np.sqrt(2))
#            h[lr] = (scale*(1.+2./np.pi)*np.sin(np.pi/(4.*alpha)) +
#                                 (1.-2./np.pi)*np.cos(np.pi/(4.*alpha)))
#
#    logger.debug(f"h={h}")
#    return h
