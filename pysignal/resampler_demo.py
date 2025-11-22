#!/bin/env python
import numpy as np
import logging

import pysignal.utils as utils
import pysignal.comms as comms
import pysignal.plottools as plottools

from pysignal.comms import QpskMapper, ShapingFilter
from pysignal.filter.resampler import Resampler
from pysignal import setup_logging

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')
logger = logging.getLogger(__name__)



def main():

    # Qpsk at 2 samples/symbol.
    mapper = QpskMapper()
    shaper = ShapingFilter(alpha=0.2, M=2, sqrt=False, wl=16, fl=15)
    shaper.plot_response(use_quantized=True)

    Nsymbs = 640
    EsN0 = 30
    symbs = comms.random_symbols([0, 1, 2, 3], num=Nsymbs, seed=49002)
    mapped_symbs = mapper.map_symbs(symbs)
    #logger.debug(f"mapped_symbs={mapped_symbs}")
    shaped = shaper.process(mapped_symbs, use_quantized=True)

    shaped = comms.addNoise(shaped, EsN0_dB=EsN0, BW=2)

    #plottools.plotSpec(shaped, fs=2, avg=True, title='Shaped Input')

    resamp = Resampler(num_paths=32,
                       fpass=0.6, fstop=1.0,
                       stopdB=66,
                       fs=64,
                       N=480,
                       wl=16, fl=15)

    resamp.design(stop_1overf=True)
    resamp.plotResponse(quantized=False, axis=[-2, 2, -80, 5]) 

    # Resample the signal.
    y = resamp.filter(shaped, 2, 2, path_init=16, rational_mode=True)

    plot_range = int(len(shaped)*2*.05)
    n = np.arange(plot_range)

    _, (ax1, ax2) = plt.subplots(2, 1)
    plt.sca(ax1)
    plottools.plot(n,
                   np.real(y[:plot_range]),
                   title='Shaped QPSK (I) Resampled',
                   marker='.',
                   addToAxes=True)

    plt.sca(ax2)
    plottools.plot(n,
                   np.imag(y[:plot_range]),
                   title='Shaped QPSK (Q) Resampled',
                   marker='.',
                   color='red',
                   addToAxes=True)
    plt.tight_layout()

    I_samp = np.real(y[25::2])
    Q_samp = np.imag(y[25::2])
    plottools.plot(I_samp, Q_samp, linestyle='none', marker='.',
                   title='Constellation')

    plottools.plotEye(np.real(shaped), sps=2, title='Eye (input)')
    plottools.plotEye(np.real(y), sps=2, title='Eye (resampled)')


    plt.ion()
    plt.show()
    utils.enterShell()


if __name__ == "__main__":
    setup_logging(rootlogger=logging.getLogger(), level='debug')
    #import logging_tree; logging_tree.printout()
    main()
