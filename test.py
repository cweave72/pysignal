#!/bin/env python

import logging

import pysignal.utils as utils
import pysignal.comms as comms
import pysignal.plottools as plottools

from pysignal import setup_logging
from pysignal.filter.filter import Filter

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

def test_filter():

    filt = Filter(fpass=1, 
                  fstop=1.7, 
                  fs=10, 
                  stopdB=60, 
                  wl=16, fl=15, 
                  w=[1, 10])
    filt.design()
    filt.plotResponse(quantized=True) 

    logger.info(f"filt.b={filt.b}")
    logger.info(f"filt={filt}")


def test_nyqFilter(alpha, sqrt):

    Fs = 4
    y = comms.nyquistFilt(alpha=alpha, M=Fs, Nsymb=12, sqrt=sqrt)

    plottools.plotTime(y,
                       marker='.',
                       title=f"Nyquist Pulse sqrt={sqrt}, alpha={alpha:.2f}, Fs={Fs}")

    plottools.plotSpec(y,
                       Nfft=1024,
                       fs=Fs,
                       title=f'Nyquist Pulse spectrum (sqrt={sqrt})',
                       xlabel='freq normalized to symbol rate',
                       ylabel='dB',
                       axis=[-Fs/2, Fs/2, -60, 5])


def main():

    test_filter()
    test_nyqFilter(alpha=0.5, sqrt=True)
    test_nyqFilter(alpha=0.5, sqrt=False)

    plt.ion()
    plt.show()
    utils.enterShell()

if __name__ == "__main__":
    setup_logging(rootlogger=logging.getLogger(), level='debug')
    #import logging_tree; logging_tree.printout()
    main()
