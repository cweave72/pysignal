#!/bin/env python
import numpy as np
import logging

import pysignal.utils as utils
import pysignal.comms as comms
import pysignal.plottools as plottools

from pysignal.comms import QpskMapper, ShapingFilter
from pysignal import setup_logging

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')
logger = logging.getLogger(__name__)



def main():

    rrcos = True
    mapper = QpskMapper()
    shaper = ShapingFilter(alpha=0.2, M=4, sqrt=rrcos, wl=16, fl=15)
    shaper.plot_response(use_quantized=True)

    symbs = comms.random_symbols([0, 2], num=64, seed=49001)
    mapped_symbs = mapper.map_symbs(symbs)
    logger.debug(f"mapped_symbs={mapped_symbs}")
    shaped = shaper.process(mapped_symbs, use_quantized=True)

    n = np.arange(len(shaped))

    _, (ax1, ax2) = plt.subplots(2, 1)
    plt.sca(ax1)
    plottools.plot(n, np.real(shaped), title='Shaped QPSK (I)', marker='.', addToAxes=True)

    plt.sca(ax2)
    plottools.plot(n, np.imag(shaped), title='Shaped QPSK (Q)', marker='.', color='red', addToAxes=True)
    plt.tight_layout()

    plt.ion()
    plt.show()
    utils.enterShell()


if __name__ == "__main__":
    setup_logging(rootlogger=logging.getLogger(), level='debug')
    #import logging_tree; logging_tree.printout()
    main()
