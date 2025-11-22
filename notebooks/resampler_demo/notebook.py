import marimo

__generated_with = "0.18.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    # Add the current directory to make pysignal importable.
    import sys
    import os
    sys.path.append(os.getcwd())
    return


@app.cell
def _():
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
    from matplotlib.gridspec import GridSpec

    matplotlib.use('agg')

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    setup_logging(rootlogger=logging.getLogger(), level='info')
    #import logging_tree; logging_tree.printout()
    return (
        GridSpec,
        QpskMapper,
        Resampler,
        ShapingFilter,
        comms,
        np,
        plottools,
        plt,
    )


@app.cell
def _(QpskMapper, ShapingFilter):
    # Qpsk with Nyquist pulse at 2 samples/symbol.
    # Excess bandwidth 
    alpha = 0.2
    # Samples/symbol
    M = 2
    mapper = QpskMapper()
    shaper = ShapingFilter(alpha=alpha, M=M, sqrt=False, wl=16, fl=15)
    shaper.plot_response(use_quantized=True)
    return mapper, shaper


@app.cell
def _(comms, mapper, plottools, shaper):
    # Generate random symbol at a specified SNR.
    Nsymbs = 1000
    EsN0 = 30
    symbs = comms.random_symbols([0, 1, 2, 3], num=Nsymbs, seed=49013)
    mapped_symbs = mapper.map_symbs(symbs)
    shaped = shaper.process(mapped_symbs, use_quantized=True)

    shaped = comms.addNoise(shaped, EsN0_dB=EsN0, BW=1)
    plottools.plotSpec(shaped, fs=2, avg=True, title='Shaped Input')
    return (shaped,)


@app.cell
def _(Resampler):
    # Create a 32-path resampler for demonstrating demodulator timing error correction.
    resamp = Resampler(num_paths=32,
                       fpass=0.6, fstop=1.0,
                       stopdB=66,
                       fs=64,
                       N=480,
                       wl=16, fl=15)

    resamp.design()
    resamp.plotResponse(quantized=False, axis=[-2, 2, -80, 5]) 
    return (resamp,)


@app.cell(hide_code=True)
def _(mo):
    path = mo.ui.number(start=0, stop=32, value=16, label="Select Path")
    path
    return (path,)


@app.cell
def _(np, path, resamp, shaped):
    # Resample the signal.
    # Fs in: 2 samples/symbol
    # Fs out: 2 samples/symbol
    # Select polyphase path above to adjust demodulator symbol timing.
    # (The constellation should get worse as the path moves away from 16.)
    y = resamp.filter(shaped, 2, 2, path_init=int(path.value), rational_mode=True)

    # Grab decision samples.
    I_samp = np.real(y[25::2])
    Q_samp = np.imag(y[25::2])
    return I_samp, Q_samp, y


@app.cell(hide_code=True)
def _(GridSpec, I_samp, Q_samp, np, path, plottools, plt, y):
    _fig = plt.figure(layout="constrained")

    gs = GridSpec(2, 2, figure=_fig)
    _ax1 = plt.subplot(gs.new_subplotspec((0, 0)))

    plt.sca(_ax1)
    plottools.plotEye(np.real(y[20:]), sps=2, title='Eye (resampled)', addToAxes=True)

    n = np.arange(len(y))
    _range = range(15, 105)

    _ax2 = plt.subplot(gs.new_subplotspec((1, 0)))
    plt.sca(_ax2)
    plottools.plot(n[_range],
                   np.real(y[_range]),
                   title='Shaped QPSK (I) Resampled',
                   marker='.',
                   addToAxes=True)

    _ax3 = plt.subplot(gs.new_subplotspec((0, 1), colspan=2, rowspan=2))
    plottools.plot(I_samp,
                   Q_samp,
                   linestyle='none',
                   marker='.',
                   axis=[-1.5, 1.5, -1.5, 1.5],
                   title=f'Constellation (path={path.value})',
                   addToAxes=True)
    _ax3.set_aspect('equal')
    _fig
    return


if __name__ == "__main__":
    app.run()
