import marimo

__generated_with = "0.18.0"
app = marimo.App(width="columns")


@app.cell(column=0)
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
    import pysignal.spectrum as spec
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
    sps = 2
    mapper = QpskMapper()
    shaper = ShapingFilter(alpha=alpha, M=sps, sqrt=False, wl=16, fl=15)
    shaper.plot_response(use_quantized=True)
    return mapper, shaper, sps


@app.cell
def _(comms, mapper, np, plottools, shaper, sps):
    # Generate random symbols at a specified SNR.
    Nsymbs = 500
    EsN0 = 60
    symbs = comms.random_symbols([0, 1, 2, 3], num=Nsymbs, seed=49013)
    mapped_symbs = mapper.map_symbs(symbs)
    shaped = shaper.process(mapped_symbs, use_quantized=True)
    shaped *= np.sqrt(2)

    shaped = comms.addNoise(shaped, EsN0_dB=EsN0, BW=1)
    plottools.plotSpec(shaped,
                       fs=sps,
                       avg=False,
                       xlabel='Freq Normalized to Symbol Rate',
                       title='Shaped Input')
    return (shaped,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Designing the Resamping prototype filter
    When designing the prototype filter to be used as a resampler, we consider the input sample rate to the filter as normalized to the symbol rate, that is, we think of the input sample rate in termal of *samples/symbol*.

    The prototype filter is designed to pass the input signal and suppress images created due to the interpolation operation, which occur at integer multiples of the input sample rate.  The resampling filter is internally a P-path polyphase interpolater. As such, the filter must be designed relative to the interpolated sample-rate (i.e. $F_{sin}*P$).

    Estimate for filter length, N:

    $N\approx\frac{F_{sin}*P}{\Delta_f}\cdot\frac{Atten_{dB}}{22}$

    where $\Delta_f=f_{stop}-f_{pass}$
    """)
    return


@app.cell
def _(Resampler, sps):
    # Create a P-path resampler for demonstrating demodulator timing error correction.
    P = 32
    # Interpolated input sample rate for prototype filter: sps*P
    _fs = sps*P 
    resamp = Resampler(num_paths=P,
                       fpass=0.6, fstop=1.5,
                       rippledB=0.1,
                       stopdB=63,
                       fs=_fs,
                       N=224,
                       wl=16, fl=15)

    resamp.design()
    resamp.plotResponse(quantized=False, axis=[-2, 2, -80, 5])
    return P, resamp


@app.cell(column=1, hide_code=True)
def _(mo):
    mo.md(r"""
    ##View Resampler Input and Output Spectrum
    """)
    return


@app.cell(hide_code=True)
def _(P, mo, sps):
    # Demonstrate the spectral effects of resampling.
    F = mo.ui.number(start=sps, stop=sps*P, step=.1, value=sps*4, label="Fs_out (samples/symbol)")
    F
    return (F,)


@app.cell
def _(F, P, np, shaped, sps):
    x_interp = np.zeros(P*len(shaped), dtype=complex)
    x_interp[::P] = shaped
    sps_out = F.value
    print(f"Resample for output sample rate sps_out={F.value} (sps_in={sps})")
    return sps_out, x_interp


@app.cell(hide_code=True)
def _(P, plottools, plt, resamp, shaped, sps, sps_out, x_interp):
    _fig, (_ax1, _ax2, _ax3) = plt.subplots(3, 1)
    _axis = [-sps_out/2, sps_out/2, -80, 5]

    plt.sca(_ax1)
    plottools.plotSpec(x_interp,
                       fs=sps*P,
                       axis=_axis,
                       label='input',
                       addToAxes=True)
    #Overlay prototype response.
    plottools.plotSpec(resamp.b, fs=sps*P, Nfft=4096,
                       addToAxes=True,
                       axis=_axis,
                       linestyle="-",
                       color='red',
                       xlabel='Freq Normalized to Symbol Rate',
                       title='Resampler Input Spectrum', label='resampler')
    plt.legend(loc='upper right')

    plt.sca(_ax2)
    y4 = resamp.filter(shaped, sps, sps_out)
    plottools.plotSpec(
        y4,
        fs=sps_out,
        wind='blackman',
        addToAxes=True,
        axis=_axis,
        xlabel='Freq Normalized to Symbol Rate',
        title=f'Resampled Output Spectrum (fs_in={sps}, fs_out={sps_out})')

    plt.sca(_ax3)
    _symb_smpl_start = int(60*sps_out)
    _symb_smpl_stop = int(68*sps_out)
    #plottools.plotTime(y4[50*sps_out:500+8*int(np.ceil(sps_out))],
    plottools.plotTime(y4[_symb_smpl_start:_symb_smpl_stop],
                       marker='.',
                       title="Resampled output (over 8 symbol times)",
                       axis=[None, None, -2, 2],
                       addToAxes=True)
    _ax3.minorticks_on()
    _ax3.grid(which='minor', linestyle=':')
    plt.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ##Using the Resampler to correct QPSK timing offset
    """)
    return


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
    I_samp = np.real(y[21::2])
    Q_samp = np.imag(y[21::2])
    return I_samp, Q_samp, y


@app.cell(hide_code=True)
def _(GridSpec, I_samp, Q_samp, np, path, plottools, plt, y):
    _fig = plt.figure(layout="constrained")

    gs = GridSpec(2, 2, figure=_fig)
    _ax1 = plt.subplot(gs.new_subplotspec((0, 0)))

    plt.sca(_ax1)
    plottools.plotEye(np.real(y[20:]), sps=2, title='Eye (resampled)', addToAxes=True)

    n = np.arange(len(y))
    _range = range(15, 205)

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
                   axis=[-2, 2, -2, 2],
                   title=f'Constellation (path={path.value})',
                   addToAxes=True)
    _ax3.set_aspect('equal')
    _fig
    return


if __name__ == "__main__":
    app.run()
