import marimo

__generated_with = "0.18.0"
app = marimo.App(width="columns")


@app.cell
def _():
    import marimo as mo
    return (mo,)


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
    return QpskMapper, ShapingFilter, comms, np, plottools, plt


@app.cell
def _(QpskMapper, ShapingFilter):
    # Qpsk with Nyquist pulse at 2 samples/symbol.
    # Excess bandwidth 
    alpha = 0.3
    # Samples/symbol
    sps = 2
    mapper = QpskMapper()
    # Note we set sqrt=False to emulate the output of a matched filter. If we were actually transmitting these symbols to a receiving matched filter, we would generate root-raised cosine shaped pulses instead.
    shaper = ShapingFilter(alpha=alpha, M=sps, sqrt=False, wl=16, fl=15)
    shaper.plot_response(use_quantized=True)
    return mapper, shaper, sps


@app.cell
def _(np):
    def M2M4_snr(sig):
        """SNR Estimator using M2 and M4 moments.
        """
        # Grab decision samples.
        I_samp = np.real(sig[20::2])
        Q_samp = np.imag(sig[20::2])

        N = len(I_samp)

        # Accumulate moments.
        M2_raw = 0.
        M4_raw = 0.
        for I, Q in zip(I_samp, Q_samp):
            Y = (I**2) + (Q**2)
            Z = Y**2
            M2_raw += Y
            M4_raw += Z

        M2 = M2_raw/N 
        M4 = M4_raw/N 

        discriminant = 2*(M2**2) - M4

        signal_power = np.sqrt(discriminant)
        noise_power = M2 - signal_power

        if noise_power < 0:
            print(f"noise power is < 0 {noise_power=}")
            return 99
        else:
            snr_linear = signal_power / noise_power
            snr_db = 10 * np.log10(snr_linear)
            return snr_db
    return (M2M4_snr,)


@app.cell
def _(comms, mapper, mo, np, plottools, plt, shaper, sps):
    # Generate random symbols at a specified SNR.
    Nsymbs = 1000
    EsN0 = 10
    symbs = comms.random_symbols([0, 1, 2, 3], num=Nsymbs, seed=49013)
    mapped_symbs = mapper.map_symbs(symbs)
    shaped = shaper.process(mapped_symbs, use_quantized=True)
    # Scale the shaped samples to make the constellation points at (+-1,+-1)
    shaped *= np.sqrt(2)

    # Note we only add noise over BW=1 (i.e. the symbol rate) instead of BW=2 which would be the sample rate (samples/symbol). This is so we show the true Es/N0 that would exist at the output of a matched filter and decision at 1 sample/symbol. We are already generating symbols as if they had come out of the matched filter (sqrt=False above when generating the shaped symbold).
    noisy_shaped = comms.addNoise(shaped, EsN0_dB=EsN0, sps=1)

    _fig0, (_ax1, _ax2) = plt.subplots(2,1)

    plt.sca(_ax1)
    plottools.plotSpec(noisy_shaped,
                       fs=sps,
                       avg=True,
                       Nfft=512,
                       normAvg=True,
                       axis=[-1, 1, -(EsN0+10), 10],
                       xlabel='Freq Normalized to Symbol Rate',
                       title='Shaped Input', addToAxes=True)
    plt.sca(_ax2)
    plottools.plotTime(noisy_shaped[:250], marker='.', xlabel='sample', addToAxes=True)

    # Grab decision samples.
    I_samp = np.real(noisy_shaped[20::2])
    Q_samp = np.imag(noisy_shaped[20::2])

    _fig1, _ax3 = plottools.plot(I_samp,
                                 Q_samp,
                                 linestyle='none',
                                 marker='.',
                                 axis=[-2, 2, -2, 2],
                                 title=f'Constellation')
    _ax3.set_aspect('equal')
    plt.tight_layout()
    mo.hstack([_fig0, _fig1], justify="start", widths="equal")
    return (shaped,)


@app.cell
def _(M2M4_snr, comms, mo, np, plottools, plt, shaped):
    # Sweep Es/N0 and check the M2M4 estimator against the true value.
    Ntrials = 50
    EsN0_range = np.arange(0, 21, 1)
    EsN0_est = np.zeros(len(EsN0_range))
    trial_errors = np.zeros((len(EsN0_range), Ntrials))

    for _idx, _EsN0 in enumerate(EsN0_range):
        _trial_sum = 0.
        for _trial in range(Ntrials):
            _noisy = comms.addNoise(shaped, EsN0_dB=_EsN0, sps=1)
            _est = M2M4_snr(_noisy)
            trial_errors[_idx, _trial] = _est - _EsN0
            _trial_sum += _est
        EsN0_est[_idx] = _trial_sum / Ntrials

    mean_error = trial_errors.mean(axis=1)
    std_error = trial_errors.std(axis=1)

    _fig2, _ax4 = plottools.plot(EsN0_range,
                                 EsN0_est,
                                 marker='.',
                                 xlabel='True Es/N0 (dB)',
                                 ylabel='Estimated Es/N0 (dB)',
                                 title='M2M4 SNR Estimator')
    _ax4.plot(EsN0_range, EsN0_range, linestyle='--', color='gray')
    plt.tight_layout()

    _fig3, _ax5 = plt.subplots()
    for _idx, _EsN0 in enumerate(EsN0_range):
        _ax5.plot([_EsN0] * Ntrials, trial_errors[_idx], linestyle='none',
                  marker='.', color='C0', alpha=0.3)
    _ax5.errorbar(EsN0_range, mean_error, yerr=std_error, fmt='o', color='red',
                  capsize=3, label='Mean ± Std')
    _ax5.set_xlabel('True Es/N0 (dB)')
    _ax5.set_ylabel('Estimation Error (dB)')
    _ax5.set_title('M2M4 Per-Trial Estimation Error')
    _ax5.grid('on')
    _ax5.legend()
    plt.tight_layout()

    mo.hstack([_fig2, _fig3], justify="start", widths="equal")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
