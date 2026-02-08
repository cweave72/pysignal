import marimo

__generated_with = "0.18.0"
app = marimo.App(width="medium")


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
    import pysignal.plottools as plottools
    from pysignal import setup_logging

    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.use('agg')

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    setup_logging(rootlogger=logging.getLogger(), level='info')
    return comms, np, plottools, plt, utils


@app.cell
def _(np, plt):
    def radian_labels(labels):
        # Format each label for radian display
        rad_out = []
        for rad in labels:
            #rad = degrees * (2*np.pi/360)
            # Convert angles > pi to negative angle equivalent
            if rad > np.pi:
                rad -= 2*np.pi
            rad_out.append(rad)
        labels = [f"{x/np.pi:.2f}π" for x in rad_out]
        return labels

    def unit_circle_plot(arg):
        """Plots a complex value on a unit circle.
        """
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        ax.quiver(0, 0, np.real(arg), np.imag(arg), scale=2, color='blue')
        ax.set_rlim(0, 1) 
        ax.grid(True)
        # Get the x-tick positions (returns in radians)
        label_positions = ax.get_xticks()
        # Convert to a list since we want to change the type of the elements
        labels = list(label_positions)
        rad_labels = radian_labels(labels)
        ax.set_xticks(labels)
        ax.set_xticklabels(rad_labels) 
        return fig, ax

    def calc_aliased_freq(f, fs):   
        if np.abs(f) < fs/2:
            # No aliasing
            return f"{f}"
        elif f >= fs/2:
            return f"{-fs/2 + (f - fs/2)} (aliased)"
        else:
            return f"{fs/2 + (f + fs/2)} (aliased)"
    return calc_aliased_freq, unit_circle_plot


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Aliasing Demo
    The cells below demostrate how aliasing occurs in the time domain and how this appears in the frequency domain.

    We use a tone to demonstrate:

    complex: $\large x[n]=e^{\frac{j2 \pi f_c}{f_s}n}$   <br>
    real: $\large x[n]=sin({\frac{j2 \pi f_{c}}{f_s}n})$

    where: <br>
    $f_c$ : Frequency of the tone, Hz <br>
    $f_s$ : Sample rate, Hz

    In the demo, we sweep a complex tone across positive and negative frequencies using a sample rate of $f_s=100$ Hz.
    Aliasing will occur when the selected tone frequency crosses the Nyquist (a.k.a *folding*) frequency of $\vert\frac{f_s}{2}\vert=50$ Hz.

    ## Nyquist Zones
    Nyquist Zones are sections of spectrum in spans of $f_s$ Hz which have the
    potential to alias back into baseband (i.e. Nyquist zone 1). In a real system,
    Nyquist Zone 1 is our only observable spectrum. What we observe in our
    time-domain samples is always a representaion of the energy which exists in
    Nyquist zone 1. We need to be aware that higher zones exist to the extent that
    any energy in those zones can undesirably alias back into zone 1 if we are not
    careful.

    Observe that when you increase the tone frequency above 50 Hz ($f_s/2$), you will observe the tone energy which existed in the negative-side of Nyquist Zone 2 walk into our zone 1 window. This
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    # Sample rate of signal
    Fs = 100
    Nsamples = 10000

    tone_opts = mo.ui.dropdown(options=["complex", "real"], value="complex", label="Choose signal type")
    fc = mo.ui.number(start=-Fs, stop=Fs, step=1, value=4, label='Tone frequency (Hz)')
    mo.vstack([fc, tone_opts])
    return Fs, Nsamples, fc, tone_opts


@app.cell
def _(Fs, Nsamples, calc_aliased_freq, comms, fc, np, tone_opts, utils):
    # Create base signal, a complex exponential.
    n = np.arange(Nsamples)

    if tone_opts.value == "complex":
        sig = np.exp(1j*(2*np.pi*fc.value/Fs)*n)
    elif tone_opts.value == "real":
        sig = np.sin((2*np.pi*fc.value/Fs)*n)

    SNR = 40 # Add some noise to make the spectrum look like an digitizer output.
    sig = comms.addNoise(sig, SNR, 1)

    # Interpolate signal by 1:2 to expose 2nd Nyquist zone in the spectrum.
    sig_up2 = utils.upsample(sig, 2)

    freq_str = calc_aliased_freq(fc.value, Fs)
    return freq_str, sig, sig_up2


@app.cell(hide_code=True)
def _(Fs, fc, freq_str, mo, np, plottools, sig, sig_up2, unit_circle_plot):
    _fig1, _ax1 = plottools.plotTime(sig[0:50], xlabel='Sample', marker='.')
    _fig2, _ax2 = plottools.plotSpec(sig_up2, xlabel='Freq (Hz)', fs=2*Fs, axis=[-Fs, Fs, -80, 20])
    _ax2.vlines([-Fs/2, Fs/2], -80, 20, linestyles='dashed', color='green')
    _ax1.set_title(f'Sampled signal (fc={freq_str} Hz)')
    _ax2.set_title(f"Sampled Spectrum (fc={fc.value}; Fs={Fs})")
    _ax2.text(-25, 10, "Nyquist Zone 1", color='green')
    _ax2.text(55, 10, "Nyquist Zone 2", color='red')
    _ax2.text(-97, 10, "Nyquist Zone 2", color='red')

    _fig3, _ax3 = unit_circle_plot(np.exp(1j*2*np.pi*fc.value/Fs))
    omega = f"\N{GREEK SMALL LETTER OMEGA}"
    rad_per_sample = fc.value * 2*np.pi/Fs
    rad_per_sample_str = f"{rad_per_sample/np.pi:.3}π" if rad_per_sample < np.pi else f"{(rad_per_sample-2*np.pi)/np.pi:.3}π"
    _ax3.set_title(f"Complex Plane View (fc={fc.value} Hz, {omega}={rad_per_sample_str} rad/sample)")

    mo.hstack([_fig1, _fig2, _fig3], widths=[1, 1, 1])
    return


if __name__ == "__main__":
    app.run()
