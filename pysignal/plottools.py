"""plottools.py

A collection of matplotlib plotting utility functions.
"""
import numpy as np

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import pysignal.spectrum as spec

import logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def plot(x, y, addToAxes=False, **kwargs):
    """ Plots two vectors as a time-domain plot.
    :param x: The x-axis vector.
    :type x: Array-type

    :param y: The y-axis vector.
    :type y: Array-type

    :param addToAxes: Add the plot axis to an existing axis (for overlaying
    plots).
    :param addToAxes: boolean

    Following kwargs are accepted.

    xlabel : x-axis label,
    ylabel : y-axis label,
    title  : plot title
    axis   : Axis limits -> [xmin, xmax, ymin, ymax]
    grid   : Turn grid on.

    Any other kwargs are passed to the plot command.
    """

    xlabel = kwargs.pop('xlabel', None)
    ylabel = kwargs.pop('ylabel', None)
    title  = kwargs.pop('title', None)
    axis   = kwargs.pop('axis', None)
    grid   = kwargs.pop('grid', 'on')

    if addToAxes:
        # Get current figure and axis.
        fig = plt.gcf()
        ax = plt.gca()
    else:
        # Create new figure and axis.
        fig, ax = plt.subplots()

    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)

    ax.grid(grid)

    if any(np.iscomplex(y)):
        color = kwargs.pop('color', None)
        ax.plot(x, np.real(y), **kwargs)
        ax.plot(x, np.imag(y), color='red', **kwargs)
    else:
        ax.plot(x, y, **kwargs)

    if axis is not None:
        ax.axis(axis)

    return fig, ax


def plotTime(y, fs=1, units=None, addToAxes=False, **kwargs):
    """ Plots a time series.
    :param y: The y-axis vector.
    :type y: Array-type

    :param fs: The sample rate, in Hz.
    :type fs: numeric.

    :param units: Time axis units.
    :type units: string ['s', 'ms', 'us', 'ns']

    :param addToAxes: Add the plot axis to an existing axis (for overlaying
    plots).
    :param addToAxes: boolean

    Following kwargs are accepted.

    xlabel : x-axis label,
    ylabel : y-axis label,
    title  : plot title
    axis   : Axis limits -> [xmin, xmax, ymin, ymax]
    grid   : Turn grid on.

    Any other kwargs are passed to the plot command.
    """

    if units is None:
        timescaler = 1.0
    else:
        timescaler = {'s': 1.0, 'ms': 1e3, 'us': 1e6, 'ns': 1e9}[units]

    Ts = timescaler/float(fs)
    time_x = np.arange(len(y)) * Ts
    fig, ax = plot(time_x, y, addToAxes=addToAxes, **kwargs)
    return fig, ax


def plotSpec(y, fs=1, units='hz', Nfft=None, avg=False, 
        addToAxes=False, **kwargs):
    """ Generate a spectral plot from the signal, y.
    :param y: The signal vector.
    :type y: Array-type

    :param fs: The sample rate, in Hz.
    :type fs: numeric.

    :param units: Frequency axis units.
    :type units: string ['hz', 'khz', 'mhz']

    :param Nfft: The FFT size to use.
    :type Nfft: int

    :param avg: Generate an averaged spectrum.
    :type avg: boolean

    :param addToAxes: Add the plot axis to an existing axis (for overlaying
                      plots).
    :param addToAxes: boolean

    Following kwargs are accepted.

    xlabel : x-axis label,
    ylabel : y-axis label,
    title  : plot title
    axis   : Axis limits -> [xmin, xmax, ymin, ymax]
    grid   : Turn grid on.

    Any other kwargs are passed to the plot command.
    """
    fs = float(fs)

    xlabel = kwargs.pop('xlabel', None)
    ylabel = kwargs.pop('ylabel', 'dB')
    title  = kwargs.pop('title', None)
    axis   = kwargs.pop('axis', None)
    grid   = kwargs.pop('grid', 'on')

    if addToAxes:
        # Get current figure and axis.
        fig = plt.gcf()
        ax = plt.gca()
    else:
        # Create new figure and axis.
        fig, ax = plt.subplots()

    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)

    ax.grid(grid)

    if avg:
        X, f = spec.computeAvgSpec(y, fs=fs, Nfft=Nfft, wind='blackman')
    else:
        X, f = spec.computeSpec(y, fs=fs, Nfft=Nfft)

    Xnorm = np.absolute(X/np.amax(X))**2
    Xnorm_dB = 10*np.log10(Xnorm)
    f_scale = {'hz': 1.0, 'khz': 1000.0, 'mhz': 1e6}[units]

    ax.plot(f/f_scale, np.fft.fftshift(Xnorm_dB), **kwargs)

    if axis is not None:
        ax.axis(axis)
    else:
        ax.axis([-fs/(2*f_scale), fs/(2*f_scale), None, None])

    return fig, ax


def plotEye(y, sps, num_symbs=2, addToAxes=False, **kwargs):
    """ Plots an eye-diagram.
    :param y: The y-axis vector.
    :type y: Array-type

    :param sps: Samples per symbol in y
    :type sps: numeric.

    Following kwargs are accepted.

    xlabel : x-axis label,
    ylabel : y-axis label,
    title  : plot title
    axis   : Axis limits -> [xmin, xmax, ymin, ymax]
    grid   : Turn grid on.

    Any other kwargs are passed to the plot command.
    """
    xlabel = kwargs.pop('xlabel', 'sample')
    ylabel = kwargs.pop('ylabel', None)
    title  = kwargs.pop('title', 'Eye Diagram')
    axis   = kwargs.pop('axis', [0, sps*num_symbs-1, None, None])
    grid   = kwargs.pop('grid', 'on')

    if addToAxes:
        # Get current figure and axis.
        fig = plt.gcf()
        ax = plt.gca()
    else:
        # Create new figure and axis.
        fig, ax = plt.subplots()

    samples_per_fold = sps * num_symbs
    # Number of sections to fold.
    Nsections = len(y)//samples_per_fold

    folded = np.empty((Nsections, samples_per_fold))

    for k in range(Nsections):
        folded[k, :] = y[samples_per_fold*k:samples_per_fold*k+samples_per_fold]

    # Create new figure and axis.
    xvals = np.arange(samples_per_fold)
    ax.plot(xvals, folded.T)

    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)

    ax.grid(grid)
    ax.axis(axis)

    return fig, ax


