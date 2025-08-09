"""
A collection of python signal utilities.

Author: cdw
"""

import logging
import inspect
import code
import numpy as np

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class FixedPoint(object):
    def __init__(self, wl, fl, x_in=None):
        self.wl = wl
        self.fl = fl

        if x_in:
            self.x_fp = quantize(x_in, self.wl, self.fl, 
                    out='scaledint')
            self.x = self.getFloat()

    def set_float(self, x_in):
        self.x_fp = quantize(x_in, self.wl, self.fl, out='scaledint')
        self.x = self.getFloat()

    def set_fp(self, x_fp):
        self.x_fp = int(x_fp)
        self.x = self.getFloat()

    def intpart(self):
        return self.x_fp >> self.fl

    def fracpart(self):
        return self.x_fp - (self.intpart()*(2**(self.fl))) 

    def getFloat(self):
        return float(self.x_fp) / 2**(self.fl)

    def quant(self, wl, fl):
        """ Quantize to new fixed point representation. Return new instance.
        """
        tmp = float(self.x_fp) / 2**(self.fl)
        return FixedPoint(wl, fl, x_in=tmp)


def csvRead(filename, oversamp=1):
    """ Reads a csv file which is assumed to contain samples.
    A single column of data represents a real signal, while 2 columns will be
    returned as a comples array. Set oversamp option to specify an oversampling
    factor.
    """

    with open(filename) as fd:
        csv = fd.readlines()

    numcols = len(csv[0].split(','))
    assert 0 < numcols <= 2,  "Unsupported number of columns for csvRead."

    array = []
    for line in csv:
        z = line.split(',')
        if numcols == 1:
            array.append(float(z[0]))
        else:
            re, im = z
            array.append(float(re) + 1j*float(im))

    if oversamp > 1:
        array = array[:-1:oversamp]

    return np.array(array)


def quantize(x, wl, fl, mode=np.round, signed=True, out='float'):
    """ Quantizes the float value x to an integer with specified wordlength and
    fractionlength.

    :x:     The input or array to be quantized.
    :wl:    Quantization wordlength.
    :fl:    Quantization fractionlength.
    :mode:  Callable which controls how the input is quantized (i.e. use
            round() or ceil() when quantizing.
    :return: Returns the quantized value/array as a float or scaled integer.
    """

    assert out in ['float', 'scaledint'], "Check 'out' parameter"

    if signed:
        minval = -2.0**(wl-fl-1)
        maxval = 2**(wl-fl-1) - 2**(-fl)
    else:
        minval = 0
        maxval = 2**(wl-fl) - 2**(-fl)

    try:
        len(x)
    except Exception:
        # Convert scalar to an array for further processing.
        x = np.array([x])

    # Convert to ndarray if x is a list (it has a len() method).
    if not isinstance(x, np.ndarray):
        x = np.array(x)

    # Quantize input value(s).
    xq = mode(x*(2**fl))/(2**fl)
    # Now clamp to max and min range.
    x_clip = np.clip(xq, minval, maxval)

    if out == 'scaledint':
        x_clip = x_clip*(2**fl)
        x_clip = x_clip.astype(int)

    if len(x_clip) == 1:
        x_clip = x_clip[0]

    return x_clip


def chunker(iterable, chunkSize, overlap=0):
    """Yields chunks of chunkSize samples from the iterable, with optional
    overlap across returned chunks.
    """
    it = iter(iterable)

    # first iteration, fill chunk with chunkSize amount of stuff.
    chunk = []
    for _ in range(chunkSize):
        chunk.append(it.next())

    yield chunk

    # Save the rightmost overlap samples for the next iteration.
    save = chunk[-overlap:] if overlap != 0 else []

    numNewItems = chunkSize - overlap
    while True:
        chunk = []
        try:
            for _ in range(numNewItems):
                chunk.append(it.next())
            out = save + chunk
            yield out
            save = out[-overlap:] if overlap != 0 else []
        except StopIteration:
            if chunk:
                yield save + chunk
            break


def enterShell():
    """ Starts an interactive console.
    """
    # Get the caller's frame.
    cf = inspect.currentframe().f_back
    # Copy the globals from that frame.
    ns = cf.f_globals.copy()
    # Add caller's locals.
    ns.update(cf.f_locals.copy())
    # Start the interactive shell.
    code.interact("*interactive*", local=ns)
