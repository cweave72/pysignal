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
    """Wraps a fixed-point value (or array of values) at a given wl/fl.

    Works equally as a scalar wrapper or an array wrapper -- x_in/x_fp may
    be a plain number or a numpy array (or anything array-like), and every
    method below operates elementwise in the array case.
    """

    def __init__(self, wl, fl, x_in=None, signed=True):
        self.wl = wl
        self.fl = fl
        self.signed = signed

        if x_in is not None:
            self.x_fp = quantize(x_in, self.wl, self.fl,
                    signed=self.signed, out='scaledint')
            self.x = self.getFloat()

    def set_float(self, x_in):
        self.x_fp = quantize(x_in, self.wl, self.fl,
                signed=self.signed, out='scaledint')
        self.x = self.getFloat()

    def set_fp(self, x_fp):
        if np.isscalar(x_fp):
            self.x_fp = int(x_fp)
        else:
            self.x_fp = np.asarray(x_fp).astype(np.int64)
        self.x = self.getFloat()

    def intpart(self):
        return self.x_fp >> self.fl

    def fracpart(self):
        return self.x_fp - (self.intpart()*(2**(self.fl)))

    def getFloat(self):
        return self.x_fp / 2**(self.fl)

    def quant(self, wl, fl, signed=None):
        """ Quantize to new fixed point representation. Return new instance.
        signed defaults to this instance's own signed-ness if not given.
        """
        tmp = self.x_fp / 2**(self.fl)
        signed = self.signed if signed is None else signed
        return FixedPoint(wl, fl, x_in=tmp, signed=signed)

    @staticmethod
    def multiply(a, b, wl=None, fl=None):
        """Multiply two FixedPoint instances. Returns a new FixedPoint.

        By default the output format is (a.wl+b.wl, a.fl+b.fl) which
        preserves full precision with no overflow. Pass wl/fl to override.
        The result is signed if either input is signed.
        """
        out_wl = a.wl + b.wl if wl is None else wl
        out_fl = a.fl + b.fl if fl is None else fl
        out_signed = a.signed or b.signed
        result = a.x * b.x
        return FixedPoint(out_wl, out_fl, x_in=result, signed=out_signed)

    def __repr__(self):
        shape = '' if np.isscalar(self.x) else f' shape={np.shape(self.x)}'
        sign = 's' if self.signed else 'u'
        return f"FixedPoint({sign}({self.wl},{self.fl}), x={self.x}{shape})"


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


def upsample(inp: np.ndarray, M: int) -> np.ndarray:
    """Zero-pads the input array by 1:M.
    """
    y = np.zeros(len(inp) * M, dtype=inp.dtype)
    y[::M] = inp
    return y


def bytes_to_bits(inp: bytes, msb_first=True):
    """Returns an array of 1's and 0's from bytearray provided.
    """
    bits = []
    shifts = list(range(8))

    if msb_first:
        shifts = shifts[::-1]

    for b in inp:
        for shift in shifts:
            bit = b >> shift & 0x1
            bits.append(bit)
    return bits


def bits_to_int(inp: list, msb_first=True, signed=False):
    """Convert a list of bits to an integer.
    :inp: Input list [1, 0, 1, ...]
    :msb_first: Assumes inp[0] corresponds to the MSB of the word.
    :signed: Assume word is a signed value.
    """
    result = 0
    bit_weights = [2**k for k in range(len(inp))]

    # If signed, negate the weight of the msb (the sign bit) for 2s complement
    if signed:
        bit_weights[-1] = -bit_weights[-1]

    if msb_first:
        bits = inp[::-1]
    else:
        bits = inp

    for k, b in enumerate(bits):
        if b == 1:
            result += bit_weights[k]

    return result


def chunker(iterable, chunkSize, overlap=0):
    """Yields chunks of chunkSize samples from the iterable, with optional
    overlap across returned chunks.
    """
    it = iter(iterable)

    # first iteration, fill chunk with chunkSize amount of stuff.
    chunk = []
    for _ in range(chunkSize):
        chunk.append(next(it))

    yield chunk

    # Save the rightmost overlap samples for the next iteration.
    save = chunk[-overlap:] if overlap != 0 else []

    numNewItems = chunkSize - overlap
    while True:
        chunk = []
        try:
            for _ in range(numNewItems):
                chunk.append(next(it))
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
