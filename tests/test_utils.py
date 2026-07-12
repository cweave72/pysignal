import logging
import numpy as np
from numpy.testing import assert_array_equal as np_assert

import pysignal.utils as utils

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def test_upsample():
    M = 4
    r = 5
    x = np.array([1]*r)
    y = utils.upsample(x, M)
    expected = [1] + [0]*(M-1)
    np_assert(y, np.array(expected*r))


def test_bytes_to_bits():

    _bytes = bytearray([0xaa, 0x55])
    bits = utils.bytes_to_bits(_bytes, msb_first=True)
    assert bits == [1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1]

    bits = utils.bytes_to_bits(_bytes, msb_first=False)
    assert bits == [0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0]


def test_bits_to_int():

    word = [1, 1, 0, 0]
    assert utils.bits_to_int(word) == 12
    word = [0, 0, 1, 1]
    assert utils.bits_to_int(word) == 3
    word = [1, 0, 0, 0, 0, 0, 0, 1]
    assert utils.bits_to_int(word) == 129
    word = [1, 1, 1, 1]
    assert utils.bits_to_int(word, signed=True) == -1
    word = [1, 1, 0, 0]
    assert utils.bits_to_int(word, signed=True) == -4
    word = [1, 0, 0, 0, 0, 0, 0, 1]
    assert utils.bits_to_int(word, signed=True) == -127

    word = [1, 1, 0, 0]
    assert utils.bits_to_int(word, msb_first=False) == 3
    word = [0, 0, 1, 1]
    assert utils.bits_to_int(word, msb_first=False) == 12
    word = [0, 0, 0, 0, 0, 0, 0, 1]
    assert utils.bits_to_int(word, msb_first=False) == 128
    word = [0, 1, 1, 1]
    assert utils.bits_to_int(word, msb_first=False, signed=True) == -2
    word = [1, 1, 0, 0]
    assert utils.bits_to_int(word, msb_first=False, signed=True) == 3 
    word = [0, 0, 0, 0, 0, 0, 0, 1]
    assert utils.bits_to_int(word, msb_first=False, signed=True) == -128


def test_fixedpoint_init():
    fp = utils.FixedPoint(16, 14, x_in=1.5, signed=True)
    assert fp.wl == 16
    assert fp.fl == 14
    assert fp.signed is True
    assert fp.x_fp == int(1.5 * 2**14)
    assert abs(fp.x - 1.5) < 2**-14

    fp_u = utils.FixedPoint(8, 4, x_in=7.0, signed=False)
    assert fp_u.signed is False
    assert fp_u.x_fp == int(7.0 * 2**4)

    arr = utils.FixedPoint(8, 4, x_in=np.array([1.0, 2.0, 3.0]))
    assert arr.x_fp.shape == (3,)
    np.testing.assert_allclose(arr.x, np.array([1.0, 2.0, 3.0]), atol=2**-4)


def test_fixedpoint_setters():
    fp = utils.FixedPoint(16, 14, x_in=1.0)
    fp.set_float(1.5)
    assert abs(fp.x - 1.5) < 2**-14

    fp.set_fp(0)
    assert fp.x == 0.0


def test_fixedpoint_parts():
    fp = utils.FixedPoint(16, 14, x_in=1.75)
    assert fp.intpart() == 1
    assert fp.fracpart() == int(0.75 * 2**14)


def test_fixedpoint_getfloat():
    fp = utils.FixedPoint(16, 14, x_in=1.5)
    assert abs(fp.getFloat() - 1.5) < 2**-14


def test_fixedpoint_repr():
    fp = utils.FixedPoint(16, 14, x_in=1.0)
    r = repr(fp)
    assert 'FixedPoint(s(16,14)' in r

    arr = utils.FixedPoint(8, 4, x_in=np.array([1.0, 2.0]))
    r = repr(arr)
    assert 'shape=(2,)' in r


def test_fixedpoint_quant():
    fp = utils.FixedPoint(16, 14, x_in=1.5)
    q = fp.quant(8, 6)
    assert q.wl == 8
    assert q.fl == 6
    assert q.signed is True
    assert abs(q.x - 1.5) < 2**-6


def test_fixedpoint_multiply_default_wlfl():
    a = utils.FixedPoint(16, 14, x_in=1.5)
    b = utils.FixedPoint(16, 14, x_in=2.0)
    c = utils.FixedPoint.multiply(a, b)
    assert c.wl == 32
    assert c.fl == 28
    assert abs(c.x - 3.0) < 0.01


def test_fixedpoint_multiply_custom_wlfl():
    a = utils.FixedPoint(16, 14, x_in=1.5)
    b = utils.FixedPoint(16, 14, x_in=2.0)
    c = utils.FixedPoint.multiply(a, b, wl=20, fl=18)
    assert c.wl == 20
    assert c.fl == 18


def test_fixedpoint_multiply_signedness():
    a = utils.FixedPoint(8, 4, x_in=3.0, signed=False)
    b = utils.FixedPoint(8, 4, x_in=2.0, signed=True)
    c = utils.FixedPoint.multiply(a, b)
    assert c.signed is True

    a = utils.FixedPoint(8, 4, x_in=3.0, signed=False)
    b = utils.FixedPoint(8, 4, x_in=2.0, signed=False)
    c = utils.FixedPoint.multiply(a, b)
    assert c.signed is False


def test_chunker():
    inp = [0, 0, 1, 1, 0, 1]
    overlap = 0
    expected = [
        [0, 0],
        [1, 1],
        [0, 1]]

    for k, chunk in enumerate(utils.chunker(inp, 2, overlap=overlap)):
        logger.debug(f"k={k} chunk={chunk}")
        assert chunk == expected[k]

    inp = [0, 0, 1, 1, 0, 1, 1]
    overlap = 2
    expected = [
        [0, 0, 1],
        [0, 1, 1],
        [1, 1, 0],
        [1, 0, 1],
        [0, 1, 1],
    ]

    for k, chunk in enumerate(utils.chunker(inp, 3, overlap=overlap)):
        logger.debug(f"k={k} chunk={chunk}")
        assert chunk == expected[k]
