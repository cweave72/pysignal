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
