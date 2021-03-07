"""test suite for utils
"""

from avae.utils import pad_word, depad_word, encode_word, decode_word

WORD_LENGTH = 5
WORD = "A" * WORD_LENGTH
LENGTH = 20
PADDED_WORD = "0" + WORD + "0" * (LENGTH - WORD_LENGTH + 1)


MAPPING = {"0": 0, "A": 1}
INVERSE_MAPPING = {0: "0", 1: "A"}
INDEX_WORD = [1] * WORD_LENGTH
INDEX_WORD = [0] + INDEX_WORD + [0] * (LENGTH - WORD_LENGTH + 1)


def test_pad_word():
    padded = pad_word(WORD, LENGTH)
    assert padded == PADDED_WORD


def test_depad_word():
    depadded = depad_word(PADDED_WORD, LENGTH)
    assert depadded == WORD


def test_encode_word():
    encoded = encode_word(WORD, MAPPING, LENGTH)
    assert encoded == INDEX_WORD


def test_decode_word():
    decoded = decode_word(INDEX_WORD, INVERSE_MAPPING, LENGTH)
    assert decoded == WORD
