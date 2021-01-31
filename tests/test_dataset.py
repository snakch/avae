print(help("modules"))
import torch

print(dir(avae))

from avae.avae.dataset import CharDataset


text = (
    "Lorem Ipsum is simply dummy text of the printing and typesetting "
    "industry. Lorem Ipsum has been the industry's standard dummy text ever "
    " since the 1500s, when an unknown printer took a galley of type and "
    "scrambled it to make a type specimen book. It has survived not only five "
    "centuries, but also the leap into electronic typesetting, remaining "
    "essentially unchanged. It was popularised in the 1960s with the release "
    "of Letraset sheets containing Lorem Ipsum passages, and more recently "
    "with desktop publishing software like Aldus PageMaker including versions "
    "of Lorem Ipsum."
)
block_size = 20

expected_x1 = "0" * block_size
expected_y1 = "0" * (block_size - 1) + "L"
expected_word = "0" * (block_size - len("Lorem")) + "Lorem"

expected_x2 = "0" * (block_size - 1) + "L"
expected_y2 = "0" * (block_size - 2) + "Lo"


def test_dataset():
    dataset = CharDataset(text, block_size)
    x, y, word = dataset[0]
    x = "".join([dataset.itos[int(idx)] for idx in x])
    y = "".join([dataset.itos[int(idx)] for idx in y])
    word = "".join([dataset.itos[int(idx)] for idx in word])
    assert x == expected_x1
    assert y == expected_y1
    assert word == expected_word

    assert len(x) == len(y)
    assert len(x) == len(word)

    x, y, word = dataset[1]
    x = "".join([dataset.itos[int(idx)] for idx in x])
    y = "".join([dataset.itos[int(idx)] for idx in y])
    word = "".join([dataset.itos[int(idx)] for idx in word])

    assert x == expected_x2
    assert y == expected_y2
    assert word == expected_word
