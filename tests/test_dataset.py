from avae.dataset import CharDataset
from _make_dataset import make_df

df = make_df()
df["other_source"] = "A"


block_size = 20

expected_x1 = "0" * block_size
expected_y1 = "0" * (block_size - 1) + "L"
expected_word = "0" * (block_size - len("Lorem")) + "Lorem"
expected_sources_1 = ["latin", "A"]
expected_no_sources = ["_random", "_random"]

expected_x2 = "0" * (block_size - 1) + "L"
expected_y2 = "0" * (block_size - 2) + "Lo"
expected_sources_2 = ("latin", "A")


def test_dataset():
    dataset = CharDataset(df, block_size)
    x, x_no_source, y, word = dataset[0]

    x_chars = get_chars(x, dataset)
    x_no_source_chars = get_chars(x_no_source, dataset)
    y_chars = get_chars(y, dataset)
    word_chars = get_chars(word, dataset)

    assert x_chars == expected_x1
    assert x_no_source_chars == expected_x1
    assert y_chars == expected_y1
    assert word_chars == expected_word

    assert get_sources(x, dataset) == expected_sources_1
    assert int(x_no_source[0]) == 0
    assert int(x_no_source[1]) == 0
    assert int(y[0]) == 0
    assert int(y[1]) == 0

    assert get_sources(word, dataset) == expected_sources_1

    assert len(x) == len(y)
    assert len(x) == len(x_no_source)
    assert len(x) == len(word)

    x, x_no_source, y, word = dataset[1]

    x_chars = get_chars(x, dataset)
    x_no_source_chars = get_chars(x_no_source, dataset)
    y_chars = get_chars(y, dataset)
    word_chars = get_chars(word, dataset)

    assert x_chars == expected_x2
    assert x_no_source_chars == expected_x2
    assert y_chars == expected_y2
    assert word_chars == expected_word


def get_chars(x, dataset):
    return "".join([dataset.itos[int(idx)] for idx in x[2:]])


def get_sources(x, dataset):
    sources = []
    for i, (source, source_map) in enumerate(dataset.itosource.items()):
        sources.append(source_map[int(x[i])])
    return sources

