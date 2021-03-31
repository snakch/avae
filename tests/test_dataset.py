import pandas as pd
from avae.dataset import CharDataset

latin_words = (
    "Lorem Ipsum is simply dummy text of the printing and typesetting "
    "industry. Lorem Ipsum has been the industry's standard dummy text ever "
    " since the 1500s, when an unknown printer took a galley of type and "
    "scrambled it to make a type specimen book. It has survived not only five "
    "centuries, but also the leap into electronic typesetting, remaining "
    "essentially unchanged. It was popularised in the 1960s with the release "
    "of Letraset sheets containing Lorem Ipsum passages, and more recently "
    "with desktop publishing software like Aldus PageMaker including versions "
    "of Lorem Ipsum."
).split(" ")

lat_df = pd.DataFrame({"word": latin_words})
lat_df["lang"] = "latin"
english_words = (
    "But I must explain to you how all this mistaken idea of denouncing "
    "pleasure and praising pain was born and I will give you a complete "
    "account of the system, and expound the actual teachings of the great "
    "explorer of the truth, the master-builder of human happiness. No one "
    "rejects, dislikes, or avoids pleasure itself, because it is pleasure, "
    "but because those who do not know how to pursue pleasure rationally "
    "encounter consequences that are extremely painful. Nor again is "
    "there anyone who loves or pursues or desires to obtain pain of "
    "itself, because it is pain, but because occasionally circumstances "
    "occur in which toil and pain can procure him some great pleasure. "
    "To take a trivial example, which of us ever undertakes laborious "
    "physical exercise, except to obtain some advantage from it? But "
    "who has any right to find fault with a man who chooses to enjoy "
    "a pleasure that has no annoying consequences, or one who avoids "
    "a pain that produces no resultant pleasure?"
).split(" ")
en_df = pd.DataFrame({"word": english_words})
en_df["lang"] = "english"

df = pd.concat([lat_df, en_df])
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

