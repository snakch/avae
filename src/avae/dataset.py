import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class CharDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        block_size: int,
        chars=None,
        p_confusion=0.0,
        stoi=None,
        itos=None,
        sourcetoi=None,
        itosource=None,
    ):
        self.p_confusion = p_confusion
        self.word_list = []
        self.block_size = block_size

        self.chars = chars
        self.stoi = stoi
        self.itos = itos
        self.sourcetoi = sourcetoi
        self.itosource = itosource

        self.seq_len = self.block_size + df.shape[1] - 1

        # Drop words which are too long
        df = df.loc[df["word"].str.len() < block_size]

        # Get source type information
        self.source_types = [col for col in df.columns if col != "word"]
        unique_sources = {
            source: list(sorted(df[source].unique()))
            for source in self.source_types
        }
        for k, v in unique_sources.items():
            if "_random" not in v:
                unique_sources[k] = ["_random"] + v
        self.n_source_type = tuple(
            [len(unique) for unique in unique_sources.values()]
        )
        # n_sources = [len(unique_sources[source]) for source in source_types]

        if chars is not None:
            chars = sorted(list(chars))

            # Drop wors which have illegal characters
            regex = fr"[^{chars}]*$"
            df.loc[df["word"].str.contains(regex, regex=True)]

        else:
            chars = list(sorted(set(df["word"].sum())))

        self.chars = chars

        self.word_list = df["word"].tolist()

        if "0" in self.chars:
            self.chars.remove("0")

        self.chars = ["0"] + self.chars

        lengths = (df["word"].str.len() + 1).tolist()

        # Get indices and lengths of each word
        self.idx_to_row = np.repeat(np.arange(len(self.word_list)), lengths)
        self.idx_to_starting_char = np.concatenate(
            [np.arange(len(word) + 1) for word in self.word_list]
        )

        print("characters: ")
        print(self.chars)

        # Get mappings
        if not self.stoi:
            self.stoi = get_source_to_index_map(self.chars)
        if not self.itos:
            self.itos = get_index_to_source_map(self.chars)

        if not self.sourcetoi:
            self.sourcetoi = {}
            for source, values in unique_sources.items():
                self.sourcetoi[source] = get_source_to_index_map(values)

        if not self.itosource:
            self.itosource = {}
            for source, values in unique_sources.items():
                self.itosource[source] = get_index_to_source_map(values)
        self.df = df

    def __len__(self):
        return len(self.idx_to_row) - self.block_size

    def _get_source_prefix(self, idx):
        prefix = []
        row = self.idx_to_row[idx]
        for source_type, source_map in self.sourcetoi.items():
            if np.random.uniform() > self.p_confusion:

                source = self.df.iloc[row][source_type]
                prefix.append(self.sourcetoi[source_type][source])
            else:
                prefix.append(self.sourcetoi[source_type]["_random"])

        return prefix

    def __getitem__(self, idx):
        # grab a chunk of (block_size + 1) characters from the data

        starting_char = self.idx_to_starting_char[idx]
        word = self.df.iloc[self.idx_to_row[idx]].word
        end_padded_word = word + "0"

        source_prefix = self._get_source_prefix(idx)
        padding = "0" * (self.block_size - starting_char)
        chunk = padding + end_padded_word[: starting_char + 1]
        chunk_int = source_prefix + [self.stoi[s] for s in chunk]

        no_source_chunk_int = [0] * len(source_prefix) + [
            self.stoi[s] for s in chunk
        ]

        # encode entire word
        word_padding = "0" * (self.block_size - len(word))
        padded_word = word_padding + word
        word_int = source_prefix + [self.stoi[s] for s in padded_word]

        # encode every character to an integer

        x = torch.tensor(chunk_int[:-1], dtype=torch.long)

        x_no_source = torch.tensor(no_source_chunk_int[:-1], dtype=torch.long)
        y = torch.tensor(no_source_chunk_int[1:], dtype=torch.long)
        word = torch.tensor(word_int, dtype=torch.long)
        # return x, x_no_source, y, word
        return x, x_no_source, y, word


def get_source_to_index_map(source):
    return {ch: i for i, ch in enumerate(source)}


def get_index_to_source_map(source):
    return {i: ch for i, ch in enumerate(source)}
