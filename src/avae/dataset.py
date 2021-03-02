import numpy as np
from torch.utils.data import Dataset
import torch


class CharDataset(Dataset):
    def __init__(
        self,
        data_sources: dict,
        block_size,
        chars=None,
        p_confusion=0.0,
        stoi=None,
        itos=None,
        sourcetoi=None,
        itosource=None,
    ):
        self.p_confusion = p_confusion
        self.word_list = []
        self.source_list = []
        self.block_size = block_size

        self.chars = chars
        self.stoi = stoi
        self.itos = itos
        self.sourcetoi = sourcetoi
        self.itosource = itosource

        _chars = []
        # Gather words
        for source, data in data_sources.items():
            words = data.split(" ")
            words = [word + "0" for word in words]
            self.word_list.extend(words)
            self.source_list.extend([source for _ in words])
            _chars.extend(list(set(data)))

        lengths = np.array([len(word) for word in self.word_list])
        self.unique_word_list = [word[:-1] for word in set(self.word_list)]

        # Get individual characters
        # if chars:
        #     chars = [
        #         char
        #         for char in chars
        #         if char not in (self.source_list + ["_random"])
        #     ]
        # _chars.add
        if not self.chars:
            self.chars = sorted(list(set(_chars)))

        if "0" not in self.chars:
            self.chars = ["0"] + self.chars

        data_size, vocab_size = (
            len(self.word_list),
            len(self.chars) + len(data_sources) + 1,
        )

        # Get indices and lengths of each word
        self.idx_to_word = np.repeat(np.arange(len(self.word_list)), lengths)
        self.idx_to_starting_char = np.concatenate(
            [np.arange(len(word)) for word in self.word_list]
        )

        self.idx_to_source = np.concatenate(
            [
                [source] * len(word)
                for source, word in zip(self.source_list, self.word_list)
            ]
        )

        self.word_lengths = np.repeat(lengths, lengths)
        print("data has %d characters, %d unique." % (data_size, vocab_size))
        print(self.chars)

        # get mappings

        if not stoi:
            self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        if not itos:
            self.itos = {i: ch for i, ch in enumerate(self.chars)}

        # Compute source-specific mappings
        offset = len(self.chars)
        unique_sources = set(self.source_list)
        self.unique_sources = list(unique_sources)

        if not sourcetoi:
            self.sourcetoi = {
                source: i + offset for i, source in enumerate(unique_sources)
            }
            max_idx = max(self.sourcetoi.values())
            self.sourcetoi["_random"] = max_idx + 1
        if not itosource:
            self.itosource = {
                (i + offset): source for i, source in enumerate(unique_sources)
            }
            max_idx = max(self.itosource.keys())
            self.itosource[max_idx + 1] = "_random"

        for k, v in self.itosource.items():
            if k not in self.itos.keys():
                self.itos[k] = v
        for k, v in self.sourcetoi.items():
            if v not in self.stoi.values():
                self.stoi[k] = v

    def __len__(self):
        return len(self.idx_to_word) - self.block_size

    def __getitem__(self, idx):
        # grab a chunk of (block_size + 1) characters from the data

        starting_char = self.idx_to_starting_char[idx]

        if np.random.uniform() > self.p_confusion:
            source_int = self.sourcetoi[self.idx_to_source[idx]]
        else:
            source_int = self.sourcetoi["_random"]

        padding = "0" * (self.block_size - starting_char - 1)
        chunk = (
            padding
            + self.word_list[self.idx_to_word[idx]][: starting_char + 1]
        )

        # encode entire word
        word_padding = "0" * (self.block_size - self.word_lengths[idx])
        word = word_padding + self.word_list[self.idx_to_word[idx]][:-1]

        word_int = [source_int] + [self.stoi[s] for s in word]

        # encode every character to an integer
        dix = [source_int] + [self.stoi[s] for s in chunk]

        """
        arrange data and targets so that the first i elements of x
        will be asked to predict the i-th element of y. Notice that
        the eventual language model will actually make block_size
        individual predictions at the same time based on this data,
        so we are being clever and amortizing the cost of the forward
        pass of the network. So for example if block_size is 4, then
        we could e.g. sample a chunk of text "hello", the integers in
        x will correspond to "hell" and in y will be "ello". This will
        then actually "multitask" 4 separate examples at the same time
        in the language model:
        - given just "h", please predict "e" as next
        - given "he" please predict "l" next
        - given "hel" predict "l" next
        - given "hell" predict "o" next

        In addition, because the DataLoader will create batches of examples,
        every forward/backward pass during traning will simultaneously train
        a LOT of predictions, amortizing a lot of computation. In particular,
        for a batched input of integers X (B, T) where B is batch size and
        T is block_size and Y (B, T), the network will during training be
        simultaneously training to make B*T predictions, all at once! Of
        course,at test time we can paralellize across batch B, but unlike
        during trainingwe cannot parallelize across the time dimension T - we
        have to runa forward pass of the network to recover the next single
        character of the sequence along each batch dimenssion, and repeatedly
        always feed in a next character to get the next one.

        So yes there is a big asymmetry between train/test time of
        autoregressive models. During training we can go B*T at a time with
        every forward pass, but during test time we can only go B at a time, T
        times, with T forward passes.
        """
        x = torch.tensor(dix[:-1], dtype=torch.long)
        dix[0] = 0
        x_no_source = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        word = torch.tensor(word_int, dtype=torch.long)
        return x, x_no_source, y, word
