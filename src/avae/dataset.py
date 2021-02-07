import numpy as np
from torch.utils.data import Dataset
import torch


class CharDataset(Dataset):
    def __init__(self, data, block_size, chars=None):

        self.word_list = data.split(" ")
        self.word_list = [word + "0" for word in self.word_list]
        lengths = np.array([len(word) for word in self.word_list])

        self.unique_word_list = [word[:-1] for word in set(self.word_list)]

        if not chars:
            chars = sorted(list(set(data)))
            if "0" not in chars:
                chars = ["0"] + chars
        if "0" not in chars:
            chars = ["0"] + chars

        data_size, vocab_size = len(data), len(chars)

        self.idx_to_word = np.repeat(np.arange(len(self.word_list)), lengths)
        self.idx_to_starting_char = np.concatenate(
            [np.arange(len(word)) for word in self.word_list]
        )

        self.word_lengths = np.repeat(lengths, lengths)
        print("data has %d characters, %d unique." % (data_size, vocab_size))

        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        self.block_size = block_size
        self.vocab_size = vocab_size

    def __len__(self):
        return len(self.idx_to_word) - self.block_size

    def __getitem__(self, idx):
        # grab a chunk of (block_size + 1) characters from the data

        starting_char = self.idx_to_starting_char[idx]
        padding = "0" * (self.block_size - starting_char)
        chunk = (
            padding
            + self.word_list[self.idx_to_word[idx]][: starting_char + 1]
        )

        # encode entire word
        word_padding = "0" * (self.block_size - self.word_lengths[idx] + 1)
        word = word_padding + self.word_list[self.idx_to_word[idx]][:-1]

        word_int = [self.stoi[s] for s in word]

        # encode every character to an integer
        dix = [self.stoi[s] for s in chunk]

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
        y = torch.tensor(dix[1:], dtype=torch.long)
        word = torch.tensor(word_int, dtype=torch.long)
        return x, y, word
