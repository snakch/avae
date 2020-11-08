from pathlib import Path
import numpy as np
import torch
from tqdm.auto import tqdm
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import re
import torch.utils.data as data
from dataclasses import dataclass

import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import math
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data.dataloader import DataLoader

from model import AttentionVae, GPTConfig
from dataset import CharDataset
from trainer import TrainerConfig, Trainer


PROJECT_DIR = Path("/home/simon/code/namegen/")
DATA_DIR = PROJECT_DIR / "data"
WIKI_DIR = PROJECT_DIR / "wikitext-2"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def clean_word(word):
    word = word.replace("\n", "").lower()
    return word


REGEX = re.compile("[^a-z-'\-]")


def validate(word):
    if REGEX.findall(word):
        return False
    if len(word) <= 2:
        return False
    return True


def clean_word_wiki(line):
    line = line.replace("\n", "").lower()
    line = line.split(" ")
    line = [word for word in line if validate(word)]
    return line


french_names = []
with open(DATA_DIR / "french_names_no_accent.txt", "r") as f:
    for word in f:
        french_names.append(clean_word(word))


all_names = french_names

raw_text = " ".join(all_names)
maxlen = len(max(all_names, key=len)) + 1
chars = sorted(list(set(raw_text)))
mapping = dict((c, i + 1) for i, c in enumerate(chars))
mapping["0"] = 0
inverse_mapping = dict((k, v) for v, k in mapping.items())


def pad_word(word, length=20):
    before = "0"
    after = "0" * ((length - len(word)) + 1)
    word = before + word + after
    return word


def depad_word(word, length=20):
    word = word.replace("0", "")
    return word


def encode_word(word, mapping, length=20):
    word = pad_word(word, length=length)
    return [mapping[char] for char in word]


def decode_word(word, inverse_mapping, length=20):
    return "".join(
        [depad_word(inverse_mapping[char], length=length) for char in word]
    )


for name in french_names:
    assert name == decode_word(encode_word(name, mapping), inverse_mapping)


def vec2word(vec, mapping):
    #     int_vec = np.rint(np.squeeze(vec) * len(mapping)).astype(int)
    inverse_mapping = dict((k, v) for v, k in mapping.items())
    words = []
    for word in vec:
        words.append(decode_word(word, inverse_mapping))
    return words


all_data = np.array(
    [encode_word(name, mapping, length=maxlen) for name in all_names]
)
for orig, word in zip(all_names, vec2word(all_data, mapping)):
    assert orig == word

train_dataset = CharDataset(raw_text, maxlen)
# for i in range(40):
#     print(train_dataset[i][0].size())

config = GPTConfig(
    train_dataset.vocab_size, maxlen, n_layer=2, n_head=4, n_embd=64
)
vae = AttentionVae(config)

tconf = TrainerConfig(
    max_epochs=2,
    batch_size=64,
    learning_rate=6e-4,
    lr_decay=True,
    warmup_tokens=512 * 20,
    final_tokens=2 * len(train_dataset) * maxlen,
    num_workers=4,
)
trainer = Trainer(vae, train_dataset, None, tconf)
trainer.train()
