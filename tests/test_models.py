import re

import torch
import torch.nn as nn
import torch.utils.data as data

from avae.dataset import CharDataset
from avae.model import AttentionVae, Block, Decoder, Encoder, GPTConfig

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


wiki = []
with open("/home/simon/code/namegen/data/wikitext-2/train.txt", "r") as f:
    for word in f:
        wiki.extend(clean_word_wiki(word))
text = " ".join(wiki)

maxlen = 20
n_embd = 64
n_head = 4
n_layer = 4
batch_size = 16

dataset = CharDataset(text, maxlen)

config = GPTConfig(
    dataset.vocab_size, maxlen, n_layer=n_layer, n_head=n_head, n_embd=n_embd
)
loader = data.DataLoader(
    dataset, shuffle=False, pin_memory=True, batch_size=batch_size
)


def test_block():
    block = Block(config)
    x, y = next(iter(loader))
    tok_emb = nn.Embedding(config.vocab_size, config.n_embd)(x)

    assert block(tok_emb).shape == torch.Size([batch_size, maxlen - 1, n_embd])


def test_encoder():

    encoder = Encoder(config)
    x, y = next(iter(loader))
    out = encoder(x)

    assert out.shape == torch.Size([batch_size, maxlen - 1, n_embd * 4])


def test_decoder():

    iter_loader = iter(loader)
    x, y = next(iter_loader)
    k, q = next(iter_loader)

    z_k = nn.Embedding(config.vocab_size, config.n_embd)(k)
    z_q = nn.Embedding(config.vocab_size, config.n_embd)(q)

    decoder = Decoder(config)

    out, _ = decoder(x, z_k, z_q)

    assert out.shape == torch.Size([batch_size, maxlen - 1, config.vocab_size])


def test_attention_vae():

    iter_loader = iter(loader)
    x, y, word = next(iter_loader)

    vae = AttentionVae(config)
    out, _, = vae(x, y, word)

    assert out.shape == torch.Size([batch_size, maxlen, config.vocab_size])

    x = torch.zeros([1, maxlen]).long()
    x = torch.repeat_interleave(x, 5, dim=0)
    y = vae.sample(x, 20, temperature=1.0, sample=True, top_k=10,)
    assert y.shape == torch.Size([5, 2 * maxlen])
    # for sent in y:
    #     completion = "".join(
    #         [train_dataset.itos[int(i)] for i in sent]
    #     )
    #     print(completion)

