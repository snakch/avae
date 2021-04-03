import re

import torch
import torch.nn as nn
import torch.utils.data as data
from avae.dataset import CharDataset
from avae.model import AttentionVae, Block, Decoder, Encoder, GPTConfig

from _make_dataset import make_df

df = make_df()
REGEX = re.compile("[^a-z-'\-]")

block_size = 20
n_embd = 64
n_head = 4
n_layer = 4
batch_size = 16
n_source_types = (3,)

dataset = CharDataset(df, block_size)

config = GPTConfig(
    len(dataset.stoi),
    dataset.seq_len,
    n_layer=n_layer,
    n_head=n_head,
    n_embd=n_embd,
    n_source_types=n_source_types,
)
loader = data.DataLoader(
    dataset, shuffle=False, pin_memory=True, batch_size=batch_size
)


def test_block():
    block = Block(config)
    x, _, y, word = next(iter(loader))

    tok_emb = nn.Embedding(config.vocab_size, config.n_embd)(x)
    block(tok_emb).shape
    assert block(tok_emb).shape == torch.Size(
        [batch_size, dataset.seq_len, n_embd]
    )


def test_encoder():

    encoder = Encoder(config)
    x, _, y, word = next(iter(loader))
    out = encoder(x)

    assert out.shape == torch.Size([batch_size, dataset.seq_len, n_embd * 4])


def test_decoder():

    iter_loader = iter(loader)
    x, _, y, _ = next(iter_loader)
    k, q, _, _ = next(iter_loader)

    z_k = nn.Embedding(config.vocab_size, config.n_embd)(k)
    z_q = nn.Embedding(config.vocab_size, config.n_embd)(q)

    decoder = Decoder(config)

    out, _, _, _, _ = decoder(x, z_k, z_q)

    assert out.shape == torch.Size(
        [batch_size, dataset.seq_len, config.vocab_size]
    )


def test_attention_vae():

    iter_loader = iter(loader)
    x, x_no_source, y, word = next(iter_loader)

    vae = AttentionVae(config)
    out, _, = vae(x, x_no_source, y, word, training=True)
    # import pdb

    # pdb.set_trace()
    assert out.shape == torch.Size(
        [batch_size, dataset.seq_len, config.vocab_size]
    )

    x = torch.zeros([1, dataset.seq_len]).long()
    x = torch.repeat_interleave(x, 5, dim=0)
    y = vae.sample(x, 20, temperature=1.0, sample=True, top_k=10,)
    assert y.shape == torch.Size(
        [5, 2 * dataset.block_size + len(n_source_types)]
    )
    # for sent in y:
    #     completion = "".join(
    #         [train_dataset.itos[int(i)] for i in sent]
    #     )
    #     print(completion)
