import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
from avae.utils import top_k_logits

# TODO understand IARFlow
# TODO not clear that the structure of the latent space, ie it being a k,q
# encoding space is actually helpfull....
# TODO Have the encoder encode the whole word, as in not try to


class GPTConfig:
    """ base GPT config, params common to all GPT versions """

    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    def __init__(self, vocab_size, block_size, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size
        for k, v in kwargs.items():
            setattr(self, k, v)


class MaskedLinear(nn.Linear):
    """ same as Linear except has a configurable mask on the weights """

    def __init__(
        self, in_features, out_features, bias=True, conditional_size=None
    ):
        super().__init__(in_features, out_features, bias)
        self.register_buffer("mask", torch.ones(out_features, in_features))

        if conditional_size is not None:
            self.cond_op = nn.Linear(conditional_size, out_features)

    def set_mask(self, mask):
        self.mask.data.copy_(torch.from_numpy(mask.astype(np.uint8).T))

    def forward(self, input, cond=None):
        out = F.linear(input, self.mask * self.weight, self.bias)
        if cond is not None:
            out = out + self.cond_op(input)
        return out


class MADE(nn.Module):
    def __init__(
        self,
        input_shape,
        d,
        hidden_size=[512, 512],
        ordering=None,
        conditional_size=None,
    ):
        super().__init__()
        self.input_shape = input_shape
        self.nin = np.prod(input_shape)
        self.nout = self.nin * d
        self.d = d
        self.hidden_sizes = hidden_size
        self.ordering = np.arange(self.nin) if ordering is None else ordering

        # define a simple MLP neural net
        self.net = []
        hs = [self.nin] + self.hidden_sizes + [self.nout]
        for h0, h1 in zip(hs, hs[1:]):
            self.net.extend(
                [
                    MaskedLinear(h0, h1, conditional_size=conditional_size),
                    nn.ReLU(),
                ]
            )
        self.net.pop()  # pop the last ReLU for the output layer
        self.net = nn.ModuleList(self.net)

        self.m = {}
        self.create_mask()  # builds the initial self.m connectivity

    def create_mask(self):
        L = len(self.hidden_sizes)

        # sample the order of the inputs and the connectivity of all neurons
        self.m[-1] = self.ordering
        for layer_idx in range(L):
            self.m[layer_idx] = np.random.randint(
                self.m[layer_idx - 1].min(),
                self.nin - 1,
                size=self.hidden_sizes[layer_idx],
            )

        # construct the mask matrices
        masks = [
            self.m[layer_idx - 1][:, None] <= self.m[layer_idx][None, :]
            for layer_idx in range(L)
        ]
        masks.append(self.m[L - 1][:, None] < self.m[-1][None, :])

        masks[-1] = np.repeat(masks[-1], self.d, axis=1)

        # set the masks in all MaskedLinear layers
        layers = [
            layer
            for layer in self.net.modules()
            if isinstance(layer, MaskedLinear)
        ]
        for layer, m in zip(layers, masks):
            layer.set_mask(m)

    def forward(self, x, cond=None):
        batch_size, seq_len = x.shape[0], x.shape[1]
        out = x.view(batch_size, seq_len, self.nin)
        for layer in self.net:
            if isinstance(out, MaskedLinear):
                out = layer(out, cond=cond)
            else:
                out = layer(out)
        out = out.view(batch_size, seq_len, self.nin, self.d)
        return out


class SelfAttention(nn.Module):
    def __init__(self, config, masked=True, proj_mult=1):

        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.masked = masked

        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)

        self.attn_pdrop = nn.Dropout(config.attn_pdrop)
        self.resid_pdrop = nn.Dropout(config.resid_pdrop)

        self.proj = nn.Linear(config.n_embd, config.n_embd * proj_mult)

        if self.masked:
            self.register_buffer(
                "mask",
                torch.tril(
                    torch.ones(config.block_size, config.block_size)
                ).view(1, 1, config.block_size, config.block_size),
            )
        else:
            self.register_buffer(
                "mask",
                torch.ones(config.block_size, config.block_size).view(
                    1, 1, config.block_size, config.block_size
                ),
            )
        self.n_head = config.n_head

    def forward(self, x, layer_past=None):

        # batch_size, seq_len, n_embd
        B, T, C = x.size()

        # batch_size, n_head, seq_len, n_embd // n_head
        k = (
            self.key(x)
            .view(B, T, self.n_head, C // self.n_head)
            .transpose(1, 2)
        )
        q = (
            self.query(x)
            .view(B, T, self.n_head, C // self.n_head)
            .transpose(1, 2)
        )
        v = (
            self.value(x)
            .view(B, T, self.n_head, C // self.n_head)
            .transpose(1, 2)
        )

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T)
        # -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_pdrop(att)
        # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        y = self.resid_pdrop(self.proj(y))
        return y


class BasicAttention(nn.Module):
    def __init__(self, config):

        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.query = nn.Linear(config.n_embd, config.n_embd)

        self.attn_pdrop = nn.Dropout(config.attn_pdrop)
        self.resid_pdrop = nn.Dropout(config.resid_pdrop)

        self.proj = nn.Linear(config.n_embd, config.n_embd)

        self.register_buffer(
            "mask",
            torch.ones(config.block_size, config.block_size).view(
                1, 1, config.block_size, config.block_size
            ),
        )
        self.n_head = config.n_head

    def forward(self, x, key, value, layer_past=None):

        # batch_size, seq_len, n_embd
        B, T, C = x.size()

        # batch_size, n_head, seq_len, n_embd // n_head
        k = key.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = key.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = (
            self.query(x)
            .view(B, T, self.n_head, C // self.n_head)
            .transpose(1, 2)
        )

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T)
        # -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_pdrop(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        y = self.resid_pdrop(self.proj(y))
        return y


class Block(nn.Module):
    def __init__(self, config, masked=True, embd_mult=1):
        super().__init__()
        self.masked = masked
        self.ln1 = nn.LayerNorm(embd_mult * config.n_embd)
        self.ln2 = nn.LayerNorm(embd_mult * config.n_embd)
        self.attn = SelfAttention(config, masked=masked, proj_mult=embd_mult)
        self.mlp = nn.Sequential(
            nn.Linear(
                config.n_embd * embd_mult, 4 * config.n_embd * embd_mult
            ),
            nn.GELU(),
            nn.Linear(
                4 * config.n_embd * embd_mult, config.n_embd * embd_mult
            ),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class BasicBlock(Block):
    def __init__(self, config):
        super().__init__(config)

        self.attn = BasicAttention(config)

    def forward(self, x, key, value):
        x = x + self.attn(self.ln1(x), key, value)
        x = x + self.mlp(self.ln2(x))
        return x


class AttentionNetwork(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is
        being very defensive: We are separating out all parameters of the model
        into two buckets: those that will experienceweight decay for
        regularization and those that won't (biases, and layernorm/embedding
        weights).We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience
        # regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear,)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name

                if pn.endswith("bias"):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(
                    m, whitelist_weight_modules
                ):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(
                    m, blacklist_weight_modules
                ):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module
        # as not decayed
        no_decay.add("decoder.pos_emb")
        no_decay.add("encoder.pos_emb")
        no_decay.add("smart_encoder.pos_emb")
        no_decay.add("smart_encoder.pos_emb")

        for layer in train_config.freeze_layers:

            if layer not in no_decay:
                no_decay.add(layer)

        decay = set(
            [
                layer
                for layer in decay
                if layer not in train_config.freeze_layers
            ]
        )
        no_decay = set(
            [
                layer
                for layer in no_decay
                if layer not in train_config.freeze_layers
            ]
        )

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay | set(train_config.freeze_layers)
        assert len(inter_params) == 0, (
            "parameters %s made it into both decay/no_decay sets!"
            % (str(inter_params),)
        )
        assert len(param_dict.keys() - union_params) == 0, (
            "parameters %s were not separated into either decay/no_decay set!"
            % (str(param_dict.keys() - union_params),)
        )

        # create the pytorch optimizer object
        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": train_config.weight_decay,
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=train_config.learning_rate,
            betas=train_config.betas,
        )
        return optimizer


class Encoder(AttentionNetwork):
    def __init__(self, config):
        super().__init__(config)

        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Parameter(
            torch.zeros(1, config.block_size, config.n_embd)
        )
        self.drop = nn.Dropout(config.embd_pdrop)

        blocks = [Block(config, masked=False) for _ in range(config.n_layer)]
        self.blocks = nn.Sequential(*blocks)

        # self.block_k = Block(config, masked=False)
        # self.block_q = Block(config, masked=False)

        # self.ln_f_q = nn.LayerNorm(config.n_embd)
        # self.ln_f_k = nn.LayerNorm(config.n_embd)
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)

        self.block_size = config.block_size
        self.apply(self._init_weights)
        # print(
        #     "number of parameters: "
        #     f"{sum(p.numel() for p in self.parameters())}"
        # )

    def forward(self, idx, targets=None):
        b, t = idx.size()
        assert (
            t <= self.block_size
        ), "Can't forward, model block size is exhausted"

        token_embeddings = self.tok_emb(idx)
        position_embeddings = self.pos_emb[:, :t, :]

        x = self.drop(token_embeddings + position_embeddings)
        x = self.blocks(x)
        # x_q = self.block_q(x)
        # x_q = self.ln_f_q(x_q)

        # x_k = self.block_k(x)
        # x_k = self.ln_f_k(x_k)
        # return x_q, x_k

        x = self.ln_f(x)
        return self.head(x)


class Decoder(AttentionNetwork):
    def __init__(self, config):
        super().__init__(config)

        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Parameter(
            torch.zeros(1, config.block_size, config.n_embd)
        )
        self.drop = nn.Dropout(config.embd_pdrop)

        self.masked_blocks = nn.Sequential(
            *[Block(config) for _ in range(config.n_layer)]
        )
        self.ln_f = nn.LayerNorm(config.n_embd)

        self.post_decoder_block = BasicBlock(config)

        self.ln_f_post = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.block_size = config.block_size
        self.apply(self._init_weights)

    def forward(self, idx, z_k, z_v, targets=None):
        b, t = idx.size()
        assert (
            t <= self.block_size
        ), "Can't forward, model block size is exhausted"

        token_embeddings = self.tok_emb(idx)
        position_embeddings = self.pos_emb[:, :t, :]

        x = self.drop(token_embeddings + position_embeddings)
        x = self.masked_blocks(x)
        x = self.ln_f(x)

        x = self.post_decoder_block(x, z_k, z_v)
        x = self.ln_f_post(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                reduction="none",
            )
        return logits, loss


class AttentionVae(AttentionNetwork):
    def __init__(self, config, use_made_prior=True):
        super().__init__(config)

        self.config = config
        # self.n_chars = (n_chars,)
        # self.latent_dim = latent_dim
        # self.seq_len = seq_len

        self.encoder = Encoder(config)
        self.smart_encoder = Encoder(config)

        self.decoder = Decoder(config)
        if use_made_prior:
            self.prior_network = MADE(
                2 * config.n_embd, 2, hidden_size=[512, 512]
            )
        else:
            self.prior_network = nn.Identity()

        print(
            "number of parameters: "
            f"{sum(p.numel() for p in self.parameters())}"
        )

    def forward(self, input, output, word, training=False):

        m_k, log_s_k, m_v, log_s_v = torch.chunk(self.encoder(word), 4, -1)

        z_k = self.reparametrize(m_k, log_s_k)
        z_v = self.reparametrize(m_v, log_s_v)
        if training:

            m_k_smart, log_s_k_smart, m_v_smart, log_s_v_smart = torch.chunk(
                self.smart_encoder(input), 4, -1
            )

            z_k_smart = self.reparametrize(m_k_smart, log_s_k_smart)
            z_v_smart = self.reparametrize(m_v_smart, log_s_v_smart)

            # Very quick and dirty way of killing the encoder
            # z_k = torch.ones_like(z_k)
            # log_s_k = torch.ones_like(log_s_k)
            # log_s_v = torch.ones_like(log_s_v)
            # z_v = torch.ones_like(z_v)
            # m_k_smart = torch.ones_like(m_k_smart)
            # log_s_k_smart = torch.ones_like(log_s_k_smart)
            # m_v_smart = torch.ones_like(m_v_smart)
            # log_s_v_smart = torch.ones_like(log_s_v_smart)

            x, CE = self.decoder(input, z_k, z_v, targets=output)
            CE = CE.view(x.shape[0], -1).sum(1)

            KLD_k = (
                (
                    -log_s_k
                    + 1 / 2 * (torch.pow(m_k, 2) + (torch.exp(2 * log_s_k)))
                    - 0.5
                )
                .sum(-1)
                .mean(1)
            )
            KLD_v = (
                (
                    -log_s_v
                    + 1 / 2 * (torch.pow(m_v, 2) + (torch.exp(2 * log_s_v)))
                    - 0.5
                )
                .sum(-1)
                .mean(1)
            )
            # Smart encoder sees partial word instead of full word and attempts
            # to map to the same place in latent space
            # Idea is that the encoder should have global stylistic info and
            # the smart encoder is a network that attempts to capture as much
            # of that as possible with partial context. Is this actually a good
            # idea?
            # Maybe we're actually encouraging to not encode global stuff
            # Also could try some neural style transfer stuff?
            # Maybe the encoding path should get to see some style thing (like
            # a language) but not the decoding path
            # Still question of how to go english language -> english name.
            # Ie how do I transfer knowledge about words
            smart_encoder_guess = F.mse_loss(z_k_smart, z_k, reduction="none")
            smart_encoder_guess += F.mse_loss(z_v_smart, z_v, reduction="none")
            smart_encoder_guess = smart_encoder_guess.sum(-1).mean(1)

            KLD_k_smart = (
                (
                    -log_s_k_smart
                    + 1
                    / 2
                    * (
                        torch.pow(m_k_smart, 2)
                        + (torch.exp(2 * log_s_k_smart))
                    )
                    - 0.5
                )
                .sum(-1)
                .mean(1)
            )
            KLD_v_smart = (
                (
                    -log_s_v_smart
                    + 1
                    / 2
                    * (
                        torch.pow(m_v_smart, 2)
                        + (torch.exp(2 * log_s_v_smart))
                    )
                    - 0.5
                )
                .sum(-1)
                .mean(1)
            )

            z = torch.cat([z_k, z_v], axis=2)
            out = self.prior_network(z)
            mu, log_std = out.chunk(2, dim=-1)
            log_std = torch.tanh(log_std)
            mu, log_std = mu.squeeze(-1), log_std.squeeze(-1)
            eps = z * torch.exp(log_std) + mu
            prior_log_prob = (
                (-0.5 * np.log(2 * np.pi) + log_std - 0.5 * eps ** 2)
                .sum(-1)
                .mean(1)
            )
            KLD = KLD_k + KLD_v
            KLD_smart = KLD_k_smart + KLD_v_smart
            loss = CE + KLD + KLD_smart + smart_encoder_guess - prior_log_prob
            return (
                x,
                OrderedDict(
                    loss=loss,
                    ce=CE,
                    kld=KLD,
                    kld_smart=KLD_smart,
                    prior_log_prob=prior_log_prob,
                    smart_encoder_guess=smart_encoder_guess,
                ),
            )

        x, _ = self.decoder(input, z_k, z_v)

        return x

    def reparametrize(self, m, log_s, sample_dim=None):
        if sample_dim:
            raise NotImplementedError()
        else:
            eps = torch.randn_like(m, device=m.device)
        z = m + torch.exp(log_s) * eps
        return z

    def sample_latent(self, context, seq_len, device, method="smart"):

        if method == "smart":
            m_k_smart, log_s_k_smart, m_v_smart, log_s_v_smart = torch.chunk(
                self.smart_encoder(context), 4, -1
            )

            z_k = self.reparametrize(m_k_smart, log_s_k_smart)
            z_v = self.reparametrize(m_v_smart, log_s_v_smart)

        elif method == "naive":
            z_k = torch.randn(
                [context.shape[0], seq_len, self.config.n_embd], device=device
            )
            z_v = torch.randn(
                [context.shape[0], seq_len, self.config.n_embd], device=device
            )
        else:
            raise NotImplementedError("Method not recognised")
        z = torch.cat([z_k, z_v], dim=2)
        for i in range(seq_len):
            mu, log_std = self.prior_network(z)[:, i].chunk(2, dim=-1)
            log_std = torch.tanh(log_std)
            mu, log_std = mu.squeeze(-1), log_std.squeeze(-1)
            z[:, i, :] = (z[:, i, :] - mu) * torch.exp(-log_std)
        z_k, z_v = torch.chunk(z, 2, dim=-1)
        return z_k, z_v

    def sample(
        self,
        x,
        steps,
        temperature=1.0,
        sample=False,
        top_k=None,
        method="smart",
    ):
        """
        take a conditioning sequence of indices in x (of shape (b,t)) and
        predict the next token inthe sequence, feeding the predictions back
        into the model each time. Clearly the sampling has quadratic complexity
        unlike an RNN that is only linear, and has a finite context window of
        block_size, unlike an RNN that has an infinite context window.
        """

        with torch.no_grad():
            assert x.size(1) >= self.config.block_size

            if method == "naive":
                z_k, z_v = self.sample_latent(
                    x[:, -self.config.block_size :],
                    self.config.block_size,
                    x.device,
                    method=method,
                )
            # print(logits.view(logits.size(0), -1).mean(1))

            for k in range(steps):

                # TODO sample only once  per word now

                # x_cond = (
                #     x
                #     if x.size(1) <= self.config.block_size
                x_cond = x[:, -self.config.block_size :]

                z_k, z_v = self.sample_latent(
                    x[:, -self.config.block_size :],
                    self.config.block_size,
                    x.device,
                    method=method,
                )

                # )  # crop context if needed

                logits, _ = self.decoder(x_cond, z_k, z_v)
                # pluck the logits at the final step and scale by temperature
                logits = logits[:, -1, :] / temperature
                # optionally crop probabilities to only the top k options
                if top_k is not None:
                    logits = top_k_logits(logits, top_k)
                # apply softmax to convert to probabilities
                probs = F.softmax(logits, dim=-1)
                # sample from the distribution or take the most likely
                if sample:
                    ix = torch.multinomial(probs, num_samples=1)
                else:
                    _, ix = torch.topk(probs, k=1, dim=-1)
                # append to the sequence and continue
                x = torch.cat((x, ix), dim=1)

            return x

    def save(self, path):
        torch.save(self, path)

    @classmethod
    def load(cls, path):
        model = torch.load(path)
        return model
