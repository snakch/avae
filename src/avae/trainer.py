import math
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import torch
from fuzzywuzzy import process
from torch.utils.data import DataLoader
from tqdm import tqdm


@dataclass
class TrainerConfig:
    max_epochs: int = 10
    batch_size: int = 64
    learning_rate: float = 3e-4
    betas: tuple = (0.9, 0.95)
    grad_norm_clip: float = 1.0
    weight_decay: float = 0.1
    lr_decay: bool = False
    warmup_tokens: int = 1000
    final_tokens: int = 1000
    ckpt_path: str = None
    num_workers: int = 0
    sample_freq: int = 1000
    freeze_layers: list = ()


class Trainer:
    def __init__(
        self,
        model,
        train_dataset,
        test_dataset,
        config,
        log_nearest_words=False,
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = config
        self.log_nearest_words = log_nearest_words

        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            # self.model = torch.nn.DataParallel(self.model).to(self.device)
            self.model = self.model.to(self.device)

        if hasattr(model, "stoi"):
            # assert model.stoi == train_dataset.stoi
            # assert model.itos == train_dataset.itos
            # assert model.sourcetoi == train_dataset.sourcetoi
            # assert model.itosource == train_dataset.itosource
            train_dataset.stoi = model.stoi
            train_dataset.itos = model.itos
            train_dataset.sourcetoi = model.sourcetoi
            train_dataset.itosource = model.itosource

        else:
            model.stoi = train_dataset.stoi
            model.itos = train_dataset.itos
            model.sourcetoi = train_dataset.sourcetoi
            model.itosource = train_dataset.itosource

        self.losses = defaultdict(list)

    def save_checkpoint(self):
        raw_model = (
            self.model.module if hasattr(self.model, "module") else self.model
        )
        print(f"Saving at {self.config.ckpt_path}")
        torch.save(raw_model.state_dict(), self.config.ckpt_path)

    def train(self):
        model, config = self.model, self.config
        raw_model = model.module if hasattr(self.model, "module") else model
        optimizer = raw_model.configure_optimizers(config)

        # best_loss = float("inf")
        self.tokens = 0

        if self.test_dataset:
            test_losses = self.test()
            for key, val in test_losses.items():
                self.losses[key].append(val)

        for epoch in range(config.max_epochs):
            self.run_epoch(epoch, optimizer, "train")
            # if self.test_dataset is not None:
            # test_loss =\ run_epoch("test")

            # for k, v in loss_dict.items():
            # self.losses[k].extend(v)

            if self.test_dataset:
                test_losses = self.test()
                for key, val in test_losses.items():
                    self.losses[key].append(val)

        return self.losses

    def run_epoch(self, epoch, optimizer, split):
        is_train = split == "train"
        self.model.train(is_train)
        data = self.train_dataset if is_train else self.test_dataset
        loader = DataLoader(
            data,
            shuffle=True,
            pin_memory=True,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
        )

        epoch_losses = defaultdict(list)
        pbar = (
            tqdm(enumerate(loader), total=len(loader))
            if is_train
            else enumerate(loader)
        )
        for it, (x, x_no_source, y, word) in pbar:
            # print(x)
            x = x.to(self.device)
            x_no_source = x_no_source.to(self.device)
            y = y.to(self.device)
            word = word.to(self.device)
            with torch.set_grad_enabled(is_train):
                logits, loss_dict = self.model(
                    x, x_no_source, y, word, training=True
                )

                loss = loss_dict["loss"].mean()
                for k, v in loss_dict.items():
                    if v is not None:
                        self.losses[k].append(v.mean().item())

            if is_train:
                self.model.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm(
                    self.model.parameters(), self.config.grad_norm_clip
                )
                optimizer.step()

                if self.config.lr_decay:
                    self.tokens += (y >= 0).sum()
                    if self.tokens < self.config.warmup_tokens:
                        lr_mult = float(self.tokens) / float(
                            max(1, self.config.warmup_tokens)
                        )
                    else:
                        progress = float(
                            self.tokens - self.config.warmup_tokens
                        ) / float(
                            max(
                                1,
                                self.config.final_tokens
                                - self.config.warmup_tokens,
                            )
                        )
                        lr_mult = max(
                            0.1, 0.5 * (1.0 + math.cos(math.pi * progress))
                        )
                    lr = self.config.learning_rate * lr_mult
                    for param_group in optimizer.param_groups:
                        param_group["lr"] = lr
                else:
                    lr = self.config.learning_rate

                pbar.set_description(
                    f"epoch {epoch + 1}) iter {it}: train_loss "
                    f"{loss.item():.7f}, lr = {lr:e}"
                )
                if it % self.config.sample_freq == 0:
                    print(f"Iteration {it} samples:")
                    self.print_samples()

        if not is_train:
            test_losses = {
                "test_" + key: np.mean(val)
                for key, val in epoch_losses.items()
            }
            # print(f"Test loss: {test_losses['test_loss']}")
            return test_losses
        return epoch_losses

    def test(self):

        loss_dict = self.run_epoch(0, optimizer=None, split="test")
        return loss_dict

    def print_samples(self):

        # create an empty word context.
        source = np.random.choice(self.train_dataset.unique_sources)
        source_int = self.train_dataset.sourcetoi[source]

        context = "0" * (self.model.config.block_size - 1)
        x = torch.tensor(
            [source_int] + [self.train_dataset.stoi[s] for s in context],
            dtype=torch.long,
        )[None, ...].to(self.device)
        x = torch.repeat_interleave(x, 5, dim=0)

        # sample some words
        with torch.no_grad():
            y = self.model.sample(
                x, 20, temperature=1.0, sample=True, top_k=10,
            )
        for gen_word in y:
            # sample = decode_word(
            #     gen_word.cpu().numpy(),
            #     self.train_dataset.itos,
            #     length=self.model.config.block_size,
            # )

            completion = "".join(
                [self.train_dataset.itos[int(i)] for i in gen_word[1:]]
            )
            sample = completion[self.model.config.block_size - 1 :].split("0")[
                0
            ]
            sample += " " * (self.model.config.block_size - len(sample))
            if self.log_nearest_words:

                # Find the nearest term in the training set.
                match = process.extract(
                    sample, self.train_dataset.unique_word_list, limit=1
                )[0][0]

                sample += f"\t closes match: {match}"

            print(f"Sample: {sample} ")  # \t Closest match is {match}")
