import math
from dataclasses import dataclass
from collections import OrderedDict, defaultdict

import numpy as np
import torch
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


class Trainer:
    def __init__(self, model, train_dataset, test_dataset, config):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = config

        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            # self.model = torch.nn.DataParallel(self.model).to(self.device)
            self.model = self.model.to(self.device)

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

        def run_epoch(split):
            is_train = split == "train"
            model.train(is_train)
            data = self.train_dataset if is_train else self.test_dataset
            loader = DataLoader(
                data,
                shuffle=True,
                pin_memory=True,
                batch_size=config.batch_size,
                num_workers=config.num_workers,
            )

            epoch_losses = defaultdict(list)
            pbar = (
                tqdm(enumerate(loader), total=len(loader))
                if is_train
                else enumerate(loader)
            )
            for it, (x, y, word) in pbar:
                x = x.to(self.device)
                y = y.to(self.device)
                word = word.to(self.device)
                with torch.set_grad_enabled(is_train):
                    logits, loss_dict = model(x, y, word)

                    loss = loss_dict["loss"].mean()
                    for k, v in loss_dict.items():
                        epoch_losses[k].append(v.mean().item())

                if is_train:
                    model.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm(
                        model.parameters(), config.grad_norm_clip
                    )
                    optimizer.step()

                    if config.lr_decay:
                        self.tokens += (y >= 0).sum()
                        if self.tokens < config.warmup_tokens:
                            lr_mult = float(self.tokens) / float(
                                max(1, config.warmup_tokens)
                            )
                        else:
                            progress = float(
                                self.tokens - config.warmup_tokens
                            ) / float(
                                max(
                                    1,
                                    config.final_tokens - config.warmup_tokens,
                                )
                            )
                            lr_mult = max(
                                0.1, 0.5 * (1.0 + math.cos(math.pi * progress))
                            )
                        lr = config.learning_rate * lr_mult
                        for param_group in optimizer.param_groups:
                            param_group["lr"] = lr
                    else:
                        lr = config.learning_rate

                    pbar.set_description(
                        f"epoch {epoch + 1}) iter {it}: train_loss "
                        f"{loss.item():.7f}, lr = {lr:e}"
                    )
                    if it % 1000 == 0:
                        context = "0" * model.config.block_size
                        x = torch.tensor(
                            [self.train_dataset.stoi[s] for s in context],
                            dtype=torch.long,
                        )[None, ...].to(self.device)

                        x = torch.repeat_interleave(x, 5, dim=0)
                        y = model.sample(
                            x, 20, temperature=1.0, sample=True, top_k=10,
                        )
                        print(f"Iteration {it} samples:")
                        for sent in y:
                            completion = "".join(
                                [self.train_dataset.itos[int(i)] for i in sent]
                            )
                            print(
                                completion[model.config.block_size :].split(
                                    "0"
                                )[0]
                            )
            if not is_train:
                test_loss = float(np.mean(epoch_losses["loss"]))
                print(f"Test loss: {test_loss}")
                return test_loss
            return epoch_losses

        best_loss = float("inf")
        self.tokens = 0

        all_losses = defaultdict(list)

        for epoch in range(config.max_epochs):
            loss_dict = run_epoch("train")
            # if self.test_dataset is not None:
            # test_loss = run_epoch("test")

            for k, v in loss_dict.items():
                all_losses[k].extend(v)

        return all_losses
