from pathlib import Path

import argparse
import joblib
import matplotlib.pyplot as plt
import numpy as np

import yaml

from avae.dataset import CharDataset
from avae.model import AttentionVae
from avae.trainer import Trainer, TrainerConfig
from avae.utils import generate_samples
from fuzzywuzzy import process


# PROJECT_DIR = Path(".").resolve().parent.parent
# DATA_DIR = PROJECT_DIR / "data"
# WIKI_DIR = DATA_DIR / "wikitext-2"
# NAMES_DIR = DATA_DIR / "names"
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path")
    args = parser.parse_args()
    return args


def test(config_path):
    return Path(config_path)


def clean_word(word):
    word = word.replace("\n", "").lower()
    return word


def load_data(path):

    names = []
    with open(path, "r") as f:
        for word in f:
            names.append(clean_word(word))
    return names


def get_datasets(names, maxlen, chars):

    split = 0.9
    idx = int(len(names) * split)
    np.random.seed(137)
    np.random.shuffle(names)
    train_names = names[:idx]
    val_names = names[idx:]

    raw_train_names = " ".join(train_names)
    raw_val_names = " ".join(val_names)

    train_dataset = CharDataset(raw_train_names, maxlen, chars=chars)
    val_dataset = CharDataset(raw_val_names, maxlen, chars=chars)
    return train_dataset, val_dataset


def get_sample(model, config, unique_train_words, context=""):

    samples = generate_samples(
        model,
        n_samples=config["n_samples"],
        initial_context=context,
        method=config["method"],
        sample=config["sample"],
        top_k=config["top_k"],
        temperature=config["temperature"],
    )

    samples = list(set(samples))
    text = ""
    for sample in samples:

        if unique_train_words:

            match = process.extract(sample, unique_train_words, limit=1)[0][0]
            sample += " " * (20 - len(sample))
            sample += f"\t\t closest match: {match}"

            text += sample
            text += "\n"
    return text + "\n"


def get_all_samples(model, config, unique_train_words=None):
    text = ""

    context = ""
    text += f"Context : {context}\n"
    sample = get_sample(model, config, unique_train_words, context=context)
    text += sample

    context = "a"
    text += f"Context : {context}\n"
    sample = get_sample(model, config, unique_train_words, context=context)
    text += sample

    context = "m"
    text += f"Context : {context}\n"
    sample = get_sample(model, config, unique_train_words, context=context)
    text += sample

    context = "sim"
    text += f"Context : {context}\n"
    sample = get_sample(model, config, unique_train_words, context=context)
    text += sample

    return text


def train(vae, train_dataset, val_dataset, trainer_config, final_tokens):
    trainer_config["final_tokens"] = final_tokens
    tconf = TrainerConfig(**trainer_config)
    trainer = Trainer(
        vae, train_dataset, val_dataset, tconf, log_nearest_words=True
    )
    losses = trainer.train()
    return losses


def main(config_path):
    with open(config_path, "r") as f:
        configs = yaml.load(f, Loader=yaml.Loader)

    for config in configs:
        fine_tune_evaluate(config)


def fine_tune_evaluate(config):
    model_path = Path(config["model_path"])

    model_outputs_dir = Path(config["outputs_dir"]) / (
        model_path.name + "_" + config["name"]
    )
    model_outputs_dir.mkdir(exist_ok=True)

    with open(model_outputs_dir / "config.yaml", "w+") as f:
        yaml.dump(config, f)

    names = load_data(config["data_path"])
    vae = AttentionVae.load(model_path)

    chars = list(vae.stoi.keys())
    maxlen = vae.config.block_size

    frozen_list = []
    for par in vae.named_parameters():
        for word in config["freeze_layers"]:
            if word:
                if word in par[0]:
                    frozen_list.append(par[0])

    trainer_config = config["trainer_config"]
    trainer_config["freeze_layers"] = tuple(frozen_list)

    train_dataset, val_dataset = get_datasets(names, maxlen, chars)
    final_tokens = 2 * len(train_dataset) * maxlen

    losses = train(
        vae, train_dataset, val_dataset, trainer_config, final_tokens
    )

    plt.figure()
    plt.plot(losses["test_loss"])
    plt.savefig(model_outputs_dir / "test_loss.png")
    plt.close()

    joblib.dump(losses, model_outputs_dir / "losses.joblib")

    sampled_text = get_all_samples(
        vae, config["sampling"], train_dataset.unique_word_list,
    )

    with open(model_outputs_dir / "samples.txt", "w") as f:
        f.write(sampled_text)


if __name__ == "__main__":
    config_path = parse_args().config_path
    # import pdb

    # pdb.set_trace()
    main(config_path)

