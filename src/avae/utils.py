import torch
import numpy as np


def top_k_logits(logits, k):
    v, ix = torch.topk(logits, k)

    out = logits.clone()
    out[out < v[:, [-1]]] = -float("Inf")
    return out


def pad_word(word: str, length: int = 20) -> str:
    """ Given a word, add 0 padding before it and until it reaches length.
        Then add another 0 before it.

    Parameters
    ----------
    word : str
        word
    length : int, optional
        length of padded word], by default 20

    Returns
    -------
    str
        padded word
    """
    before = "0"
    after = "0" * ((length - len(word)) + 1)
    word = before + word + after
    return word


def depad_word(word: str, length: int = 20) -> str:
    """ Depad the word of initial 0 and later 0s, inverse function of pad word

    Parameters
    ----------
    word : str
        padded word
    length : int, optional
        length of padded word], by default 20

    Returns
    -------
    str
        depadded word
    """
    word = word.replace("0", "")
    return word


def encode_word(word: str, mapping: dict, length: int = 20) -> list:
    """ Given a mapping of char to int, map that word to its int
        representation.

    Parameters
    ----------
    word : str
        word to encode
    mapping : dict
        word to index dict
    length : int, optional
        length of the final word, by default 20

    Returns
    -------
    list
        index representation of the word
    """
    word = pad_word(word, length=length)
    return [mapping[char] for char in word]


def decode_word(word: list, inverse_mapping: dict, length: int = 20) -> str:
    """ Given an index rep of a word and the mapping from index to character,
        decode the word back to a strin

    Parameters
    ----------
    word : list
        index reppresentation of a word
    inverse_mapping : dict
        index to character
    length : int, optional
        length of the padded word, by default 20

    Returns
    -------
    str
        decoded word
    """
    char_list = [
        depad_word(inverse_mapping[char], length=length) for char in word
    ]
    return "".join(char_list)


def str_to_tensor(context, vae, n_samples, source_dict=None):
    if source_dict is None:
        source_dict = {
            source_type: "_random" for source_type in vae.source_types
        }

    prefix = get_source_prefix(source_dict, vae)

    context = "0" * (vae.config.block_size - len(context) - 1) + context

    x = torch.tensor(
        prefix + [vae.stoi[s] for s in context], dtype=torch.long,
    )[None, ...].to(vae.device)
    x = torch.repeat_interleave(x, n_samples, dim=0)
    return x


def get_source_prefix(source_dict, vae):
    prefix = []
    for source_type, source_map in vae.sourcetoi.items():
        source_value = source_dict[source_type]
        prefix.append(source_map[source_value])
    return prefix


def generate_samples(
    initial_context,
    vae,
    n_samples=10,
    method="smart",
    sample=False,
    temperature=3.0,
    source_dict=None,
    top_k=10,
    second_context=None,
):

    x = str_to_tensor(
        initial_context, vae, n_samples, source_dict=source_dict,
    )
    around_word = None
    if method == "word":
        x = str_to_tensor("", vae, n_samples, source_dict=source_dict,)
        around_word = str_to_tensor(
            initial_context, vae, n_samples, source_dict=source_dict,
        )

    if method == "interpolate":
        x = str_to_tensor(
            "",
            vae.sourcetoi,
            vae.config.block_size,
            n_samples,
            source_dict=source_dict,
        )
        around_word = str_to_tensor(
            initial_context, vae, n_samples, source_dict=source_dict,
        )
        second_context = str_to_tensor(
            second_context, vae, n_samples, source_dict=source_dict,
        )

    initial_randomness = len(initial_context) == 0

    y = vae.sample(
        x,
        21,
        sample=sample,
        method=method,
        temperature=temperature,
        top_k=top_k,
        initial_randomness=initial_randomness,
        around_word=around_word,
        second_context=second_context,
    )

    completions = []

    start = vae.config.block_size - len(initial_context) - 1

    if method in ["word", "interpolate"]:
        start += len(initial_context)

    for sent in y:

        completion = "".join([vae.itos[int(i)] for i in sent[1:]])

        completions.append(completion[start:].split("0")[0])

    return completions
