import torch


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


def generate_samples(
    vae,
    n_samples=10,
    initial_context="",
    method="smart",
    sample=False,
    temperature=1.0,
):
    context = (
        "0" * (vae.config.block_size - len(initial_context)) + initial_context
    )

    device = "cpu"
    if torch.cuda.is_available():
        device = torch.cuda.current_device()

    x = torch.tensor([vae.stoi[s] for s in context], dtype=torch.long,)[
        None, ...
    ].to(device)
    x = torch.repeat_interleave(x, n_samples, dim=0)
    y = vae.sample(
        x, 20, sample=sample, method=method, temperature=temperature
    )

    completions = []
    for sent in y:
        completion = "".join([vae.itos[int(i)] for i in sent])
        completions.append(
            completion[vae.config.block_size - len(initial_context) :].split(
                "0"
            )[0]
        )
    return completions
