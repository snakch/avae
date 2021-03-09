

## Introduction 

This is  a Small project seeking to explore ideas in NLP and representation learning by doing character-level word generation.

The idea is to create a generative model with an interesting latent-space. The model should learn 'global' information about a word, such as style, language,  and patterns like suffixes and prefixes. for 


Try this model out at https://the-namegen.herokuapp.com/. This app is a name generator which allows the user to specify a name 'style' (language), or to define an anchor name and generate a name in close proximity in latent space.


## Ideas and architetcture.


### 1.Attention VAE

The generator model uses the an encoder-decoder architecture similar to Attention is all you need. A main difference is that encoder key and value vectors actually map to a latent space with a normal gaussian prior a la VAE. This allows us to leverage the efficiency and power of transformers whilst allowing the model to learn a meanigful latent representation of words.


Note: we used the very pedagogical https://github.com/karpathy/minGPT as inspiration to design the transformer architecture.

### 2. Encode full word, decode partial word

Another difference with regular encoder / decoder networks is that the encoder actually gets to see the entire word, whilst the decoder sees only a partial word as usual. The idea is to incentivise the latent-space to encode global and stylistic information about the word rather than simply the next letter.

### 3. Autoregressive prior + Smart encoder

We use a couple of tricks to help the model at generation time. Whilst the encoder gets to see the entire word at encoding time, this is not possible during word generation since we generate one letter at a time.
A couple of things are done:
1. We train a MADE autoregressive model on the latent space to help make latent encodings more powerful
2. We train a 'smart encoder', a simple MLP network which aims to mimick full word encoding from partial word. In other words, if the full word is x and the partial word is x', the smart encoder SE aims to mimick the encoder E via SE(z|x') ~ E(z|x).

### 4. Using for better semantic encoding and generation

We further aim to create a model that disentangles known information about a word, such as language of origin. The model should also be able to sample word conditioned on such a label.

To do that, we let the encoder part of the network see the label, and we jointly train a simple MLP network, a 'supervisor' to predict the label from the latent-space embeddings and from the final prediction logits. We find that this helps with overall cross entropy of the model and does so at the level of the latent space.



## A few results:

Incoming soon!



## Installation

Ideally create a new environment, for example using conda

```bash
conda create -n avae
conda activate avae

```

From the root of the repo, run:
```bash
pip install -r requirements
pip install .

```