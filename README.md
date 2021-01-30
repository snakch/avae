Small project seeking to explore ideas in NLP by doing character-level predictions and generation to generate words. This could be generalised to other quanta of language, like individual sentences.


Uses the very pedagogical repo at https://github.com/karpathy/minGPT as a backbone and modifies it.

Currently, the main idea is to use a Transformer network, but make it explicitly into a latent model by treating the encoder and doecoders like VAE encoders/decoder.

Also implement a few tricks to be able to learn how words are formed more explicitly.

A few ideas of interest revolve around how to do transfer learning (for example teach a model how to create words in one languge, and see if you can do so using a different smaller dataset in another langugae)
and encode meaningful features in the latent space (for example can we encode "style" features in the latent space?)

### Installation

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