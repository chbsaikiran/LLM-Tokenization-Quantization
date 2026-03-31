# Tokenizer Folder Overview

This folder contains code to **train a custom BPE tokenizer** on WikiText and then **test encode/decode behavior** on a sample sentence.

## Files

- `train_tokenizer.py`
  - Loads the `wikitext-103-raw-v1` dataset via Hugging Face `datasets`.
  - Trains a BPE tokenizer (`tokenizers` library) on the **train split**.
  - Uses:
    - **Normalization:** Unicode decomposition (`NFD`), lowercasing, accent stripping.
    - **Pre-tokenization:** split by whitespace, punctuation, and individual digits.
    - **Trainer config:** vocab size `30000` and standard special tokens (`[UNK]`, `[CLS]`, `[SEP]`, `[PAD]`, `[MASK]`).
  - Filters empty lines from the dataset before training.
  - Saves the tokenizer JSON to `data/tokenizer-wiki.json`.

- `encode_decode_text.py`
  - Loads the trained tokenizer from `data/tokenizer-wiki.json`.
  - Encodes a test sentence: `"Hello, y'all! How are you 😁 ?"`
  - Prints:
    - token strings (`output.tokens`)
    - token IDs (`output.ids`)
    - one token offset (`output.offsets[9]`)
    - decoded text from token IDs
    - a single original character slice (`sentence[26:27]`) to compare with offsets.

- `data/tokenizer-wiki.json`
  - The saved tokenizer model/config produced by training.

## Typical workflow

1. Run training:
   - `python train_tokenizer.py`
2. Run encode/decode test:
   - `python encode_decode_text.py`

## What this demonstrates

- How to build a tokenizer from raw text data.
- How normalization and pre-tokenization choices affect token boundaries.
- How to inspect token IDs and character offsets for downstream NLP tasks.
