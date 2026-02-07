from datasets import load_dataset
import struct
import os

# --------------------
# Config
# --------------------

DATASET_NAME = "stanfordnlp/sst2"
SPLIT = "validation"          # "train" or "validation"
OUT_FILE = "sst2_validation.bin"

MAX_LEN = 256            # truncate / pad length
PAD_TOKEN = 0
UNK_TOKEN = 1

ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789 .,!?;:'\"()-"

# Build vocab
vocab = {ch: i + 2 for i, ch in enumerate(ALPHABET)}
VOCAB_SIZE = len(vocab) + 2

# --------------------
# Encoding
# --------------------

def encode(text):
    text = text.lower()
    tokens = [vocab.get(ch, UNK_TOKEN) for ch in text]
    tokens = tokens[:MAX_LEN]
    if len(tokens) < MAX_LEN:
        tokens += [PAD_TOKEN] * (MAX_LEN - len(tokens))
    return tokens

# --------------------
# Conversion
# --------------------

print("Loading dataset...")
ds = load_dataset(DATASET_NAME, split=SPLIT)

print(f"Writing {len(ds)} samples to {OUT_FILE}")

with open(OUT_FILE, "wb") as f:
    for sample in ds:
        tokens = encode(sample["sentence"])
        label = sample["label"]

        # Write tokens
        for t in tokens:
            f.write(struct.pack("i", t))

        # Write label
        f.write(struct.pack("b", label))

print("Done.")
print("Vocab size:", VOCAB_SIZE)
print("Sequence length:", MAX_LEN)
