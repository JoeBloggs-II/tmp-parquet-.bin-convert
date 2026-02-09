import json
import glob
import numpy as np
from datasets import load_dataset, Dataset, concatenate_datasets, Features, ClassLabel, Value
import struct

# --------------------------
# 1. Load JSON files safely
# --------------------------
json_files = glob.glob("json_data/*.json")  # <-- change to your folder
all_entries = []
seen_texts = set()

for file in json_files:
    try:
        with open(file, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if not content:
                print(f"Skipping empty file: {file}")
                continue
            data = json.loads(content)
            for entry in data:
                text = entry.get("text", "").strip()
                label = entry.get("label")
                if not text or label is None:
                    continue
                if text not in seen_texts:
                    all_entries.append({"sentence": text, "label": label})
                    seen_texts.add(text)
    except json.JSONDecodeError:
        print(f"Skipping invalid JSON file: {file}")

print(f"Loaded {len(all_entries)} unique entries from JSON files.")

# --------------------------
# 2. Load SST-2 dataset
# --------------------------
sst2_dataset = load_dataset("glue", "sst2")
train_dataset = sst2_dataset["train"]

# --------------------------
# 2b. Fix JSON dataset features to match SST-2
# --------------------------
features = Features({
    "sentence": Value("string"),
    "label": ClassLabel(names=["negative", "positive"])
})

# Map int 0/1 to ClassLabel strings
def map_labels(example):
    return {"label": "negative" if example["label"] == 0 else "positive"}

json_dataset = Dataset.from_list(all_entries).map(map_labels)
json_dataset = json_dataset.cast(features)

# --------------------------
# 3. Combine datasets safely
# --------------------------
combined_train = concatenate_datasets([train_dataset, json_dataset])
combined_train = combined_train.shuffle(seed=42)

# --------------------------
# 4. Create new validation split (10%) using HF Dataset method
# --------------------------
split = combined_train.train_test_split(test_size=0.1, seed=42)
train_data = split["train"]
val_data = split["test"]

# --------------------------
# 5. Build alphabet
# --------------------------
all_text = "".join(train_data["sentence"]) + "".join(val_data["sentence"])
alphabet = sorted(list(set(all_text)))
char_to_idx = {c: i for i, c in enumerate(alphabet)}
alphabet_size = len(alphabet)
print(f"Alphabet size: {alphabet_size}")

# --------------------------
# 6. One-hot encoding helper
# --------------------------
def encode_one_hot(text, max_len):
    arr = np.zeros((max_len, alphabet_size), dtype=np.uint8)
    for i, c in enumerate(text[:max_len]):
        arr[i, char_to_idx[c]] = 1
    return arr

# --------------------------
# 7. Save Swift-ready .bin
# --------------------------
def save_bin(filename, dataset):
    num_samples = len(dataset)
    max_seq_len = max(len(t) for t in dataset["sentence"])
    print(f"{filename}: {num_samples} samples, max seq len {max_seq_len}")

    # Pre-allocate arrays
    X = np.zeros((num_samples, max_seq_len, alphabet_size), dtype=np.uint8)

    # Convert labels to 0/1
    y = np.zeros(num_samples, dtype=np.uint8)
    for i, l in enumerate(dataset["label"]):
        if isinstance(l, int):
            y[i] = l
        elif isinstance(l, str):
            y[i] = 0 if l == "negative" else 1
        else:
            raise ValueError(f"Unexpected label type: {l}")

    # Fill X
    for i, text in enumerate(dataset["sentence"]):
        X[i] = encode_one_hot(text, max_seq_len)

    # Flatten X for contiguous memory
    X_flat = X.flatten()

    with open(filename, "wb") as f:
        # Header: num_samples, max_seq_len, alphabet_size
        f.write(struct.pack("III", num_samples, max_seq_len, alphabet_size))
        # One-hot data
        f.write(X_flat.tobytes())
        # Labels
        f.write(y.tobytes())

# --------------------------
# 8. Save train and validation .bin files
# --------------------------
save_bin("train.bin", train_data)
save_bin("validation.bin", val_data)

# --------------------------
# 9. Save alphabet for Swift reference
# --------------------------
with open("alphabet.txt", "w", encoding="utf-8") as f:
    f.write("".join(alphabet))

print("Done! Files created: train.bin, validation.bin, alphabet.txt")
