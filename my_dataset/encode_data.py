import json
import glob
import numpy as np
from datasets import load_dataset, Dataset, concatenate_datasets, Features, ClassLabel, Value
import struct

# --------------------------
# 1. Load JSON files safely
# --------------------------
json_files = glob.glob("json_data/*.json")
all_entries = []
seen_texts = set()

for file in json_files:
    try:
        with open(file, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if not content:
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
        pass

print(f"Loaded {len(all_entries)} unique entries from JSON files.")

# --------------------------
# 2. Load SST-2
# --------------------------
sst2 = load_dataset("glue", "sst2")["train"]

features = Features({
    "sentence": Value("string"),
    "label": ClassLabel(names=["negative", "positive"])
})

def map_labels(example):
    return {"label": "negative" if example["label"] == 0 else "positive"}

json_ds = Dataset.from_list(all_entries).map(map_labels).cast(features)

combined = concatenate_datasets([sst2, json_ds]).shuffle(seed=42)
split = combined.train_test_split(test_size=0.1, seed=42)

train_data = split["train"]
val_data = split["test"]

# --------------------------
# 3. Build alphabet (+ PAD = 0)
# --------------------------
all_text = "".join(train_data["sentence"]) + "".join(val_data["sentence"])
alphabet = sorted(set(all_text))

PAD = "\0"
alphabet = [PAD] + alphabet
char_to_idx = {c: i for i, c in enumerate(alphabet)}

alphabet_size = len(alphabet)
print(f"Alphabet size (incl PAD): {alphabet_size}")

# --------------------------
# 4. Encode helper (indexed)
# --------------------------
def encode_indices(text, max_len):
    arr = np.zeros(max_len, dtype=np.uint8)  # PAD = 0
    for i, c in enumerate(text[:max_len]):
        arr[i] = char_to_idx[c]
    return arr

# --------------------------
# 5. Save binary (FAST)
# --------------------------
def save_bin(filename, dataset):
    num_samples = len(dataset)
    max_len = max(len(t) for t in dataset["sentence"])

    print(f"{filename}: {num_samples} samples, max len {max_len}")

    X = np.zeros((num_samples, max_len), dtype=np.uint8)
    y = np.zeros(num_samples, dtype=np.uint8)

    for i, (text, label) in enumerate(zip(dataset["sentence"], dataset["label"])):
        X[i] = encode_indices(text, max_len)
        y[i] = 0 if label == "negative" else 1

    with open(filename, "wb") as f:
        # Header
        f.write(struct.pack("III", num_samples, max_len, alphabet_size))
        # Data
        f.write(X.tobytes())
        f.write(y.tobytes())

# --------------------------
# 6. Write files
# --------------------------
save_bin("train.bin", train_data)
save_bin("validation.bin", val_data)

with open("alphabet.txt", "w", encoding="utf-8") as f:
    f.write("".join(alphabet))

print("Done.")
