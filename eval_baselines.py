import json
import os
import numpy as np
import pyarrow.parquet as pq

PREPROCESSED_DIR = "preprocessed_test7"   # change if needed

def read_y_counts(parquet_path, target_col="y"):
    pf = pq.ParquetFile(parquet_path)
    counts = {}
    n = 0
    for batch in pf.iter_batches(batch_size=200_000, columns=[target_col]):
        y = batch.column(0).to_numpy(zero_copy_only=False)
        n += len(y)
        uniq, cnt = np.unique(y, return_counts=True)
        for u, c in zip(uniq.tolist(), cnt.tolist()):
            counts[int(u)] = counts.get(int(u), 0) + int(c)
    return counts, n

def accuracy_majority(y_true, majority_class):
    return float((y_true == majority_class).mean())

def neg_log_likelihood_from_probs(y_true, probs, eps=1e-12):
    # probs: shape [num_classes]
    p = np.clip(probs, eps, 1.0)
    p = p / p.sum()
    return float(-np.log(p[y_true]).mean())

def load_all_y(parquet_path, target_col="y"):
    pf = pq.ParquetFile(parquet_path)
    ys = []
    for batch in pf.iter_batches(batch_size=200_000, columns=[target_col]):
        ys.append(batch.column(0).to_numpy(zero_copy_only=False).astype(np.int64, copy=False))
    return np.concatenate(ys, axis=0)

def main():
    meta_path = os.path.join(PREPROCESSED_DIR, "metadata.json")
    meta = json.load(open(meta_path, "r", encoding="utf-8"))
    num_classes = max(meta["labels"]["label_to_id"].values()) + 1

    train_path = os.path.join(PREPROCESSED_DIR, "train.parquet")
    val_path   = os.path.join(PREPROCESSED_DIR, "val.parquet")
    test_path  = os.path.join(PREPROCESSED_DIR, "test.parquet")

    train_counts, train_n = read_y_counts(train_path)
    train_counts_arr = np.zeros(num_classes, dtype=np.int64)
    for k, v in train_counts.items():
        if 0 <= k < num_classes:
            train_counts_arr[k] = v

    majority_class = int(train_counts_arr.argmax())
    train_prior = train_counts_arr / max(train_counts_arr.sum(), 1)

    print("=== Train class distribution ===")
    for i in range(num_classes):
        print(f"class {i}: count={train_counts_arr[i]}  frac={train_prior[i]:.4f}")
    print(f"\nMajority class = {majority_class} (frac={train_prior[majority_class]:.4f})")
    print(f"Uniform chance accuracy = {1/num_classes:.4f}")
    print(f"Uniform loss log(K)     = {np.log(num_classes):.4f}")

    # Evaluate baselines on VAL/TEST
    for split_name, path in [("val", val_path), ("test", test_path)]:
        y = load_all_y(path)
        maj_acc = accuracy_majority(y, majority_class)

        uniform_probs = np.ones(num_classes, dtype=np.float64) / num_classes
        uniform_nll = neg_log_likelihood_from_probs(y, uniform_probs)

        prior_nll = neg_log_likelihood_from_probs(y, train_prior)

        print(f"\n=== {split_name.upper()} baselines ===")
        print(f"Majority-class accuracy: {maj_acc:.4f}")
        print(f"Uniform NLL (loss):      {uniform_nll:.4f}")
        print(f"Train-prior NLL (loss):  {prior_nll:.4f}")

if __name__ == "__main__":
    main()
