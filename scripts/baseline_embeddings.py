"""Stronger baseline: sentence-embedding (frozen, pretrained, no fine-tuning) +
LightGBM, to test whether semantic embeddings close the gap to the zero-shot LLM
ceiling without needing to fine-tune an LLM at all. Uses the exact same sampling/
split logic as baseline_compdesc.py for a fair comparison."""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
torch.set_num_threads(20)
from lightgbm import LGBMClassifier
from sentence_transformers import SentenceTransformer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

DATA_DIR = Path(__file__).parent.parent / "data"
OUT_DIR = Path(__file__).parent.parent / "output"

N_SAMPLE = 120_000
MIN_CLASS_COUNT = 50
SEED = 42
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # fast, CPU-friendly, 384-dim


def main():
    cols = ["CDESCR", "COMPDESC", "MAKETXT", "MODELTXT", "YEARTXT",
            "CRASH", "FIRE", "MEDICAL_ATTN", "VEHICLES_TOWED_YN", "INJURED", "DEATHS"]
    df = pd.read_parquet(DATA_DIR / "cmpl.parquet", columns=cols)
    df = df.sample(n=N_SAMPLE, random_state=SEED).reset_index(drop=True)
    df["COMPDESC_TOP"] = df["COMPDESC"].astype(str).str.split(":").str[0]

    counts = df["COMPDESC_TOP"].value_counts()
    keep = counts[counts >= MIN_CLASS_COUNT].index
    df = df[df["COMPDESC_TOP"].isin(keep)].reset_index(drop=True)
    print(f"Sample: {len(df):,} rows, {df['COMPDESC_TOP'].nunique()} classes kept (>= {MIN_CLASS_COUNT} each)")

    le = LabelEncoder()
    y = le.fit_transform(df["COMPDESC_TOP"])
    idx_train, idx_test = train_test_split(np.arange(len(df)), test_size=0.2, random_state=SEED, stratify=y)

    t0 = time.time()
    print(f"Loading embedding model {MODEL_NAME} ...")
    model = SentenceTransformer(MODEL_NAME, device="cpu")
    print(f"Loaded in {time.time()-t0:.1f}s. Encoding {len(df):,} narratives (batch, {__import__('os').cpu_count()} cores available)...")

    t1 = time.time()
    embeddings = model.encode(
        df["CDESCR"].tolist(),
        batch_size=256,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    print(f"Encoded {embeddings.shape} in {time.time()-t1:.1f}s")
    np.save(OUT_DIR / "step4b_embeddings.npy", embeddings)

    t2 = time.time()
    clf = LGBMClassifier(
        n_estimators=400, num_leaves=63, max_depth=-1, learning_rate=0.08,
        min_child_samples=30, reg_lambda=1.0, subsample=0.8, colsample_bytree=0.8,
        n_jobs=-1, random_state=SEED, verbosity=-1,
    )
    clf.fit(embeddings[idx_train], y[idx_train])
    pred = clf.predict(embeddings[idx_test])

    report = {
        "model": MODEL_NAME,
        "embedding_dim": int(embeddings.shape[1]),
        "n_sample": len(df),
        "n_classes": int(df["COMPDESC_TOP"].nunique()),
        "accuracy": float(accuracy_score(y[idx_test], pred)),
        "macro_f1": float(f1_score(y[idx_test], pred, average="macro")),
        "weighted_f1": float(f1_score(y[idx_test], pred, average="weighted")),
        "embed_time_s": time.time() - t1,
        "train_time_s": time.time() - t2,
    }
    print("Embedding + LightGBM (text only):", report)

    with open(OUT_DIR / "step4b_embedding_baseline.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote {OUT_DIR / 'step4b_embedding_baseline.json'}")


if __name__ == "__main__":
    main()
