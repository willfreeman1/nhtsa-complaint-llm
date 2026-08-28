"""Step 4b/4c: COMPDESC (top-level category) baselines.

- TF-IDF + Logistic Regression on CDESCR narrative text.
- Structured-fields-only LightGBM on make/model/year + Y/N flags, no text.

Both trained/evaluated on the same random subsample + split for a fair,
fast comparison (full 2.24M rows isn't needed to answer the feasibility
question; this is an exploratory gate, not the final model).
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

DATA_DIR = Path(__file__).parent.parent / "data"
OUT_DIR = Path(__file__).parent.parent / "output"

N_SAMPLE = 400_000
MIN_CLASS_COUNT = 50  # drop ultra-rare top-level categories before modeling
SEED = 42


def main():
    t0 = time.time()
    cols = ["CDESCR", "COMPDESC", "MAKETXT", "MODELTXT", "YEARTXT", "CRASH", "FIRE",
            "MEDICAL_ATTN", "VEHICLES_TOWED_YN", "INJURED", "DEATHS"]
    df = pd.read_parquet(DATA_DIR / "cmpl.parquet", columns=cols)
    df = df.sample(n=N_SAMPLE, random_state=SEED).reset_index(drop=True)

    df["COMPDESC_TOP"] = df["COMPDESC"].astype(str).str.split(":").str[0]
    counts = df["COMPDESC_TOP"].value_counts()
    keep = counts[counts >= MIN_CLASS_COUNT].index
    df = df[df["COMPDESC_TOP"].isin(keep)].reset_index(drop=True)
    n_classes = df["COMPDESC_TOP"].nunique()
    print(f"Sample: {len(df):,} rows, {n_classes} top-level COMPDESC classes kept (>= {MIN_CLASS_COUNT} each)")

    le = LabelEncoder()
    y = le.fit_transform(df["COMPDESC_TOP"])

    idx_train, idx_test = train_test_split(np.arange(len(df)), test_size=0.2, random_state=SEED, stratify=y)

    report = {"n_sample": int(len(df)), "n_classes": int(n_classes)}

    # ---- Baseline: majority class ----
    majority_class = pd.Series(y[idx_train]).mode()[0]
    maj_pred = np.full(len(idx_test), majority_class)
    report["majority_baseline"] = {
        "accuracy": float(accuracy_score(y[idx_test], maj_pred)),
        "macro_f1": float(f1_score(y[idx_test], maj_pred, average="macro")),
    }
    print("Majority-class baseline:", report["majority_baseline"])

    # ---- Baseline A: TF-IDF + Logistic Regression on narrative text ----
    t1 = time.time()
    tfidf = TfidfVectorizer(max_features=30_000, ngram_range=(1, 2), min_df=3, sublinear_tf=True)
    X_text_train = tfidf.fit_transform(df["CDESCR"].iloc[idx_train])
    X_text_test = tfidf.transform(df["CDESCR"].iloc[idx_test])

    clf_text = LogisticRegression(max_iter=200, n_jobs=-1, C=5.0)
    clf_text.fit(X_text_train, y[idx_train])
    pred_text = clf_text.predict(X_text_test)

    report["tfidf_logreg_text_only"] = {
        "accuracy": float(accuracy_score(y[idx_test], pred_text)),
        "macro_f1": float(f1_score(y[idx_test], pred_text, average="macro")),
        "weighted_f1": float(f1_score(y[idx_test], pred_text, average="weighted")),
        "train_time_s": time.time() - t1,
    }
    print("TF-IDF + LogReg (text only):", report["tfidf_logreg_text_only"])

    # ---- Baseline B: structured fields only (no text), LightGBM ----
    t2 = time.time()
    struct_cols_cat = ["MAKETXT", "MODELTXT", "YEARTXT"]
    struct_cols_bin = ["CRASH", "FIRE", "MEDICAL_ATTN", "VEHICLES_TOWED_YN"]

    oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    X_cat = oe.fit_transform(df[struct_cols_cat].astype(str))
    X_bin = (df[struct_cols_bin].astype(str) == "Y").astype(int).to_numpy()
    X_num = df[["INJURED", "DEATHS"]].fillna(0).to_numpy()
    X_struct = np.hstack([X_cat, X_bin, X_num])

    clf_struct = LGBMClassifier(
        n_estimators=300, num_leaves=31, max_depth=7, learning_rate=0.05,
        min_child_samples=100, reg_lambda=1.0, subsample=0.8, colsample_bytree=0.8,
        n_jobs=-1, random_state=SEED, verbosity=-1,
    )
    clf_struct.fit(X_struct[idx_train], y[idx_train],
                    categorical_feature=[0, 1, 2])
    pred_struct = clf_struct.predict(X_struct[idx_test])

    # sanity check vs. a plain majority-class predictor restricted to the same
    # feature set's implied class distribution (catches "worse than majority"
    # pathologies from overfitting on high-cardinality categoricals)
    train_majority = pd.Series(y[idx_train]).mode()[0]
    struct_vs_majority_acc = float(accuracy_score(y[idx_test], pred_struct))
    print(f"[sanity] structured-only acc {struct_vs_majority_acc:.4f} vs majority-baseline acc {report['majority_baseline']['accuracy']:.4f}")

    report["structured_only_lightgbm"] = {
        "accuracy": float(accuracy_score(y[idx_test], pred_struct)),
        "macro_f1": float(f1_score(y[idx_test], pred_struct, average="macro")),
        "weighted_f1": float(f1_score(y[idx_test], pred_struct, average="weighted")),
        "train_time_s": time.time() - t2,
        "features": struct_cols_cat + struct_cols_bin + ["INJURED", "DEATHS"],
    }
    print("Structured-only LightGBM (no text):", report["structured_only_lightgbm"])

    report["total_time_s"] = time.time() - t0
    with open(OUT_DIR / "step4_compdesc_baselines.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote {OUT_DIR / 'step4_compdesc_baselines.json'} (total {report['total_time_s']:.0f}s)")


if __name__ == "__main__":
    main()
