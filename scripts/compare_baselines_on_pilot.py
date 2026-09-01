"""Quick check: does the TF-IDF / embedding baseline actually agree with the
teacher LLM (and/or the manually-adjudicated "true" answer) on the rows where
the teacher disagreed with NHTSA's raw COMPDESC label?

Retrains the exact same TF-IDF+LogReg and embedding+LightGBM baselines used in
step4 of the feasibility phase (same sample, seed, and hyperparameters as
scripts/baseline_compdesc.py / scripts/baseline_embeddings.py -- reuses the
precomputed step4b_embeddings.npy to skip the slow encode step), then predicts
on the exact 20 rows from the anthropic teacher-labeling pilot
(output/teacher_labels_anthropic_20.json) for a row-by-row comparison.
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

DATA_DIR = Path(__file__).parent.parent / "data"
OUT_DIR = Path(__file__).parent.parent / "output"
SEED = 42


def load_pilot_rows():
    with open(OUT_DIR / "teacher_labels_anthropic_20.json") as f:
        pilot = json.load(f)["results"]
    cmplids = [str(r["cmplid"]) for r in pilot]
    df = pd.read_parquet(DATA_DIR / "cmpl.parquet", columns=["CMPLID", "CDESCR"])
    df = df[df["CMPLID"].isin(cmplids)].set_index("CMPLID")
    rows = []
    for r in pilot:
        rows.append({
            "cmplid": r["cmplid"],
            "narrative": df.loc[str(r["cmplid"]), "CDESCR"],
            "gold": r["gold_COMPDESC_TOP"],
            "teacher_pred": r["raw_response"]["component"],
        })
    return rows


def fit_tfidf_logreg():
    print("=== TF-IDF + LogReg (replicating baseline_compdesc.py) ===")
    t0 = time.time()
    cols = ["CDESCR", "COMPDESC"]
    df = pd.read_parquet(DATA_DIR / "cmpl.parquet", columns=cols)
    df = df.sample(n=400_000, random_state=SEED).reset_index(drop=True)
    df["COMPDESC_TOP"] = df["COMPDESC"].astype(str).str.split(":").str[0]
    counts = df["COMPDESC_TOP"].value_counts()
    keep = counts[counts >= 50].index
    df = df[df["COMPDESC_TOP"].isin(keep)].reset_index(drop=True)

    le = LabelEncoder()
    y = le.fit_transform(df["COMPDESC_TOP"])
    idx_train, _ = train_test_split(
        np.arange(len(df)), test_size=0.2, random_state=SEED, stratify=y
    )

    tfidf = TfidfVectorizer(max_features=30_000, ngram_range=(1, 2), min_df=3, sublinear_tf=True)
    X_train = tfidf.fit_transform(df["CDESCR"].iloc[idx_train])
    clf = LogisticRegression(max_iter=200, n_jobs=-1, C=5.0)
    clf.fit(X_train, y[idx_train])
    print(f"  fit in {time.time()-t0:.0f}s")
    return tfidf, clf, le


def fit_embedding_lightgbm():
    print("=== Embedding + LightGBM (reusing precomputed step4b_embeddings.npy) ===")
    t0 = time.time()
    cols = ["CDESCR", "COMPDESC"]
    df = pd.read_parquet(DATA_DIR / "cmpl.parquet", columns=cols)
    df = df.sample(n=120_000, random_state=SEED).reset_index(drop=True)
    df["COMPDESC_TOP"] = df["COMPDESC"].astype(str).str.split(":").str[0]
    counts = df["COMPDESC_TOP"].value_counts()
    keep = counts[counts >= 50].index
    df = df[df["COMPDESC_TOP"].isin(keep)].reset_index(drop=True)

    embeddings = np.load(OUT_DIR / "step4b_embeddings.npy")
    assert len(embeddings) == len(df), (len(embeddings), len(df))

    le = LabelEncoder()
    y = le.fit_transform(df["COMPDESC_TOP"])
    idx_train, _ = train_test_split(
        np.arange(len(df)), test_size=0.2, random_state=SEED, stratify=y
    )

    clf = LGBMClassifier(
        n_estimators=400, num_leaves=63, max_depth=-1, learning_rate=0.08,
        min_child_samples=30, reg_lambda=1.0, subsample=0.8, colsample_bytree=0.8,
        n_jobs=-1, random_state=SEED, verbosity=-1,
    )
    clf.fit(embeddings[idx_train], y[idx_train])
    print(f"  fit in {time.time()-t0:.0f}s")

    print("  loading sentence-transformer to encode the 20 pilot narratives...")
    encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    return encoder, clf, le


def main():
    rows = load_pilot_rows()

    tfidf, tfidf_clf, tfidf_le = fit_tfidf_logreg()
    X_pilot_text = tfidf.transform([r["narrative"] for r in rows])
    tfidf_preds = tfidf_le.inverse_transform(tfidf_clf.predict(X_pilot_text))

    encoder, embed_clf, embed_le = fit_embedding_lightgbm()
    pilot_embeddings = encoder.encode(
        [r["narrative"] for r in rows], convert_to_numpy=True, normalize_embeddings=True
    )
    embed_preds = embed_le.inverse_transform(embed_clf.predict(pilot_embeddings))

    for r, tp, ep in zip(rows, tfidf_preds, embed_preds):
        r["tfidf_pred"] = tp
        r["embedding_pred"] = ep
        r["teacher_vs_gold_agree"] = r["teacher_pred"] == r["gold"]
        r["tfidf_vs_gold_agree"] = tp == r["gold"]
        r["embedding_vs_gold_agree"] = ep == r["gold"]
        r["tfidf_matches_teacher_not_gold"] = (tp == r["teacher_pred"]) and (tp != r["gold"])
        r["embedding_matches_teacher_not_gold"] = (ep == r["teacher_pred"]) and (ep != r["gold"])

    n = len(rows)
    summary = {
        "n": n,
        "teacher_vs_gold_agreement": sum(r["teacher_vs_gold_agree"] for r in rows) / n,
        "tfidf_vs_gold_agreement": sum(r["tfidf_vs_gold_agree"] for r in rows) / n,
        "embedding_vs_gold_agreement": sum(r["embedding_vs_gold_agree"] for r in rows) / n,
        "disagreement_rows": [r for r in rows if not r["teacher_vs_gold_agree"]],
    }

    out = {"rows": rows, "summary": {k: v for k, v in summary.items() if k != "disagreement_rows"}}
    with open(OUT_DIR / "baseline_vs_teacher_pilot_comparison.json", "w") as f:
        json.dump(out, f, indent=2)

    print("\n=== Agreement vs. raw NHTSA gold label (n=20) ===")
    print(f"  Teacher (Claude Sonnet 5):     {summary['teacher_vs_gold_agreement']:.0%}")
    print(f"  TF-IDF + LogReg:               {summary['tfidf_vs_gold_agreement']:.0%}")
    print(f"  Embedding + LightGBM:          {summary['embedding_vs_gold_agreement']:.0%}")

    print("\n=== On the 8 rows where the teacher disagreed with gold ===")
    print(f"{'cmplid':>10} {'gold':<28} {'teacher':<28} {'tfidf':<28} {'embedding':<28}")
    for r in summary["disagreement_rows"]:
        print(f"{r['cmplid']:>10} {r['gold']:<28} {r['teacher_pred']:<28} {r['tfidf_pred']:<28} {r['embedding_pred']:<28}")

    n_tfidf_matches_teacher = sum(r["tfidf_matches_teacher_not_gold"] for r in rows)
    n_embed_matches_teacher = sum(r["embedding_matches_teacher_not_gold"] for r in rows)
    print(f"\nOf the 8 teacher/gold disagreements: TF-IDF independently landed on the same "
          f"(non-gold) answer as the teacher {n_tfidf_matches_teacher} times; "
          f"embedding+LightGBM did {n_embed_matches_teacher} times.")

    print(f"\nWrote {OUT_DIR / 'baseline_vs_teacher_pilot_comparison.json'}")


if __name__ == "__main__":
    main()
