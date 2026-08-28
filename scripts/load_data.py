"""Load the raw tab-delimited FLAT_CMPL.txt into a Parquet file for fast reuse.

Gotchas handled: tab-delimited, no header row, Windows-1252 (cp1252) encoding
(not UTF-8), 51 columns per CMPL.txt.
"""
import csv
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from schema import COLUMNS

DATA_DIR = Path(__file__).parent.parent / "data"
RAW_PATH = DATA_DIR / "FLAT_CMPL.txt"
PARQUET_PATH = DATA_DIR / "cmpl.parquet"


def main():
    t0 = time.time()
    print(f"Reading {RAW_PATH} ...")
    df = pd.read_csv(
        RAW_PATH,
        sep="\t",
        header=None,
        names=COLUMNS,
        encoding="latin-1",
        dtype=str,
        na_values=[],
        keep_default_na=False,
        quoting=csv.QUOTE_NONE,
        on_bad_lines="warn",
    )
    print(f"Loaded {len(df):,} rows x {len(df.columns)} cols in {time.time()-t0:.1f}s")

    for col in ["INJURED", "DEATHS", "MILES", "OCCURENCES", "NUM_CYLS", "VEH_SPEED"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df.to_parquet(PARQUET_PATH, index=False)
    print(f"Wrote {PARQUET_PATH} ({PARQUET_PATH.stat().st_size / 1e6:.0f} MB)")
    print(df.head(3).T)


if __name__ == "__main__":
    main()
