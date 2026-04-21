"""Build a small stratified test dataset for fast end-to-end workflow verification.

Samples N rows per class from the full product CSV, copies matching JPGs into
`data/test_images/`, and writes `data/test_products.csv` with the image_path
column rewritten to point at the test folder.

Run from the project root:

    python scripts/build_test_dataset.py
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
SRC_CSV = ROOT / "data" / "processed_products_with_images.csv"
SRC_IMG_DIR = ROOT / "data" / "images"
DST_IMG_DIR = ROOT / "data" / "test_images"
DST_CSV = ROOT / "data" / "test_products.csv"

PER_CLASS = 10
RANDOM_STATE = 42


def main() -> None:
    df = pd.read_csv(SRC_CSV)
    print(f"Source CSV: {SRC_CSV.relative_to(ROOT)}  ({len(df):,} rows, "
          f"{df['class_id'].nunique()} classes)")

    sampled = (
        df.groupby("class_id", group_keys=False)
          .sample(n=PER_CLASS, random_state=RANDOM_STATE)
          .reset_index(drop=True)
    )
    print(f"Sampled {len(sampled)} rows ({PER_CLASS} per class, across "
          f"{sampled['class_id'].nunique()} classes)")

    DST_IMG_DIR.mkdir(parents=True, exist_ok=True)
    kept_rows = []
    missing = 0
    for row in sampled.itertuples(index=False):
        src = SRC_IMG_DIR / f"{row.sku}.jpg"
        dst = DST_IMG_DIR / f"{row.sku}.jpg"
        if not src.exists():
            print(f"  skip: source missing {src.name}")
            missing += 1
            continue
        shutil.copyfile(src, dst)
        kept_rows.append(row._asdict())

    out_df = pd.DataFrame(kept_rows)
    out_df["image_path"] = out_df["sku"].apply(lambda s: f"data/test_images/{s}.jpg")
    out_df.to_csv(DST_CSV, index=False)

    print(
        f"Done. {len(out_df)} images copied → {DST_IMG_DIR.relative_to(ROOT)}, "
        f"CSV written → {DST_CSV.relative_to(ROOT)}"
        + (f"  ({missing} sources missing)" if missing else "")
    )
    print(f"Class coverage: {out_df['class_id'].nunique()} / "
          f"{df['class_id'].nunique()} classes represented")


if __name__ == "__main__":
    main()
