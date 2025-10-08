import argparse, os, glob, re
import pandas as pd
import numpy as np
from tqdm import tqdm
from app.utils import clean_text, chunk_text

KEEP_COLS = ["review_id","product_id","product_parent","product_title","star_rating","review_headline","review_body","review_date","product_category"]

def load_kaggle_tsvs(input_dir: str, category: str = None, limit: int = None):
    paths = glob.glob(os.path.join(input_dir, "*.tsv")) + glob.glob(os.path.join(input_dir, "*.tsv.gz"))
    if not paths: raise FileNotFoundError(f"No TSV files found in {input_dir}")
    dfs = []
    for p in tqdm(paths, desc="Reading TSVs"):
        df = pd.read_csv(p, sep="\t", dtype=str, on_bad_lines="skip")
        cols = [c for c in KEEP_COLS if c in df.columns]
        df = df[cols].copy()
        for c in ["review_body","review_headline","product_title","product_category"]:
            if c in df.columns:
                df[c] = df[c].fillna("").astype(str)
        if category and "product_category" in df.columns:
            df = df[df["product_category"].str.contains(category, case=False, na=False)]
        dfs.append(df)
    full = pd.concat(dfs, ignore_index=True)
    if limit: full = full.sample(n=min(limit, len(full)), random_state=42)
    full["text"] = (full.get("review_headline","") + ". " + full.get("review_body","")).apply(clean_text)
    full["star_rating"] = pd.to_numeric(full.get("star_rating"), errors="coerce").fillna(0).astype(int)
    full["review_date"] = pd.to_datetime(full.get("review_date"), errors="coerce").dt.date.astype(str)
    full = full.dropna(subset=["text"]); full = full[full["text"].str.len() > 0]
    full = full.drop_duplicates(subset=["review_id"]).reset_index(drop=True)
    return full

def make_passages(reviews: pd.DataFrame, target_words: int = 90):
    rows = []
    for _, row in tqdm(reviews.iterrows(), total=len(reviews), desc="Chunking"):
        for j, ch in enumerate(chunk_text(row["text"], target_words=target_words)):
            rows.append({
                "pid": None,
                "review_id": row.get("review_id"),
                "product_id": row.get("product_id"),
                "product_parent": row.get("product_parent"),
                "product_title": row.get("product_title"),
                "product_category": row.get("product_category"),
                "star_rating": row.get("star_rating"),
                "review_date": row.get("review_date"),
                "chunk_id": j,
                "text": ch
            })
    import pandas as pd
    passages = pd.DataFrame(rows)
    passages["pid"] = range(len(passages))
    return passages

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--category", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out_dir", default="data/processed")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    reviews = load_kaggle_tsvs(args.input_dir, category=args.category, limit=args.limit)
    reviews.to_parquet(os.path.join(args.out_dir, "reviews.parquet"))
    passages = make_passages(reviews)
    passages.to_parquet(os.path.join(args.out_dir, "passages.parquet"))
    print("Saved:", os.path.join(args.out_dir, "reviews.parquet"), os.path.join(args.out_dir, "passages.parquet"))

if __name__ == "__main__":
    main()
