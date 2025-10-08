import argparse, os, re
import numpy as np
import pandas as pd
from tqdm import tqdm
import faiss
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

def simple_tok(s: str): return re.findall(r"[a-z0-9]+", (s or "").lower())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--passages", default="data/processed/passages.parquet")
    ap.add_argument("--out_dir", default="index")
    ap.add_argument("--embedder", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--batch_size", type=int, default=256)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df = pd.read_parquet(args.passages)
    texts = df["text"].astype(str).tolist()

    tokens = [simple_tok(t) for t in texts]
    np.save(os.path.join(args.out_dir, "bm25_tokens.npy"), np.array(tokens, dtype=object), allow_pickle=True)

    meta = df[["pid","review_id","product_id","product_parent","product_title","product_category","star_rating","review_date","text"]].copy()
    meta.to_parquet(os.path.join(args.out_dir, "pid_map.parquet"))

    model = SentenceTransformer(args.embedder)
    emb = []
    for i in tqdm(range(0, len(texts), args.batch_size), desc="Embedding"):
        batch = texts[i:i+args.batch_size]
        e = model.encode(batch, normalize_embeddings=True, show_progress_bar=False)
        emb.append(e.astype("float32"))
    X = np.vstack(emb)
    index = faiss.IndexFlatIP(X.shape[1])
    index.add(X)
    faiss.write_index(index, os.path.join(args.out_dir, "faiss.index"))
    print("Index built:", len(texts), "passages")

if __name__ == "__main__":
    main()
