import argparse, re, random
import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
from app.retriever import load_resources, hybrid_search

TOPICS = ["battery","screen","display","camera","sound","audio","noise","durability","charging","fit","comfort","performance","speed"]

def has_topic(text: str, topic: str) -> bool:
    return re.search(rf"\\b{re.escape(topic)}\\b", text, flags=re.I) is not None

def build_silver_queries(df: pd.DataFrame, topics=TOPICS, per_product: int = 3, seed: int = 42):
    random.seed(seed)
    queries = []
    by_prod = df.groupby("product_title")
    for title, g in by_prod:
        text = " ".join(g["text"].astype(str).tolist()).lower()
        ts = [t for t in topics if t in text]
        if not ts: continue
        chosen = random.sample(ts, k=min(per_product, len(ts)))
        for t in chosen:
            queries.append({"query": f"{t} for {title}?", "product_title": title, "topic": t})
    return pd.DataFrame(queries)

def precision_at_k(rel: set, retrieved: list, k: int) -> float:
    hits = sum(1 for pid in retrieved[:k] if pid in rel)
    return hits / max(1,k)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--passages", default="data/processed/passages.parquet")
    ap.add_argument("--bm25_tokens", default="index/bm25_tokens.npy")
    ap.add_argument("--pid_map", default="index/pid_map.parquet")
    ap.add_argument("--faiss_index", default="index/faiss.index")
    ap.add_argument("--embedder", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--k", type=int, nargs="+", default=[1,3,5,10])
    args = ap.parse_args()

    rr = load_resources(args.passages, args.bm25_tokens, args.pid_map, args.faiss_index, args.embedder)
    df = rr.passages

    qdf = build_silver_queries(df)
    print("Queries:", len(qdf))

    metrics = defaultdict(list)
    for _, row in tqdm(qdf.iterrows(), total=len(qdf)):
        q, title, topic = row["query"], row["product_title"], row["topic"]
        sub = df[(df["product_title"]==title) & (df["text"].str.contains(rf"\\b{re.escape(topic)}\\b", case=False, regex=True))]
        rel_pids = set(sub["pid"].astype(int).tolist())
        results = hybrid_search(rr, q, k=50)
        retrieved_pids = [int(r["pid"]) for r in results]
        for kk in args.k:
            metrics[f"P@{kk}"].append(precision_at_k(rel_pids, retrieved_pids, kk))

    out = {k: float(np.mean(v)) for k, v in metrics.items()}
    print("Precision@k:", out)

if __name__ == "__main__":
    main()
