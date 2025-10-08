import re
import faiss
import numpy as np
import pandas as pd
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from dataclasses import dataclass

def _simple_tok(s: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", (s or "").lower())

@dataclass
class RetrievalResources:
    passages: pd.DataFrame
    bm25: BM25Okapi
    bm25_tokens: List[List[str]]
    embedder: SentenceTransformer
    faiss_index: faiss.Index

def load_resources(passages_path: str, bm25_tokens_path: str, pid_map_path: str, faiss_index_path: str, embedder_name: str):
    passages = pd.read_parquet(passages_path)
    bm25_tokens = np.load(bm25_tokens_path, allow_pickle=True).tolist()
    bm25 = BM25Okapi(bm25_tokens)
    faiss_index = faiss.read_index(faiss_index_path)
    embedder = SentenceTransformer(embedder_name)
    return RetrievalResources(passages, bm25, bm25_tokens, embedder, faiss_index)

def _rrf(ranks: Dict[int, int], k: int = 60) -> Dict[int, float]:
    return {pid: 1.0 / (k + r) for pid, r in ranks.items()}

def hybrid_search(rr: RetrievalResources, query: str, k: int = 5, bm25_topn: int = 200, embed_topn: int = 200, rrf_k: int = 60):
    bm25_scores = rr.bm25.get_scores(_simple_tok(query))
    top_bm25_idx = np.argsort(-bm25_scores)[:bm25_topn]
    bm25_ranks = {int(i): r for r, i in enumerate(top_bm25_idx)}
    bm25_rrf = _rrf(bm25_ranks, rrf_k)

    q_emb = rr.embedder.encode([query], normalize_embeddings=True, show_progress_bar=False)
    D, I = rr.faiss_index.search(q_emb.astype(np.float32), embed_topn)
    I = I[0].tolist()
    embed_ranks = {int(pid): r for r, pid in enumerate(I)}
    embed_rrf = _rrf(embed_ranks, rrf_k)

    scores = {}
    for pid, sc in bm25_rrf.items():
        scores[pid] = scores.get(pid, 0) + sc
    for pid, sc in embed_rrf.items():
        scores[pid] = scores.get(pid, 0) + sc

    top = sorted(scores.items(), key=lambda x: -x[1])[:max(k, 50)]
    rows = []
    for pid, sc in top[:k]:
        row = rr.passages.iloc[pid].to_dict()
        row["pid"] = int(pid)
        row["score"] = float(sc)
        rows.append(row)
    return rows
