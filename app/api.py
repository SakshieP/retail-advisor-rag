import os
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from .retriever import load_resources, hybrid_search
from .generator import make_answer

PASSAGES = os.environ.get("PASSAGES", "data/processed/passages.parquet")
BM25_TOKENS = os.environ.get("BM25_TOKENS", "index/bm25_tokens.npy")
PID_MAP = os.environ.get("PID_MAP", "index/pid_map.parquet")  # not used in slim retriever but kept for parity
FAISS_INDEX = os.environ.get("FAISS_INDEX", "index/faiss.index")
EMBEDDER = os.environ.get("EMBEDDER", "sentence-transformers/all-MiniLM-L6-v2")

app = FastAPI(title="Retail Advisor RAG")
RR = load_resources(PASSAGES, BM25_TOKENS, PID_MAP, FAISS_INDEX, EMBEDDER)

class AskRequest(BaseModel):
    query: str
    k: int = 5
    product_filter: Optional[str] = None

@app.post("/ask")
def ask(req: AskRequest):
    hits = hybrid_search(RR, req.query, k=max(req.k, 10))
    if req.product_filter:
        hits = [h for h in hits if req.product_filter.lower() in (h.get("product_title","").lower())]
    hits = hits[:req.k]
    ans = make_answer(req.query, hits, max_claims=6)
    return {"query": req.query, "k": req.k, "answer": ans["answer"], "citations": ans["citations"], "hits": hits}

@app.get("/health")
def health():
    return {"status":"ok"}
