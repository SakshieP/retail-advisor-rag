from typing import List, Dict, Any
import re
from sklearn.feature_extraction.text import TfidfVectorizer

def _score_sentences(query: str, sentences: List[str]) -> List[float]:
    if not sentences: return []
    vec = TfidfVectorizer(ngram_range=(1,2), min_df=1, max_df=0.9).fit([query] + sentences)
    qv = vec.transform([query]); sv = vec.transform(sentences)
    return (sv @ qv.T).toarray().flatten().tolist()

def make_answer(query: str, hits: List[Dict[str, Any]], max_claims: int = 6) -> Dict[str, Any]:
    sents, back = [], []
    for idx, h in enumerate(hits, start=1):
        text = h.get("text","")
        parts = re.split(r'(?<=[.!?])\s+', text)
        for p in parts:
            p = p.strip()
            if len(p.split()) < 5: continue
            sents.append(p); back.append(idx)
    scores = _score_sentences(query, sents)
    ranked = sorted(zip(sents, back, scores), key=lambda x: -x[2])

    claims, seen = [], set()
    for s, refidx, _ in ranked:
        if len(claims) >= max_claims: break
        if s in seen: continue
        seen.add(s)
        claims.append(f"- {s} [R{refidx}]")
    if not claims:
        claims = ["- I couldn't find specific evidence in the top results. Try rephrasing or adding a product name."]

    citations = []
    for i, h in enumerate(hits, start=1):
        citations.append({
            "id": f"R{i}",
            "review_id": h.get("review_id"),
            "product_title": h.get("product_title"),
            "star_rating": h.get("star_rating"),
            "review_date": h.get("review_date"),
            "product_id": h.get("product_id"),
            "product_category": h.get("product_category"),
            "passage_preview": (h.get("text","")[:240] + "...") if h.get("text") else ""
        })

    answer = "Here’s what the reviews say:\n\n" + "\n".join(claims)
    return {"answer": answer, "citations": citations}
