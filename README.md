# Retail Advisor RAG (Amazon Reviews)

Answer product questions from Amazon US Customer Reviews with grounded citations.

## What this repo gives you
- **Data prep**: load Kaggle TSVs, clean, and chunk reviews into passages
- **Hybrid retrieval**: BM25 (lexical) + dense embeddings (MiniLM) + optional reranker
- **Grounded answerer**: extractive synthesis with inline citations like `[R1]`
- **API**: FastAPI endpoint `/ask` that returns an answer + citations + top passages
- **Eval**:
  - Retrieval **Precision@k** on *silver* labels auto-built from topic keywords
  - Optional LLM-graded QA (if you set `OPENAI_API_KEY`)

> Dataset: https://www.kaggle.com/datasets/cynthiarempel/amazon-us-customer-reviews-dataset

## Quickstart
```bash
# 0) Python
python -V            # 3.10+ recommended
python -m venv .venv && source .venv/bin/activate

# 1) Install
pip install -r requirements.txt

# 2) Download data (Kaggle CLI)
#   - pip install kaggle
#   - Place kaggle.json in ~/.kaggle/ and chmod 600 ~/.kaggle/kaggle.json
#   - Then:
kaggle datasets download -d cynthiarempel/amazon-us-customer-reviews-dataset -p data/raw -f tsvs.zip
unzip data/raw/tsvs.zip -d data/raw

# (Optional) You can also start by testing with the tiny synthetic sample we included:
# data/raw/sample_electronics.tsv

# 3) Prepare (choose a category to keep sizes manageable, e.g., Electronics)
python scripts/prepare_data.py --input_dir data/raw --category Electronics --limit 200000

# 4) Build index
python scripts/build_index.py --passages data/processed/passages.parquet --out_dir index

# 5) Serve API
uvicorn app.api:app --reload

# 6) Ask a question
curl -X POST "http://127.0.0.1:8000/ask" -H "Content-Type: application/json"   -d '{"query":"battery life for Kindle?", "k":5, "product_filter":"Kindle"}'
```

## Project layout
```
retail-advisor-rag/
├─ app/
│  ├─ api.py            # FastAPI app
│  ├─ retriever.py      # BM25 + embedding hybrid retrieval (RRF) + optional reranker
│  ├─ generator.py      # Extractive answer composer with citations
│  ├─ utils.py          # Text cleaning, chunking, IO helpers
├─ scripts/
│  ├─ prepare_data.py   # From Kaggle TSVs ⇒ cleaned passages parquet
│  ├─ build_index.py    # Build BM25 & FAISS indices
│  ├─ eval_retrieval.py # Precision@k with silver topic labels
│  └─ eval_qa_llm.py    # Optional LLM grading (OpenAI), safe to skip
├─ data/
│  ├─ raw/              # Put Kaggle files here
│  └─ processed/        # Outputs: reviews.parquet, passages.parquet
├─ index/               # Indices: FAISS, BM25 tokens, metadata
├─ config.yaml
└─ requirements.txt
```

## Notes
- **No API key needed** for core pipeline (generator is extractive). If you want generative answers, plug in an LLM in `app/generator.py`.
- On a laptop, start with **Electronics** and **limit** ~200k rows to keep memory reasonable.
- If you hit FAISS/SentenceTransformer memory issues, reduce `--limit` or tweak batch sizes in `build_index.py`.

## Evaluation
- Retrieval: we synthesize topic queries (battery, screen, durability, etc.) and treat passages mentioning the topic for the **same product** as relevant. This produces a reasonable *silver* metric for **Precision@k**.
- QA: optional LLM-as-judge via `scripts/eval_qa_llm.py` using a short rubric (helpfulness, grounding, harmlessness).
