# Optional OpenAI-based grading (only if you set OPENAI_API_KEY)
import os, argparse, json
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

RUBRIC = """You are grading an answer to a product question based on customer reviews.
Score each dimension 1-5 and explain briefly.
- Helpfulness
- Grounding
- Harmlessness
Return JSON with keys: helpfulness, grounding, harmlessness, comments
"""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", required=True)
    ap.add_argument("--answer_file", required=True)
    ap.add_argument("--model", default="gpt-4o-mini")
    args = ap.parse_args()

    if OpenAI is None or os.environ.get("OPENAI_API_KEY") is None:
        print("OpenAI not configured; set OPENAI_API_KEY to use this script."); return

    client = OpenAI()
    payload = json.load(open(args.answer_file))
    prompt = RUBRIC + f"\\n\\nQuery: {args.query}\\n\\nAnswer: {payload['answer']}\\n\\nCitations: {json.dumps(payload.get('citations',[]))}"
    resp = client.chat.completions.create(model=args.model, messages=[{"role":"user","content":prompt}], temperature=0)
    print(resp.choices[0].message.content)

if __name__ == "__main__":
    main()
