import re
import random
from pathlib import Path
import pandas as pd
import pymupdf

from chunking import create_chunks
from embeddings import create_embeddings
from vector_store import create_vector_store, search_similar_chunks

PARQUET = r"C:\Users\ambar\OneDrive\Desktop\PDF_Analyzer_AI\dataset\OHR-Bench\OHR-Bench_v2.parquet"
PDF_ROOT = Path(r"D:\PDF_Analyzer_Dataset")
N_DOCS = 5          # documents to test
MAX_PAGES = 10      # pages per document
random.seed(42)


def tokens(t):
    return re.findall(r"[a-z0-9]+", t.lower())


def f1(a, b):
    from collections import Counter
    ca, cb = Counter(tokens(a)), Counter(tokens(b))
    common = sum((ca & cb).values())
    if not common:
        return 0.0
    p, r = common / sum(ca.values()), common / sum(cb.values())
    return 2 * p * r / (p + r)


def pick_query(gt):
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", gt)
             if 12 <= len(s.split()) <= 40 and sum(c.isalpha() for c in s) / len(s) > 0.8]
    return random.choice(sents) if sents else None


df = pd.read_parquet(PARQUET, columns=["domain", "doc_name", "page_idx", "gt_text"])
df = df[df.domain == "academic"]

pdf_lookup = {p.stem: p for p in PDF_ROOT.rglob("*.pdf")}
rows = []
docs_done = 0

for doc_name, g in df.groupby("doc_name"):
    stem = doc_name.split("/")[-1]
    if stem not in pdf_lookup:
        continue
    if docs_done >= N_DOCS:
        break

    doc = pymupdf.open(pdf_lookup[stem])
    page_data = [(i + 1, doc[i].get_text()) for i in range(min(len(doc), MAX_PAGES))]
    doc.close()

    chunks = create_chunks(page_data, chunk_size=200, overlap=40)
    if not chunks:
        continue
    index = create_vector_store(create_embeddings(chunks))

    for _, r in g.iterrows():
        p = int(r.page_idx) + 1
        if p > MAX_PAGES:
            continue
        extracted = dict(page_data).get(p, "")
        query = pick_query(r.gt_text)
        if query is None:
            continue
        res = search_similar_chunks(query, index, chunks, top_k=4)
        pages = [x["page"] for x in res]
        rows.append({
            "doc": stem, "page": p,
            "extraction_f1": round(f1(extracted, r.gt_text), 3),
            "hit_at_1": int(pages[:1] == [p]),
            "hit_at_4": int(p in pages),
            "query": query[:120],
        })
    docs_done += 1
    print(f"done {stem} ({docs_done}/{N_DOCS})")

out = pd.DataFrame(rows)
out.to_csv("evaluation_results.csv", index=False)

print("\n=== Results ===")
print("Documents:", out.doc.nunique(), " Pages/queries:", len(out))
print("Extraction F1 (mean):", round(out.extraction_f1.mean(), 3))
print("Hit@1:", round(out.hit_at_1.mean(), 3))
print("Hit@4:", round(out.hit_at_4.mean(), 3))
print("Saved: evaluation_results.csv")