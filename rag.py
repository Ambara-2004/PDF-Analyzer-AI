from functools import lru_cache
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_NAME = "google/flan-t5-base"
MIN_SCORE = 0.12          # below this, the PDF probably has no answer
torch.set_num_threads(4)


@lru_cache(maxsize=1)
def load_generator():
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    model.eval()
    return tok, model


def answer_question(question, results, max_words_per_chunk=100):
    pages = sorted({r["page"] for r in results[:3]})

    if not results or results[0]["score"] < MIN_SCORE:
        return "Not found in the document", pages

    context = "\n\n".join(
        " ".join(r["text"].split()[:max_words_per_chunk]) for r in results[:3]
    )
    prompt = (
        f"Question: {question}\n\n"
        f"Read the passages below and answer the question in one or two "
        f"complete sentences.\n\nPassages: {context}\n\n"
        f"Question: {question}\nAnswer:"
    )

    tok, model = load_generator()
    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=768)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=80, num_beams=2)
    answer = tok.decode(out[0], skip_special_tokens=True).strip()

    if not answer:
        answer = "Not found in the document"
    return answer, pages