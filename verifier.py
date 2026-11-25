import re
import numpy as np
from embeddings import get_model

SUPPORTED = 0.60      # heuristic thresholds
PARTIAL = 0.40


def verify_answer(answer, results):
    if not results or answer.strip().lower().startswith("not found"):
        return "Not Found", 0.0, None

    sents = []
    for r in results:
        for s in re.split(r"(?<=[.!?])\s+", r["text"]):
            if len(s.split()) >= 6:
                sents.append((s.strip(), r["page"]))
    if not sents:
        return "Not Found", 0.0, None

    emb = get_model().encode(
        [answer] + [s for s, _ in sents],
        batch_size=16, normalize_embeddings=True,
        show_progress_bar=False, convert_to_numpy=True,
    )
    sims = emb[1:] @ emb[0]
    i = int(np.argmax(sims))
    score = float(sims[i])

    if score >= SUPPORTED:
        label = "Supported"
    elif score >= PARTIAL:
        label = "Partially Supported"
    else:
        label = "Not Found"
    return label, score, sents[i]