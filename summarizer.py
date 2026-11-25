import re
import numpy as np
from embeddings import get_model

NOISE = re.compile(
    r"university|institute|conference|ieee|proceedings|doi|copyright|"
    r"@|http|www\.|keywords|et al\.|fig\.|table \d|authorized licensed",
    re.I,
)


def summarize(page_data, n_sentences=6):
    text = "\n".join(t for _, t in page_data)

    text = re.split(r"\n[#*\s]*(references|bibliography)[*\s]*\n", text, flags=re.I)[0]

    m = re.search(r"abstract", text, flags=re.I)
    if m:
        text = text[m.end():]

    text = re.sub(r"\s+", " ", text)
    text = text.lstrip("_*—-– ")
    text = re.sub(r"\b(FUTURE SCOPE AND CONCLUSION|[IVX]+\. [A-Z ]{4,})\b", " ", text)
    text = re.sub(r"\s+", " ", text)

    sentences = re.split(r"(?<=[.!?])\s+", text)

    good = []
    for s in sentences:
        s = s.strip()
        words = s.split()
        if not (8 <= len(words) <= 60):
            continue
        if NOISE.search(s):
            continue
        if sum(c.isalpha() for c in s) / max(len(s), 1) < 0.75:
            continue
        good.append(s)

    if len(good) < 2:
        return None

    emb = get_model().encode(
        good, batch_size=16, normalize_embeddings=True,
        show_progress_bar=False, convert_to_numpy=True,
    )
    centroid = emb.mean(axis=0)
    scores = emb @ centroid
    top = sorted(np.argsort(scores)[::-1][:n_sentences])
    return " ".join(good[i] for i in top)