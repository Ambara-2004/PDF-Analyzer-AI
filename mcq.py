import random
import re
from collections import Counter
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

BAD = re.compile(r"http|www\.|@|et al\.|doi|copyright|fig\.|table \d", re.I)


def generate_mcqs(page_data, n=5, seed=None):
    rng = random.Random(seed)
    text = "\n".join(t for _, t in page_data)
    text = re.split(r"\n[#*\s]*(references|bibliography)[*\s]*\n", text, flags=re.I)[0]

    # word pool: frequent, meaningful words
    words = re.findall(r"[A-Za-z]{6,}", text.lower())
    freq = Counter(w for w in words if w not in ENGLISH_STOP_WORDS)
    pool = [w for w, c in freq.items() if c >= 2]
    if len(pool) < 6:
        return []

    # candidate sentences, remembering their page
    sents = []
    for page, t in page_data:
        t = re.sub(r"\s+", " ", re.sub(r"[#*_|]", " ", t))
        for s in re.split(r"(?<=[.!?])\s+", t):
            w = s.split()
            if 10 <= len(w) <= 30 and not BAD.search(s) \
                    and sum(c.isalpha() for c in s) / len(s) > 0.8:
                sents.append((s.strip(), page))
    rng.shuffle(sents)

    pool_set = set(pool)
    mcqs = []
    for s, page in sents:
        cands = [w for w in re.findall(r"[A-Za-z]+", s) if w.lower() in pool_set]
        if not cands:
            continue
        answer = rng.choice(cands)
        a = answer.lower()
        similar = [w for w in pool if w != a and abs(len(w) - len(a)) <= 3]
        if len(similar) < 3:
            continue
        options = rng.sample(similar, 3) + [a]
        rng.shuffle(options)
        question = re.sub(rf"\b{re.escape(answer)}\b", "_____", s, count=1)
        mcqs.append({"question": question, "options": options,
                     "answer": a, "page": page})
        if len(mcqs) == n:
            break
    return mcqs