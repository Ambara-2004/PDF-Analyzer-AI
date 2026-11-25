import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def _all_vectors(data):
    return data["index"].reconstruct_n(0, data["index"].ntotal)


def doc_vector(data):
    v = _all_vectors(data).mean(axis=0)
    return v / (np.linalg.norm(v) + 1e-9)


def similarity_matrix(datas):
    vecs = [doc_vector(d) for d in datas]
    return np.array([[float(a @ b) for b in vecs] for a in vecs])


def matching_passages(a, b, top=3):
    """Closest chunk in b for each chunk in a; return the best pairs."""
    scores, idx = b["index"].search(_all_vectors(a), 1)
    order = np.argsort(scores[:, 0])[::-1][:top]
    return [
        {
            "score": float(scores[i, 0]),
            "a": a["chunks"][i],
            "b": b["chunks"][int(idx[i, 0])],
        }
        for i in order
    ]


def keywords(datas, n=15):
    texts = [" ".join(c["text"] for c in d["chunks"]) for d in datas]
    vec = TfidfVectorizer(stop_words="english", max_features=5000)
    tfidf = vec.fit_transform(texts)
    terms = np.array(vec.get_feature_names_out())
    tops = []
    for i in range(len(datas)):
        row = tfidf[i].toarray()[0]
        tops.append([t for t in terms[np.argsort(row)[::-1][:n]] if t.isalpha()])
    shared = set(tops[0]).intersection(*tops[1:]) if len(tops) > 1 else set()
    unique = [[t for t in top if t not in shared] for top in tops]
    return sorted(shared), unique