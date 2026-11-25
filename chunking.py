import re


def clean_text(text):
    """Remove boilerplate and markup noise from extracted PDF text."""
    text = re.sub(
        r"Authorized licensed use limited to:.*?Restrictions apply\.?",
        " ", text, flags=re.S
    )
    text = re.sub(r"<br>|</?mark>", " ", text)
    text = re.sub(r"\|", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"[-_*#]{2,}", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def create_chunks(page_data, chunk_size=200, overlap=40):
    chunks = []
    step = max(chunk_size - overlap, 1)

    for page_number, page_text in page_data:
        words = clean_text(page_text).split()
        start = 0

        while start < len(words):
            chunk_words = words[start:start + chunk_size]
            chunk_text = " ".join(chunk_words).strip()

            if len(chunk_words) >= 15:
                chunks.append({"text": chunk_text, "page": page_number})

            start += step

    return chunks