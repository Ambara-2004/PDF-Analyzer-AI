import re
import time
from deep_translator import GoogleTranslator, MyMemoryTranslator


def _split(text, max_chars=450):
    sentences = re.split(r"(?<=[.!?])\s+", text)
    parts, current = [], ""
    for s in sentences:
        if len(current) + len(s) + 1 > max_chars and current:
            parts.append(current.strip())
            current = ""
        current += s + " "
    if current.strip():
        parts.append(current.strip())
    return parts


def _translate_with(translator, parts):
    out = []
    for p in parts:
        for attempt in range(3):
            try:
                out.append(translator.translate(p))
                break
            except Exception:
                time.sleep(2 * (attempt + 1))
        else:
            raise RuntimeError("translation service kept failing")
        time.sleep(0.5)
    return " ".join(out)


def to_malayalam(text):
    parts = _split(text)
    try:
        return _translate_with(GoogleTranslator(source="en", target="ml"), parts)
    except Exception:
        return _translate_with(MyMemoryTranslator(source="en-US", target="ml-IN"), parts)