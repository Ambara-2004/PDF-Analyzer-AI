import hashlib
import streamlit as st
import pymupdf
import pymupdf4llm

from chunking import create_chunks
from embeddings import create_embeddings
from vector_store import create_vector_store, search_similar_chunks
from summarizer import summarize
from rag import answer_question
from translator import to_malayalam
from verifier import verify_answer
from compare import similarity_matrix, matching_passages, keywords
from mcq import generate_mcqs

st.set_page_config(page_title="PDF Analyzer AI", page_icon="📄", layout="wide")

# ---------------- Styling ----------------
st.markdown(
    """
<style>
.block-container {padding-top: 1.5rem; max-width: 1100px;}
.hero {
  background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 55%, #EC4899 100%);
  border-radius: 18px; padding: 2rem 2.2rem; color: #fff; margin-bottom: 1.2rem;
}
.hero h1 {margin: 0; font-size: 2.2rem; color: #fff;}
.hero p {margin: .4rem 0 .9rem 0; font-size: 1.05rem; opacity: .92; color: #fff;}
.chip {
  display: inline-block; background: rgba(255,255,255,.18); color: #fff;
  border: 1px solid rgba(255,255,255,.35); border-radius: 999px;
  padding: .2rem .8rem; margin: 0 .4rem .4rem 0; font-size: .85rem;
}
div[data-testid="stMetric"] {
  background: rgba(128,128,128,.08); border: 1px solid rgba(128,128,128,.25);
  border-radius: 14px; padding: .8rem 1rem;
}
.stTabs [data-baseweb="tab-list"] {gap: .4rem;}
.stTabs [data-baseweb="tab"] {
  border-radius: 10px 10px 0 0; padding: .5rem 1.1rem; font-weight: 600;
}
.badge {
  display: inline-block; padding: .3rem .9rem; border-radius: 999px;
  font-weight: 700; font-size: .9rem; margin: .3rem 0;
}
.b-ok {background: rgba(34,197,94,.18); color: #16a34a; border: 1px solid #16a34a;}
.b-mid {background: rgba(245,158,11,.18); color: #d97706; border: 1px solid #d97706;}
.b-bad {background: rgba(239,68,68,.18); color: #dc2626; border: 1px solid #dc2626;}
.answer-text {font-size: 1.1rem; line-height: 1.6;}
.footer {text-align: center; opacity: .6; font-size: .85rem; margin-top: 2rem;}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="hero">
<h1>📄 PDF Analyzer AI</h1>
<p>Upload a PDF, then ask questions, get summaries, take a quiz or compare documents.</p>
<span class="chip">💬 Q&A with page citations</span>
<span class="chip">✅ Evidence check</span>
<span class="chip">🌐 English &amp; Malayalam summary</span>
<span class="chip">📝 Auto quiz</span>
<span class="chip">🔀 Compare PDFs</span>
</div>
""",
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Processing PDF (runs once per file)...")
def process_pdf(file_hash, _pdf_bytes):
    doc = pymupdf.open(stream=_pdf_bytes, filetype="pdf")
    try:
        pages = pymupdf4llm.to_markdown(doc, page_chunks=True)
        page_data = [(i + 1, p["text"]) for i, p in enumerate(pages)]
    finally:
        doc.close()

    chunks = create_chunks(page_data, chunk_size=200, overlap=40)
    if not chunks:
        return None
    embeddings = create_embeddings(chunks)
    index = create_vector_store(embeddings)
    return {
        "page_data": page_data,
        "chunks": chunks,
        "dim": embeddings.shape[1],
        "index": index,
    }


def set_question(text):
    st.session_state["q_text"] = text


# ---------------- Upload ----------------
with st.container(border=True):
    st.subheader("📤 Step 1: Upload your PDF")
    uploaded_file = st.file_uploader(
        "Choose a text-based PDF", type=["pdf"], label_visibility="collapsed"
    )

data = None

if uploaded_file is not None:
    pdf_bytes = uploaded_file.getvalue()
    file_hash = hashlib.md5(pdf_bytes).hexdigest()

    if st.session_state.get("file_hash") != file_hash:
        st.session_state["file_hash"] = file_hash
        st.session_state["qa"] = None
        st.session_state["summary_text"] = None
        st.session_state["mcqs"] = None
        st.session_state["q_text"] = ""

    try:
        data = process_pdf(file_hash, pdf_bytes)
        if data is None:
            st.error("No extractable text found (the PDF may be scanned).")
    except Exception as e:
        st.error(f"Could not read this PDF: {e}")
        data = None

if data:
    st.success(f"✅ {uploaded_file.name} processed successfully!")
    words = sum(len(t.split()) for _, t in data["page_data"])
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📄 Pages", len(data["page_data"]))
    c2.metric("🧩 Chunks", len(data["chunks"]))
    c3.metric("🔤 Words", f"{words:,}")
    c4.metric("⏱️ Reading time", f"~{max(1, round(words / 200))} min")

st.write("")
tab_ask, tab_sum, tab_quiz, tab_cmp = st.tabs(
    ["💬 Ask", "📌 Summary", "📝 Quiz", "🔀 Compare"]
)

# ---------------- Ask ----------------
with tab_ask:
    if not data:
        st.info("⬆️ Upload a PDF above to ask questions.")
    else:
        st.markdown("**Try an example:**")
        e1, e2, e3 = st.columns(3)
        e1.button("What is the main purpose?", use_container_width=True,
                  on_click=set_question, args=("What is the main purpose of this document?",))
        e2.button("Which methods or tools are used?", use_container_width=True,
                  on_click=set_question, args=("Which methods or tools are used?",))
        e3.button("What is the conclusion?", use_container_width=True,
                  on_click=set_question, args=("What is the conclusion?",))

        question = st.text_input(
            "Your question",
            key="q_text",
            placeholder="Type a specific question about your PDF...",
        )

        if st.button("🔍 Get Answer", type="primary"):
            if question.strip():
                results = search_similar_chunks(
                    question, data["index"], data["chunks"], top_k=4
                )
                with st.spinner("Searching the document and writing an answer..."):
                    answer, pages = answer_question(question, results)
                    label, score, evidence = verify_answer(answer, results)
                st.session_state["qa"] = {
                    "question": question, "answer": answer, "pages": pages,
                    "label": label, "score": score, "evidence": evidence,
                    "results": results,
                }
            else:
                st.session_state["qa"] = None
                st.warning("Please enter a question.")

        qa = st.session_state.get("qa")
        if qa:
            with st.container(border=True):
                st.caption(f"❓ {qa['question']}")
                st.markdown(
                    f"<div class='answer-text'>{qa['answer']}</div>",
                    unsafe_allow_html=True,
                )
                if qa["label"] == "Supported":
                    cls, txt = "b-ok", "✅ Supported"
                elif qa["label"] == "Partially Supported":
                    cls, txt = "b-mid", "⚠️ Partially Supported"
                else:
                    cls, txt = "b-bad", "❌ Not Found"
                st.markdown(
                    f"<span class='badge {cls}'>{txt} · match {qa['score']:.2f}</span>"
                    f"&nbsp; 📑 Source pages: <b>{', '.join(map(str, qa['pages']))}</b>",
                    unsafe_allow_html=True,
                )
                if qa["evidence"]:
                    st.info(
                        f"**Best evidence (page {qa['evidence'][1]}):** {qa['evidence'][0]}"
                    )

            with st.expander("📚 Show retrieved passages"):
                for r in qa["results"]:
                    st.markdown(f"**Page {r['page']}** · similarity {r['score']:.2f}")
                    st.write(r["text"])
                    st.divider()

        with st.expander("📖 View extracted text"):
            st.text_area(
                "Extracted Text",
                "\n".join(f"--- Page {n} ---\n{t}" for n, t in data["page_data"]),
                height=300,
                label_visibility="collapsed",
            )

# ---------------- Summary ----------------
with tab_sum:
    if not data:
        st.info("⬆️ Upload a PDF above to generate a summary.")
    else:
        language = st.radio("Summary language", ["English", "Malayalam"], horizontal=True)

        if st.button("✨ Generate Summary", type="primary"):
            summary = summarize(data["page_data"])
            if not summary:
                st.session_state["summary_text"] = None
                st.warning("Not enough readable text to generate a summary.")
            elif language == "English":
                st.session_state["summary_text"] = summary
                st.session_state["summary_lang"] = "en"
            else:
                try:
                    with st.spinner("Translating to Malayalam..."):
                        st.session_state["summary_text"] = to_malayalam(summary)
                    st.session_state["summary_lang"] = "ml"
                except Exception as e:
                    st.error(f"Translation failed (check your internet): {e}")
                    st.session_state["summary_text"] = summary
                    st.session_state["summary_lang"] = "en"

        if st.session_state.get("summary_text"):
            with st.container(border=True):
                st.markdown(
                    f"<div class='answer-text'>{st.session_state['summary_text']}</div>",
                    unsafe_allow_html=True,
                )
            st.download_button(
                "⬇️ Download summary",
                st.session_state["summary_text"],
                file_name=f"summary_{st.session_state.get('summary_lang', 'en')}.txt",
            )

# ---------------- Quiz ----------------
with tab_quiz:
    if not data:
        st.info("⬆️ Upload a PDF above to generate a quiz.")
    else:
        st.write("Fill-in-the-blank questions made from your document.")
        n_q = st.slider("Number of questions", 3, 10, 5)
        if st.button("🎯 Generate Quiz", type="primary"):
            st.session_state["mcqs"] = generate_mcqs(data["page_data"], n=n_q)
            st.session_state["quiz_nonce"] = st.session_state.get("quiz_nonce", 0) + 1

        mcqs = st.session_state.get("mcqs")
        if mcqs is not None and len(mcqs) == 0:
            st.warning("Could not make questions from this PDF (not enough text).")
        elif mcqs:
            nonce = st.session_state.get("quiz_nonce", 0)
            picks = []
            for i, q in enumerate(mcqs):
                with st.container(border=True):
                    st.markdown(f"**Q{i + 1}.** {q['question']}")
                    picks.append(
                        st.radio("Choose:", q["options"], index=None,
                                 key=f"mcq_{nonce}_{i}", horizontal=True,
                                 label_visibility="collapsed")
                    )
            if st.checkbox("Show answers"):
                score = 0
                for i, (q, p) in enumerate(zip(mcqs, picks)):
                    ok = p == q["answer"]
                    score += ok
                    icon = "✅" if ok else "❌"
                    st.write(f"{icon} Q{i + 1}: **{q['answer']}** (page {q['page']})")
                st.success(f"🏆 Score: {score} / {len(mcqs)}")

# ---------------- Compare ----------------
with tab_cmp:
    compare_files = st.file_uploader(
        "Upload 2 or 3 PDFs to compare",
        type=["pdf"],
        accept_multiple_files=True,
        key="compare_uploader",
    )

    if compare_files:
        if len(compare_files) < 2:
            st.info("Upload at least 2 PDFs.")
        elif len(compare_files) > 3:
            st.warning("Please upload at most 3 PDFs (memory limit).")
        else:
            names, datas, ok = [], [], True
            for f in compare_files:
                try:
                    b = f.getvalue()
                    d = process_pdf(hashlib.md5(b).hexdigest(), b)
                except Exception as e:
                    st.error(f"Could not read {f.name}: {e}")
                    ok = False
                    break
                if d is None:
                    st.error(f"No extractable text in {f.name}.")
                    ok = False
                    break
                names.append(f.name)
                datas.append(d)

            if ok:
                st.subheader("Overall topic similarity")
                sim = similarity_matrix(datas)
                for i in range(len(datas)):
                    for j in range(i + 1, len(datas)):
                        st.metric(f"{names[i]}  vs  {names[j]}",
                                  f"{sim[i][j] * 100:.0f}%")

                st.subheader("Keywords")
                shared, unique = keywords(datas)
                with st.container(border=True):
                    st.write("**Shared:** " + (", ".join(shared) if shared else "none"))
                    for n, u in zip(names, unique):
                        st.write(f"**Mainly in {n}:** " + ", ".join(u))

                st.subheader("Closest matching passages")
                for i in range(len(datas)):
                    for j in range(i + 1, len(datas)):
                        st.markdown(f"**{names[i]} ↔ {names[j]}**")
                        for m in matching_passages(datas[i], datas[j]):
                            with st.container(border=True):
                                st.caption(f"Similarity {m['score']:.2f}")
                                st.write(f"*{names[i]}, page {m['a']['page']}:* {m['a']['text'][:300]}...")
                                st.write(f"*{names[j]}, page {m['b']['page']}:* {m['b']['text'][:300]}...")

st.markdown(
    "<div class='footer'>PDF Analyzer AI · Retrieval-Augmented Generation with pretrained models</div>",
    unsafe_allow_html=True,
)