# 📄 PDF Analyzer AI

### Intelligent Document Analysis Using Retrieval-Augmented Generation (RAG)

PDF Analyzer AI is an AI-powered application that allows users to upload PDF documents, ask questions and receive answers based on the document's content. It also provides document summarization, Malayalam translation, quiz generation and PDF comparison.

The application uses pretrained AI models and a Retrieval-Augmented Generation (RAG) pipeline without training or fine-tuning models from scratch.

## 🚀 Live Application

**[Open PDF Analyzer AI](https://pdf-analyzer-ai-e2gdjueaxkpwjfvmf5mpzk.streamlit.app/)**

Upload a text-based PDF and explore the available features.

*Note: The first request may take longer because the AI models need to load. If the application is asleep, click Wake up and wait for it to start.*

## ✨ Features

- **PDF Upload:** Upload PDF documents and extract text page by page.
- **Question Answering:** Ask questions and receive answers based on the uploaded document.
- **Page References:** Display page numbers supporting the generated answers.
- **Evidence Verification:** Classify answers as Supported, Partially Supported or Not Found.
- **AI Summarization:** Generate concise English summaries.
- **Malayalam Translation:** Translate summaries into Malayalam with a download option.
- **Quiz Generation:** Automatically generate fill-in-the-blank questions from PDF content.
- **PDF Comparison:** Compare 2 or 3 PDFs using topic similarity, shared keywords and matching passages.
- **Error Handling:** Display meaningful messages for invalid or scanned PDFs.

## ⚙️ How It Works

The application follows a Retrieval-Augmented Generation workflow.

```text
PDF Upload
    ↓
Text Extraction
    ↓
Text Chunking
    ↓
Embedding Generation
    ↓
FAISS Vector Database
    ↓
Relevant Information Retrieval
    ↓
Answer Generation
    ↓
Answer + Page References + Evidence Verification
```

## 🛠️ Technologies Used

- Python
- Streamlit
- PyMuPDF
- PyMuPDF4LLM
- Sentence Transformers
- FAISS
- Transformers
- PyTorch
- Scikit-learn
- NumPy
- Pandas
- Deep Translator

## 📁 Project Structure

| File | Description |
|---|---|
| `app.py` | Main Streamlit application |
| `chunking.py` | Text cleaning and chunking |
| `embeddings.py` | Generates text embeddings |
| `vector_store.py` | FAISS vector storage and retrieval |
| `rag.py` | RAG-based question answering |
| `verifier.py` | Evidence verification |
| `summarizer.py` | English summary generation |
| `translator.py` | Malayalam translation |
| `mcq.py` | Quiz generation |
| `compare.py` | PDF comparison |
| `evaluate.py` | Model evaluation |
| `requirements.txt` | Project dependencies |

## 📊 Evaluation

The OHR-Bench dataset was used to evaluate document extraction and retrieval performance. It was used only for evaluation and not for model training.

**Evaluation setup:** 5 academic PDFs covering 50 pages.

| Evaluation Metric | Result |
|---|---:|
| Extraction F1 | 0.909 |
| Hit@1 | 0.62 |
| Hit@4 | 0.82 |

To run the evaluation, use `python evaluate.py`. Results are saved in `evaluation_results.csv`.

## 📌 Project Information

- **Project Name:** PDF Analyzer AI
- **Category:** Artificial Intelligence / Generative AI
- **Approach:** Retrieval-Augmented Generation (RAG)
- **Frontend:** Streamlit
- **Deployment:** Streamlit Community Cloud

---

**PDF Analyzer AI – Making document analysis simpler with Artificial Intelligence.**
