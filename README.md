# RAG WW2 — Retrieval-Augmented Generation chatbot on World War II

A question-answering chatbot about **World War II** built with a
**Retrieval-Augmented Generation (RAG)** pipeline. It retrieves relevant passages
from a curated corpus (Wikipedia articles, a geographic PDF and CSV data) using a
**FAISS** vector index and semantic embeddings, then generates grounded answers
with a local **Llama 3.1 (8B)** model served through **Ollama**. A **Streamlit**
app provides the chat interface.

## Architecture

```
                ┌──────────────┐
  Wikipedia ──▶ │              │
  Geo PDF   ──▶ │  Ingestion   │──▶ data/processed/*.jsonl ──▶ documentos.jsonl
  CSV data  ──▶ │              │                                     │
                └──────────────┘                                     ▼
                                                        ┌────────────────────────┐
   User question ──▶ embed (MiniLM) ──▶ FAISS search ──▶│  top-k relevant chunks │
                                                        └────────────────────────┘
                                                                     │
                                                                     ▼
                                              Llama 3.1 8B (Ollama) ──▶ grounded answer
```

- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2`
- **Vector store:** FAISS (`index/faiss_index.bin`)
- **LLM:** `llama3.1:8b` served locally via Ollama (`http://localhost:11434`)
- **UI:** Streamlit

## Repository structure

```
.
├── src/
│   ├── config.py              # Paths and chunking parameters
│   ├── ingest_wikipedia.py    # Downloads WW2 Wikipedia articles → wiki_docs.jsonl
│   ├── ingest_geo_pdf.py      # Extracts the geographic PDF → geo_pdf_docs.jsonl
│   ├── build_dataset.py       # Merges all sources → documentos.jsonl
│   ├── build_index.py         # Chunks + embeds documents → FAISS index
│   ├── query_rag.py           # CLI test of retrieval
│   └── rag_chat.py            # Full RAG: retrieval + Llama generation
├── streamlit/
│   └── app.py                 # Chat web interface
├── data/
│   ├── raw/pdfs/              # Source PDF(s)
│   └── processed/             # Ingested/merged corpora (.jsonl)
├── index/                     # Prebuilt FAISS index + metadata
├── requirements.txt
└── README.md
```

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com/) running locally with the Llama 3.1 model:

```bash
ollama pull llama3.1:8b
```

Install Python dependencies:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Pipeline

Run from the project root. The repository already ships with a prebuilt index, so
you can skip straight to step 5 to try it; steps 1–4 regenerate the corpus and
index from scratch.

```bash
python src/ingest_wikipedia.py    # 1. Download WW2 articles from Wikipedia
python src/ingest_geo_pdf.py      # 2. Extract text from the geographic PDF
python src/build_dataset.py       # 3. Merge all sources into documentos.jsonl
python src/build_index.py         # 4. Build the FAISS index
python src/query_rag.py           # 5. Quick retrieval test (no LLM)
python src/rag_chat.py            # 6. Full RAG chat in the terminal
```

## Run the app

Make sure Ollama is running, then:

```bash
streamlit run streamlit/app.py
```

## Notes

- `ingest_wikipedia.py` sends a Wikipedia-recommended `User-Agent` with a contact
  email — update it with your own if you fork the project.
- The generation model runs **fully locally** via Ollama; no external API keys are
  required.

---

> **Note:** source code comments are written in Spanish.
