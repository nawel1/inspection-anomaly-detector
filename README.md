# Inspection Anomaly Detector

Uploads any quality inspection PDF (QIMA, Bureau Veritas, fabric reports), retrieves relevant context from a local knowledge base of standards using RAG, and calls Mistral to return a structured PASS/FAIL verdict with detected defects and a plain-language summary.

---

## Stack

Python, FastAPI, Streamlit, Docker, ChromaDB, sentence-transformers, Mistral API

---

## How it works

1. Upload any inspection PDF (QIMA, Bureau Veritas, fabric reports...)
2. RAG retrieves relevant context from quality standards stored in ChromaDB
3. Mistral analyzes the report and returns PASS/FAIL + defects + summary

---

## How to run

1. Add reference PDFs to `data/rag_documents/`
2. Add your key to `.env`: `MISTRAL_API_KEY=your_key`
3. `docker-compose up --build`

Dashboard at `http://localhost:8501` — API docs at `http://localhost:8000/docs`

---

## Test the project

Sample reports are available in `data/` — upload any PDF to the dashboard.
Reference documents for RAG are in `data/rag_documents/`.
Just run `docker-compose up --build` — everything is included.

---

## Project structure

```
src/analyze.py        # PDF extraction + RAG + Mistral analysis
src/rag.py            # ChromaDB ingestion and retrieval
api/main.py           # FastAPI endpoint
dashboard/app.py      # Streamlit dashboard
```

---

## Known limitations

- Scanned PDFs (image-only) are not supported — text extraction requires a selectable text layer
- Mistral free tier has rate limits; if you hit a 429, wait ~1 minute and retry
