import os
from pathlib import Path
import chromadb
from chromadb.config import Settings
import pdfplumber
from sentence_transformers import SentenceTransformer

os.environ["ANONYMIZED_TELEMETRY"] = "False"

CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"
DOCS_DIR   = Path(__file__).parent.parent / "data" / "rag_documents"
COLLECTION = "inspection_standards"
EMBED_MODEL = "all-MiniLM-L6-v2"
_model = SentenceTransformer(EMBED_MODEL)

def _chunk(text, size=500, overlap=50):
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap
    return [c.strip() for c in chunks if c.strip()]

def _col():
    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False, allow_reset=True),
    )
    return client.get_or_create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})

def _embed(texts):
    return _model.encode(texts, batch_size=64).tolist()

def ingest_documents(docs_dir=None):
    docs_dir = Path(docs_dir) if docs_dir else DOCS_DIR
    col = _col()
    if col.count() > 0:
        print(f"[RAG] Already {col.count()} chunks — skipping ingest")
        return col.count()
    ids, docs, metas = [], [], []
    for pdf in sorted(docs_dir.glob("*.pdf")):
        text = ""
        with pdfplumber.open(pdf) as p:
            for page in p.pages:
                text += (page.extract_text() or "") + "\n"
        for i, chunk in enumerate(_chunk(text)):
            ids.append(f"{pdf.stem}_{i:04d}")
            docs.append(chunk)
            metas.append({"source": pdf.name})
    if not ids:
        print("[RAG] No PDFs found")
        return 0
    print(f"[RAG] Embedding {len(ids)} chunks…")
    col.upsert(ids=ids, documents=docs, embeddings=_embed(docs), metadatas=metas)
    print(f"[RAG] Done — {len(ids)} chunks stored")
    return len(ids)

def get_context(query):
    try:
        col = _col()
        if col.count() == 0:
            return ""
        vec = _embed([query])[0]
        results = col.query(
            query_embeddings=[vec],
            n_results=min(3, col.count()),
            include=["documents", "metadatas"],
        )
        parts = [
            f"[{m['source']}]\n{d}"
            for d, m in zip(results["documents"][0], results["metadatas"][0])
        ]
        return "\n\n".join(parts)
    except Exception as e:
        print(f"[RAG] get_context failed: {e}")
        return ""

def get_stats() -> dict:
    try:
        col = _col()
        sample = col.peek(3)
        return {
            "total_chunks": col.count(),
            "sources": list(set(m["source"] for m in sample["metadatas"]))
        }
    except Exception as e:
        return {"error": str(e)}
