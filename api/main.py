import sys
sys.path.append("/app/src")

import os
import tempfile
from fastapi import FastAPI, UploadFile, File
from contextlib import asynccontextmanager
from rag import ingest_documents, get_stats
from analyze import analyze_report

@asynccontextmanager
async def lifespan(app):
    ingest_documents()
    yield

app = FastAPI(title="Inspection Report Analyzer", lifespan=lifespan)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    try:
        return analyze_report(tmp_path)
    finally:
        os.unlink(tmp_path)

@app.get("/rag/stats")
def rag_stats():
    stats = get_stats()
    if "error" in stats:
        return {"status": "unavailable", "error": stats["error"]}
    return {"status": "ok", **stats}
