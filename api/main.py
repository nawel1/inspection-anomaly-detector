import sys
sys.path.append(".")

from fastapi import FastAPI, UploadFile, File
import joblib
import pandas as pd
import tempfile
import os

from src.extractor import extract_from_pdf
from src.features import build_features

app = FastAPI(title="QIMA Inspection Anomaly API")

model  = joblib.load("models/isolation_forest.pkl")
scaler = joblib.load("models/scaler.pkl")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        df, metadata = extract_from_pdf(tmp_path)

        if df.empty:
            return {
                "metadata": metadata,
                "total_inspected": 0,
                "anomalies_detected": 0,
                "results": []
            }

        X = build_features(df)
        X_scaled = scaler.transform(X)

        # Risk score continu pour le ranking
        df["risk_score"] = [-round(float(s), 3) for s in model.decision_function(X_scaled)]

        # Règle métier ASTM D5430 — seuil de 20 pts
        df["anomaly"] = (
            (pd.to_numeric(df["total_penalty_points"], errors="coerce").fillna(0) > 20) |
            (pd.to_numeric(df["points_per_100sqyd"],   errors="coerce").fillna(0) > 20)
        )
        df["alert"] = df["anomaly"].apply(lambda x: "HIGH RISK" if x else "NORMAL")

        # Final conclusion toujours déduite depuis les données
        metadata["final_conclusion"] = (
            "NOT CONFORM" if df["anomaly"].any() else "CONFORM"
        )

        # Remplace NaN/Inf par None pour JSON
        df = df.replace({float("nan"): None, float("inf"): None, float("-inf"): None})

        return {
            "metadata": metadata,
            "total_inspected": len(df),
            "anomalies_detected": int(df["anomaly"].sum()),
            "results": df.to_dict(orient="records")
        }

    finally:
        os.unlink(tmp_path)