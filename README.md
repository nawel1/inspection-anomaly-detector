# Inspection Anomaly Detector

An end-to-end AI pipeline that analyzes quality inspection PDF reports, extracts structured data using a LLM, and flags anomalies using business rules and machine learning.

Built as a side project to demonstrate AI/ML engineering skills in the context of quality control and supply chain inspection.

---

## Demo

**Fabric inspection report (TESTCOO format)**
- 26 rolls extracted from a 28-page PDF
- PEACOCK #14 flagged HIGH RISK (53 penalty points)
- GREEN #54 flagged HIGH RISK (21.6 pts/100 sq.yd)
- Final conclusion: NOT CONFORM

**Generic inspection report**
- Product extracted from unstructured text
- Inspection score converted to penalty points
- Final conclusion: NOT CONFORM

---

## Architecture

```
PDF Upload
    │
    ▼
pdfplumber (text extraction)
    │
    ├── Structured tables found? ──► Direct DataFrame
    │
    └── No structure detected?  ──► Mistral API (LLM extraction)
                                        │
                                        ├── fabric prompt (roll-by-roll)
                                        └── generic prompt (score-based)
    │
    ▼
Feature Engineering (penalty ratio, severity score, length delta)
    │
    ▼
Isolation Forest (continuous risk score for ranking)
    │
    ▼
Business Rule — ASTM D5430 (pts > 20 = HIGH RISK)
    │
    ▼
FastAPI + Streamlit Dashboard
```

---

## Key Design Decisions

**LLM for extraction, rules for decisions**
The LLM (Mistral) handles unstructured document understanding. The final HIGH RISK / NORMAL decision always follows the ASTM D5430 business rule (penalty points > 20). This ensures auditability and consistency — critical in a quality control context.

**Prompt-per-format architecture**
The pipeline detects the report type (fabric inspection vs generic quality report) and applies the appropriate extraction prompt. Adding a new format takes ~20 minutes. The natural V2 would be a trained format classifier on a historical report catalog.

**Isolation Forest for ranking**
Beyond the binary HIGH RISK / NORMAL decision, the ML model provides a continuous risk score to prioritize rolls within a report. This is where ML adds value over pure rules — ranking and trend detection across multiple reports over time.

---

## Stack

| Layer | Technology |
|---|---|
| PDF extraction | pdfplumber |
| LLM extraction | Mistral API (mistral-small-latest) |
| Anomaly detection | scikit-learn Isolation Forest |
| API | FastAPI |
| Dashboard | Streamlit + Plotly |
| Containerization | Docker + Docker Compose |

---

## Project Structure

```
inspection-anomaly-detector/
├── data/
│   └── fabricinspectionsamplereport.pdf
├── models/
│   └── .gitkeep
├── src/
│   ├── extractor.py    # PDF extraction + Mistral fallback
│   ├── features.py     # Feature engineering
│   └── train.py        # Isolation Forest training
├── api/
│   └── main.py         # FastAPI endpoint
├── dashboard/
│   └── app.py          # Streamlit dashboard
├── Dockerfile
├── start.sh
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

---

## Getting Started

**Prerequisites**
- Docker Desktop
- Mistral API key (free tier at console.mistral.ai)

**Setup**

1. Clone the repo
```bash
git clone https://github.com/yourusername/inspection-anomaly-detector
cd inspection-anomaly-detector
```

2. Create your `.env` file
```bash
cp .env.example .env
# Add your Mistral API key
```

3. Train the model
```bash
python src/train.py
```

4. Build and run
```bash
docker-compose up --build
```

5. Open the dashboard
```
http://localhost:8501
```

API docs available at `http://localhost:8000/docs`

---

## Supported Report Formats

| Format | Detection | Extraction |
|---|---|---|
| Fabric inspection (4-point system) | Keyword detection | Mistral page-by-page |
| Generic quality report (score-based) | Keyword detection | Mistral full-text |
| Scanned PDFs | — | Not supported (OCR roadmap) |

---

## Roadmap

- Format classifier trained on historical reports (replace keyword detection)
- OCR support for scanned PDFs (Tesseract)
- Multi-report trend analysis (supplier risk scoring over time)
- Unit tests and CI/CD pipeline

---

## Why This Architecture

This project was built with a production mindset:

- **LLM handles ambiguity** — document understanding, format variations, unstructured text
- **Rules handle decisions** — auditable, explainable, legally defensible
- **ML handles ranking** — continuous scoring for prioritization, extensible to trend detection

This separation is the standard approach for AI-assisted quality control systems where explainability and compliance matter.