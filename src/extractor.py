import pdfplumber
import pandas as pd
import json
import re
import os
from mistralai import Mistral

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
MODEL = "mistral-small-latest"

PROMPTS = {
    "fabric": """You are a data extraction assistant for fabric inspection reports.
Extract ALL roll inspection records from this page.
The text may have spaces removed between words (e.g. "RollNo./Color" instead of "Roll No. / Color").
Return ONLY a valid JSON array, no markdown, no explanation.
If no roll data found, return [].

Each object must have exactly these keys:
- roll_no (integer or null)
- color (string)
- length_ticketed (float or null)
- length_actual (float or null)
- total_penalty_points (integer or null)
- points_per_100sqyd (float or null)
- defects (string, comma-separated defect types or "None")
- shade_issue (boolean, true if shade issue mentioned)
- narrow_width (boolean, true if narrow width mentioned)

Page text:
""",

    "generic": """You are a data extraction assistant for quality inspection reports.
Extract the inspection record from the text below.
Return ONLY a valid JSON array with one object, no markdown, no explanation.

Each object must have exactly these keys:
- roll_no (null)
- color (string, product name or "Not applicable")
- length_ticketed (null)
- length_actual (null)
- total_penalty_points (integer, convert inspection score to penalty points: score < 70 = 30, score 70-85 = 15, score > 85 = 5)
- points_per_100sqyd (null)
- defects (string, comma-separated list of defects found)
- shade_issue (false)
- narrow_width (false)

Text:
"""
}

def detect_report_type(text: str) -> str:
    text_lower = text.lower()
    if any(k in text_lower for k in ["penalty points", "4 point system", "rollno", "roll no"]):
        return "fabric"
    elif any(k in text_lower for k in ["inspection score", "defects found", "recommendation"]):
        return "generic"
    else:
        return "generic"

def extract_raw_text(pdf_path: str) -> str:
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

# Extraction 

def extract_with_pdfplumber(pdf_path: str) -> pd.DataFrame | None:
    rows = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            blocks = re.findall(
                r"Roll No\.\s*/\s*Color\s+(\d+)\s+([A-Z\s]+)\n"
                r"Piece Length.*?(\d+\.?\d*)\s+(\d+\.?\d*)\n"
                r".*?Total penalty points.*?(\d+)\n"
                r"Points per 100 sq\. yards.*?(\d+\.?\d*)",
                text,
                re.DOTALL
            )
            for match in blocks:
                roll, color, len_ticket, len_actual, pts, pts_per100 = match
                rows.append({
                    "roll_no":              int(roll),
                    "color":                color.strip(),
                    "length_ticketed":      float(len_ticket),
                    "length_actual":        float(len_actual),
                    "total_penalty_points": int(pts),
                    "points_per_100sqyd":   float(pts_per100),
                })

    if not rows:
        return None
    return pd.DataFrame(rows)

# Mistral API — page per page

def extract_with_mistral(pdf_path: str) -> pd.DataFrame:
    client  = Mistral(api_key=MISTRAL_API_KEY)
    records = []

    full_text   = extract_raw_text(pdf_path)
    report_type = detect_report_type(full_text)
    print(f"[Extractor] Report type detected: {report_type}")

    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""

            if report_type == "fabric":
                text_clean = text.lower().replace(" ", "")
                if "rollno." not in text_clean and "rollno" not in text_clean:
                    continue

            print(f"[Mistral] Processing page {i+1} ({report_type})...")

            prompt = PROMPTS[report_type] + text[:3000]

            try:
                response = client.chat.complete(
                    model=MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0
                )

                raw = response.choices[0].message.content
                raw = re.sub(r"```json|```", "", raw).strip()
                parsed = json.loads(raw)

                if isinstance(parsed, list):
                    for r in parsed:
                        if r.get("color"):
                            records.append(r)
                            print(f"[Mistral] ✓ {r.get('color')} #{r.get('roll_no')} — pts: {r.get('total_penalty_points')}")
                elif isinstance(parsed, dict) and parsed.get("color"):
                    records.append(parsed)
                    print(f"[Mistral] ✓ {parsed.get('color')} #{parsed.get('roll_no')} — pts: {parsed.get('total_penalty_points')}")

            except (json.JSONDecodeError, Exception) as e:
                print(f"[Mistral] Error on page {i+1}: {e}")
                continue

    if records:
        df = pd.DataFrame(records)
        df = df[df["total_penalty_points"].notna()]
        df["color"] = df["color"].str.strip().str.upper()
        df = df.sort_values("total_penalty_points", ascending=False)
        df = df.drop_duplicates(subset=["roll_no", "color"], keep="first")
        df = df.reset_index(drop=True)
        return df

    return pd.DataFrame()

# Metadata Extraction  (regex)

def extract_metadata(pdf_path: str) -> dict:
    metadata = {
        "po_number":            None,
        "total_length":         None,
        "rolls_inspected":      None,
        "final_conclusion":     None,
        "total_penalty_points": None,
    }

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages[:5]:
            text = page.extract_text() or ""

            if not metadata["po_number"]:
                m = re.search(r"P\.O\.\s*Number\s+(\w+)", text)
                if m:
                    metadata["po_number"] = m.group(1)

            if not metadata["total_length"]:
                m = re.search(r"Total Fabric Length\s+([\d\s,]+)\s*Meters", text)
                if m:
                    metadata["total_length"] = int(m.group(1).replace(",", "").strip())

            if not metadata["rolls_inspected"]:
                m = re.search(r"Total Rolls/Pieces Inspected:\s*(\d+)", text)
                if m:
                    metadata["rolls_inspected"] = int(m.group(1))

            if not metadata["total_penalty_points"]:
                m = re.search(r"Total Penalty Points:\s*(\d+)", text)
                if m:
                    metadata["total_penalty_points"] = int(m.group(1))

    return metadata

def extract_from_pdf(pdf_path: str) -> tuple[pd.DataFrame, dict]:
    print(f"[Extractor] Reading: {pdf_path}")

    metadata = extract_metadata(pdf_path)
    print(f"[Extractor] Metadata: {metadata}")

    df = extract_with_pdfplumber(pdf_path)

    if df is not None and len(df) >= 5:
        print(f"[Extractor] pdfplumber: {len(df)} rolls extracted")
    else:
        print("[Extractor] pdfplumber insufficient → fallback Mistral API (page by page)")
        df = extract_with_mistral(pdf_path)
        print(f"[Extractor] Mistral API: {len(df)} rolls extracted")

    return df, metadata