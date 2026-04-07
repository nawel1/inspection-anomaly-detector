import os
import re
import json
import tempfile
import pdfplumber
from mistralai import Mistral
from rag import get_context

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
MODEL = "mistral-small-latest"

def analyze_report(pdf_path: str) -> dict:
    # Extract full text
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += (page.extract_text() or "") + "\n"

    context = get_context(text[:500])

    prompt = f"""You are a quality inspection expert. Analyze this inspection
report and determine if it passes or fails quality standards.

Reference context from industry standards:
{context}

Report text:
{text[:4000]}

Return ONLY valid JSON with these exact keys:
{{
  "overall_result": "PASS" or "FAIL",
  "confidence": "high", "medium", or "low",
  "main_defects": [
    severity must be exactly one of: "Critical", "Major", or "Minor"
    {{"name": "Big oil stain", "severity": "Major"}}
  ],
  "summary": "one sentence explaining the result"
}}
Keep max 5 defects. Empty list if PASS."""

    client = Mistral(api_key=MISTRAL_API_KEY)
    try:
        response = client.chat.complete(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
    except Exception as e:
        if "429" in str(e) or "capacity" in str(e).lower():
            return {
                "overall_result": "ERROR",
                "confidence": "low",
                "main_defects": [],
                "summary": "Mistral API rate limit reached — wait 1 minute and try again."
            }
        raise

    raw = response.choices[0].message.content
    raw = re.sub(r"```json|```", "", raw).strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {
            "overall_result": "UNKNOWN",
            "confidence": "low",
            "main_defects": [],
            "summary": "Could not parse the report."
        }
