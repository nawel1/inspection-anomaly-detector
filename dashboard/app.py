import streamlit as st
import requests

st.set_page_config(page_title="Inspection Report Analyzer", layout="centered")
st.title("Inspection Report Analyzer")

API_URL = "http://api:8000"

try:
    requests.get(f"{API_URL}/health", timeout=2)
    st.success("API connected")
except:
    st.error("API not reachable")

uploaded = st.file_uploader("Upload an inspection report (PDF)", type=["pdf"])

if uploaded:
    with st.spinner("Analyzing... this may take 30-60 seconds"):
        try:
            response = requests.post(
                f"{API_URL}/analyze",
                files={"file": (uploaded.name, uploaded.getvalue(), "application/pdf")},
                timeout=120
            )
        except requests.exceptions.Timeout:
            st.error("Request timed out — try again")
            st.stop()
        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to API")
            st.stop()

    if response.status_code == 200:
        data = response.json()

        result = data.get("overall_result", "UNKNOWN")
        color = "green" if result == "PASS" else "red"
        st.markdown(f"<h1 style='color:{color}'>{result}</h1>", unsafe_allow_html=True)

        st.markdown(f"**Confidence:** {data.get('confidence', 'N/A')}")
        st.markdown(f"**Summary:** {data.get('summary', 'N/A')}")

        defects = data.get("main_defects", [])
        if defects and data.get("overall_result") != "PASS":
            st.subheader("Main defects")
            import pandas as pd
            df = pd.DataFrame(defects)
            if "severity" in df.columns:
                def color_severity(val):
                    colors = {"Critical": "red", "Major": "orange", "Minor": "gold"}
                    return f"color: {colors.get(val, 'white')}"
                st.dataframe(
                    df.style.applymap(color_severity, subset=["severity"]),
                    use_container_width=True,
                    hide_index=True
                )
            else:
                for d in defects:
                    st.markdown(f"- {d}")
        else:
            st.success("No defects detected.")
    else:
        st.error(f"API error {response.status_code} — {response.text}")
