import streamlit as st
import requests
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Inspection Monitor", layout="wide")
st.title("Inspection Anomaly Monitor")

API_URL = "http://api:8000"

try:
    requests.get(f"{API_URL}/health", timeout=2)
    st.success("API connected")
except:
    st.error("API not reachable — make sure the API is running")

uploaded = st.file_uploader("Upload an inspection report (PDF)", type=["pdf"])

if uploaded:
    with st.spinner("Analyzing report... this may take 30-60 seconds"):
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
        data    = response.json()
        meta    = data["metadata"]
        results = pd.DataFrame(data["results"])

        # KPIs
        st.subheader("Report summary")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("PO Number",          meta.get("po_number",        "N/A"))
        col2.metric("Final conclusion",   meta.get("final_conclusion", "N/A"))
        col3.metric("Rolls analyzed",     data["total_inspected"])
        col4.metric("Anomalies detected", data["anomalies_detected"])

        if results.empty:
            st.warning("No rolls extracted from this report.")
            st.stop()

        # report type detection
        is_generic = results["roll_no"].isna().all()

        if is_generic:
            results = results.rename(columns={"color": "product"})
            display_cols = [c for c in [
                "product", "total_penalty_points",
                "risk_score", "alert", "defects"
            ] if c in results.columns]
            x_col = "product"
        else:
            display_cols = [c for c in [
                "roll_no", "color", "total_penalty_points",
                "points_per_100sqyd", "risk_score", "alert"
            ] if c in results.columns]
            results["label"] = results.apply(
                lambda r: f"{r['color']} #{int(r['roll_no'])}"
                if pd.notna(r.get("roll_no")) else r["color"],
                axis=1
            )
            x_col = "label"

        # table results
        sort_col = "risk_score" if "risk_score" in results.columns else display_cols[0]
        st.subheader("Roll-by-roll results")
        st.dataframe(
            results[display_cols].sort_values(sort_col, ascending=False),
            use_container_width=True,
            hide_index=True
        )

        # Charts
        if "risk_score" in results.columns and "alert" in results.columns:
            col5, col6 = st.columns(2)

            with col5:
                st.subheader("Risk score distribution")
                fig = px.histogram(
                    results, x="risk_score", color="alert",
                    color_discrete_map={"HIGH RISK": "red", "NORMAL": "green"}
                )
                st.plotly_chart(fig, use_container_width=True)

            with col6:
                if "total_penalty_points" in results.columns:
                    st.subheader("Penalty points by roll")
                    fig2 = px.bar(
                        results.sort_values("total_penalty_points", ascending=False),
                        x=x_col,
                        y="total_penalty_points",
                        color="alert",
                        color_discrete_map={"HIGH RISK": "red", "NORMAL": "green"}
                    )
                    fig2.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig2, use_container_width=True)

        # Defects breakdown
        if "defects" in results.columns:
            st.subheader("Defect breakdown")
            defect_map = {}
            for d in results["defects"].dropna():
                if d and d != "None":
                    for item in d.split(","):
                        k = item.strip().title()  # normalise la casse
                        if k:
                            defect_map[k] = defect_map.get(k, 0) + 1

            if defect_map:
                defect_df = pd.DataFrame(
                    list(defect_map.items()),
                    columns=["defect", "count"]
                ).sort_values("count", ascending=False)
                fig3 = px.bar(
                    defect_df, x="defect", y="count",
                    color_discrete_sequence=["#378ADD"]
                )
                st.plotly_chart(fig3, use_container_width=True)

    else:
        st.error(f"API error {response.status_code} — {response.text}")