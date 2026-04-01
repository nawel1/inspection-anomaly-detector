import pandas as pd
import numpy as np

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["total_penalty_points"] = pd.to_numeric(df["total_penalty_points"] if "total_penalty_points" in df.columns else 0, errors="coerce").fillna(0)
    df["points_per_100sqyd"]   = pd.to_numeric(df["points_per_100sqyd"]   if "points_per_100sqyd"   in df.columns else 0, errors="coerce").fillna(0)
    df["length_ticketed"]      = pd.to_numeric(df["length_ticketed"]       if "length_ticketed"       in df.columns else 50, errors="coerce").fillna(50)
    df["length_actual"]        = pd.to_numeric(df["length_actual"]         if "length_actual"         in df.columns else 50, errors="coerce").fillna(50)

    df["penalty_ratio"]  = df["total_penalty_points"] / 20
    df["length_delta"]   = (df["length_actual"] - df["length_ticketed"]).abs()
    df["shade_flag"]     = df["shade_issue"].astype(int)  if "shade_issue"  in df.columns else 0
    df["narrow_flag"]    = df["narrow_width"].astype(int) if "narrow_width" in df.columns else 0
    df["severity_score"] = (
        df["penalty_ratio"] * 0.6 +
        df["shade_flag"]    * 0.2 +
        df["narrow_flag"]   * 0.1 +
        df["length_delta"]  * 0.1
    )

    feature_cols = [
        "total_penalty_points",
        "points_per_100sqyd",
        "penalty_ratio",
        "length_delta",
        "shade_flag",
        "narrow_flag",
        "severity_score",
    ]
    return df[feature_cols].fillna(0)