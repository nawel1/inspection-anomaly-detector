import pandas as pd
import joblib
import os
import sys
sys.path.append(".")
from src.features import build_features
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# Data for training 
# Prod : replace with real reports history
def generate_training_data(n: int = 500) -> pd.DataFrame:
    import numpy as np
    np.random.seed(42)

    # Correct data — conform rolls
    df_normal = pd.DataFrame({
        "total_penalty_points": np.random.randint(0, 15, int(n * 0.85)),
        "points_per_100sqyd":   np.random.uniform(0, 14, int(n * 0.85)),
        "length_ticketed":      np.random.uniform(48, 52, int(n * 0.85)),
        "length_actual":        np.random.uniform(48, 52, int(n * 0.85)),
        "shade_issue":          np.random.choice([True, False], int(n * 0.85), p=[0.2, 0.8]),
        "narrow_width":         np.random.choice([True, False], int(n * 0.85), p=[0.3, 0.7]),
    })

    # Incorrect data — non conforms rolls
    df_anomaly = pd.DataFrame({
        "total_penalty_points": np.random.randint(21, 80, int(n * 0.15)),
        "points_per_100sqyd":   np.random.uniform(20, 100, int(n * 0.15)),
        "length_ticketed":      np.random.uniform(48, 52, int(n * 0.15)),
        "length_actual":        np.random.uniform(48, 52, int(n * 0.15)),
        "shade_issue":          np.random.choice([True, False], int(n * 0.15), p=[0.6, 0.4]),
        "narrow_width":         np.random.choice([True, False], int(n * 0.15), p=[0.8, 0.2]),
    })

    df = pd.concat([df_normal, df_anomaly], ignore_index=True)
    return df.sample(frac=1, random_state=42).reset_index(drop=True)

    idx = np.random.choice(n, 50, replace=False)
    df.loc[idx, "total_penalty_points"] = np.random.randint(25, 80, 50)
    df.loc[idx, "points_per_100sqyd"]   = np.random.uniform(20, 100, 50)

    return df

def train():
    os.makedirs("models", exist_ok=True)

    df = generate_training_data()
    X  = build_features(df)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(
        n_estimators=200,
        contamination=0.15,
        random_state=42
    )
    model.fit(X_scaled)

    joblib.dump(model,  "models/isolation_forest.pkl")
    joblib.dump(scaler, "models/scaler.pkl")
    print(f"Model trained on {len(df)} samples and saved.")

if __name__ == "__main__":
    train()