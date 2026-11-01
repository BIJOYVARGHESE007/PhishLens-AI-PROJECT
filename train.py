"""Train the Random Forest. Usage: python train.py [path/to/urls.csv]
CSV needs columns: url,label  (label: 1 = phishing, 0 = legitimate; text labels are standardised)."""
import sys, json
import pandas as pd, numpy as np, joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (precision_score, recall_score, f1_score, accuracy_score,
                             confusion_matrix, roc_auc_score)
from features import features_vector, FEATURE_NAMES

path = sys.argv[1] if len(sys.argv) > 1 else "data/urls.csv"
df = pd.read_csv(path)
df.columns = [c.strip().lower() for c in df.columns]

# --- Module 1: clean & standardise ---
df = df[["url", "label"]].dropna().drop_duplicates(subset="url")
map_ = {"phishing": 1, "bad": 1, "malicious": 1, "1": 1, "legitimate": 0, "good": 0, "benign": 0, "0": 0}
df["label"] = df["label"].astype(str).str.strip().str.lower().map(map_)
df = df.dropna(subset=["label"]); df["label"] = df["label"].astype(int)
print(f"Clean rows: {len(df)} | phishing share: {df.label.mean():.2%}")

# --- Module 2: features ---
X = np.array([features_vector(u) for u in df.url]); y = df.label.values
Xtr, Xte, ytr, yte, utr, ute = train_test_split(X, y, df.url.values, test_size=0.2, stratify=y, random_state=42)

# --- Module 3: model ---
model = RandomForestClassifier(n_estimators=300, max_depth=None, min_samples_leaf=2,
                               class_weight="balanced", n_jobs=-1, random_state=42)
model.fit(Xtr, ytr)
pred, proba = model.predict(Xte), model.predict_proba(Xte)[:, 1]

metrics = {
    "accuracy": accuracy_score(yte, pred), "precision": precision_score(yte, pred),
    "recall": recall_score(yte, pred), "f1": f1_score(yte, pred), "roc_auc": roc_auc_score(yte, proba),
    "confusion_matrix": confusion_matrix(yte, pred).tolist(),
    "n_train": int(len(ytr)), "n_test": int(len(yte)),
    "feature_importance": dict(sorted(zip(FEATURE_NAMES, map(float, model.feature_importances_)), key=lambda x: -x[1])),
    "dataset": path,
}
# false positives = legit sites flagged as phishing (worth reporting honestly)
fp = ute[(pred == 1) & (yte == 0)][:20].tolist()
metrics["sample_false_positives"] = fp

baseline = np.median(Xtr[ytr == 0], axis=0).tolist()  # "typical legitimate URL" for what-if explanations
joblib.dump({"model": model, "features": FEATURE_NAMES, "baseline": baseline}, "models/phishlens_rf.joblib", compress=3)
json.dump(metrics, open("models/metrics.json", "w"), indent=2)
print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in metrics.items()
                  if k not in ("feature_importance", "sample_false_positives")}, indent=2))
