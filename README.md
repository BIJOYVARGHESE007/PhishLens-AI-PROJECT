# PhishLens AI - Explainable Phishing URL Detector

Paste a link *before* clicking it. PhishLens AI extracts 21 structural features from the URL text
(never visiting the site), scores it with a Random Forest, and explains why it looks risky.

**Unique features:** risk gauge with 3 bands (not just a yes/no), plain-language warning flags,
"what pushed the score up" what-if explanations, batch scan with CSV download, SQLite scan history,
and an honest performance tab that lists false positives.

## Run locally
```bash
python -m venv .venv
.venv\Scripts\activate          # Windows   (Mac/Linux: source .venv/bin/activate)
pip install -r requirements.txt
# 1. Put a REAL dataset at data/urls.csv  (columns: url,label)   -- see below
#    (for a quick smoke test only:  python make_demo_data.py   -> synthetic data)
python train.py                 # trains and writes models/phishlens_rf.joblib + metrics.json
streamlit run app.py
```

## Real datasets (use one for your report)
- Kaggle: "Phishing Site URLs" or "Malicious URLs dataset" (url + label columns)
- PhishTank / OpenPhish feeds for phishing URLs + Tranco top-sites list for legitimate ones
Rename the columns to `url,label`. Text labels (phishing/benign/good/bad) are standardised automatically.

## Deploy free (Streamlit Community Cloud)
1. Create a GitHub repo and push this folder (commit `models/` too, or train on startup).
2. Go to share.streamlit.io, sign in with GitHub, click **Create app**.
3. Pick the repo, branch `main`, main file `app.py`, then **Deploy**. You get a public link.
Alternative: Hugging Face Spaces (SDK: Streamlit) - upload the same files.

## Limits
URL-structure only. No domain age, page content, ownership or threat-intel checks. Expect false positives.
Metrics must come from your real dataset, not the synthetic demo file.
