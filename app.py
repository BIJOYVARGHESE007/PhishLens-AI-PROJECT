import json, sqlite3, os, base64, datetime as dt
import joblib, numpy as np, pandas as pd, streamlit as st, plotly.graph_objects as go
from features import features_vector, extract_features, indicators, FEATURE_NAMES, normalize

st.set_page_config(page_title="PhishLens AI", page_icon="assets/logo.svg", layout="wide")
DB = "scan_history.db"

def b64(path):
    return base64.b64encode(open(path, "rb").read()).decode()
HERO, LOGO = b64("assets/hero.svg"), b64("assets/logo.svg")

CSS = """<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
html, body, [class*="css"] {font-family:'Inter',sans-serif;}
#MainMenu, footer, header[data-testid="stHeader"] {visibility:hidden;}
.block-container {padding-top:1.2rem; max-width:1200px;}
.hero {display:flex; align-items:center; gap:24px; padding:34px 40px; border-radius:24px; border:1px solid #243556;
 background:radial-gradient(circle at 85% 20%, rgba(99,102,241,.35), transparent 45%), linear-gradient(135deg,#0f1a30,#0b1220);}
.hero h1 {font-size:3rem; font-weight:800; margin:0; background:linear-gradient(90deg,#22d3ee,#818cf8); -webkit-background-clip:text; -webkit-text-fill-color:transparent;}
.hero p {color:#a8b6d1; font-size:1.1rem; max-width:520px; margin:10px 0 18px;}
.hero img {width:430px; max-width:42%;}
.pill {display:inline-block; padding:5px 14px; border-radius:99px; background:#12203d; border:1px solid #2a3d63; color:#7dd3fc; font-size:.8rem; margin-right:8px;}
.card {background:#111a2e; border:1px solid #243556; border-radius:18px; padding:20px 22px; height:100%;}
.card h4 {margin:0 0 6px; color:#e6edf7;} .card p {margin:0; color:#9fb0cc; font-size:.92rem;}
.num {width:34px; height:34px; border-radius:10px; background:linear-gradient(135deg,#22d3ee,#6366f1); color:#06101f; font-weight:800; display:flex; align-items:center; justify-content:center; margin-bottom:10px;}
.verdict {border-radius:20px; padding:22px 26px; border:1px solid; margin-bottom:12px;}
.verdict .t {font-size:1.7rem; font-weight:800; margin:0;} .verdict .s {color:#c3cee3; margin:4px 0 0;}
.v-bad {background:rgba(239,68,68,.12); border-color:#ef4444; color:#fca5a5;}
.v-mid {background:rgba(245,158,11,.12); border-color:#f59e0b; color:#fcd34d;}
.v-ok {background:rgba(34,197,94,.12); border-color:#22c55e; color:#86efac;}
.chip {display:block; padding:10px 14px; border-radius:12px; margin:6px 0; font-size:.93rem; border-left:4px solid; background:#111a2e;}
.c-high {border-color:#ef4444;} .c-medium {border-color:#f59e0b;} .c-low {border-color:#eab308;}
.kpi {background:#111a2e; border:1px solid #243556; border-radius:16px; padding:16px 18px; text-align:center;}
.kpi b {display:block; font-size:1.8rem; color:#22d3ee;} .kpi span {color:#9fb0cc; font-size:.85rem;}
.stTabs [data-baseweb="tab-list"] {gap:6px;} .stTabs [data-baseweb="tab"] {background:#111a2e; border-radius:12px; padding:8px 18px;}
.stTabs [aria-selected="true"] {background:linear-gradient(135deg,#0e7490,#4338ca);}
.foot {text-align:center; color:#6b7c9c; font-size:.8rem; margin-top:30px;}
</style>"""
st.markdown(CSS, unsafe_allow_html=True)

@st.cache_resource
def load():
    art = joblib.load("models/phishlens_rf.joblib")
    metrics = json.load(open("models/metrics.json")) if os.path.exists("models/metrics.json") else None
    return art, metrics
art, metrics = load()
model, baseline = art["model"], np.array(art["baseline"])

def db():
    c = sqlite3.connect(DB)
    c.execute("CREATE TABLE IF NOT EXISTS scans(ts TEXT, url TEXT, score REAL, verdict TEXT)")
    return c

def score(url):
    return float(model.predict_proba([features_vector(url)])[0][1])

def verdict(p):
    return "Potentially phishing" if p >= 0.7 else ("Suspicious - verify manually" if p >= 0.4 else "Likely legitimate")

def drivers(url, k=4):
    x = np.array(features_vector(url), dtype=float); base = score(url); out = []
    for i, n in enumerate(FEATURE_NAMES):
        x2 = x.copy(); x2[i] = baseline[i]
        drop = base - float(model.predict_proba([x2])[0][1])
        if drop > 0.005: out.append((n, drop))
    return sorted(out, key=lambda t: -t[1])[:k]

def gauge(p):
    color = "#ef4444" if p >= .7 else "#f59e0b" if p >= .4 else "#22c55e"
    fig = go.Figure(go.Indicator(mode="gauge+number", value=round(p * 100, 1), number={"suffix": "%", "font": {"color": color}},
        gauge={"axis": {"range": [0, 100], "tickcolor": "#6b7c9c"}, "bar": {"color": color, "thickness": .3}, "bgcolor": "#0b1220",
               "steps": [{"range": [0, 40], "color": "#12301f"}, {"range": [40, 70], "color": "#3a2c0f"}, {"range": [70, 100], "color": "#3a1414"}]}))
    fig.update_layout(height=250, margin=dict(t=20, b=0, l=20, r=20), paper_bgcolor="rgba(0,0,0,0)", font={"color": "#e6edf7"})
    return fig

def kpi(label, value):
    return f'<div class="kpi"><b>{value}</b><span>{label}</span></div>'

# ---------- Sidebar ----------
EXAMPLES = {"Fake verification (IP host)": "http://192.0.2.10/account/verify?user=12345",
            "Brand look-alike": "http://paypal-login-482.xyz/verify/index.php?id=4412",
            "Normal website": "https://www.wikipedia.org/wiki/Machine_learning"}
def setx(u): st.session_state["url_in"] = u
with st.sidebar:
    st.markdown(f'<img src="data:image/svg+xml;base64,{LOGO}" width="56"> <b style="font-size:1.4rem">PhishLens AI</b>', unsafe_allow_html=True)
    st.caption("Explainable phishing-link checker")
    st.markdown("**Try an example**")
    for name, u in EXAMPLES.items():
        st.button(name, on_click=setx, args=(u,), use_container_width=True)
    st.markdown("---")
    st.markdown("**Safe by design**  \nOnly the URL text is analysed. The website is never opened.")
    st.markdown("**Tip**  \nNever type a password on a page because a tool said it looks fine.")

# ---------- Hero ----------
st.markdown(f"""<div class="hero"><div><span class="pill">Machine Learning</span><span class="pill">Random Forest</span><span class="pill">Explainable</span>
<h1>PhishLens AI</h1><p>Paste a link before you click it. PhishLens inspects the structure of the URL, scores its phishing risk and tells you exactly why it looks suspicious.</p></div>
<img src="data:image/svg+xml;base64,{HERO}"></div>""", unsafe_allow_html=True)
st.write("")

h1, h2, h3 = st.columns(3)
h1.markdown('<div class="card"><div class="num">1</div><h4>Paste a link</h4><p>Copy the suspicious URL from an SMS, email or chat. Do not open it.</p></div>', unsafe_allow_html=True)
h2.markdown('<div class="card"><div class="num">2</div><h4>AI analyses 21 signals</h4><p>Length, IP host, keywords, hyphens, HTTPS, entropy and more feed a Random Forest.</p></div>', unsafe_allow_html=True)
h3.markdown('<div class="card"><div class="num">3</div><h4>Get a clear verdict</h4><p>See a risk score, plain-language warnings and what drove the result.</p></div>', unsafe_allow_html=True)
st.write("")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Scan a URL", "Batch scan", "History", "Model performance", "About & limits"])

with tab1:
    url = st.text_input("Link to check", key="url_in", placeholder="http://192.0.2.10/account/verify?user=12345", label_visibility="collapsed")
    go_btn = st.button("Analyze link", type="primary")
    if go_btn and url.strip():
        p = score(url); v = verdict(p)
        cls, sub = (("v-bad", "High similarity to known phishing patterns. Do not enter any details.") if p >= .7 else
                    ("v-mid", "Some warning signs found. Verify through the official website.") if p >= .4 else
                    ("v-ok", "Few warning signs in the URL structure. Stay careful anyway."))
        c1, c2 = st.columns([1, 1.25])
        with c1:
            st.markdown(f'<div class="verdict {cls}"><p class="t">{v}</p><p class="s">{sub}</p></div>', unsafe_allow_html=True)
            st.plotly_chart(gauge(p), use_container_width=True)
        with c2:
            st.markdown("#### Warning indicators")
            ind = indicators(url)
            if not ind: st.markdown('<div class="chip c-low">No rule-based warning signs found.</div>', unsafe_allow_html=True)
            for sev, msg in ind: st.markdown(f'<div class="chip c-{sev}"><b>{sev.upper()}</b> &nbsp; {msg}</div>', unsafe_allow_html=True)
            d = drivers(url)
            if d:
                st.markdown("#### What pushed the score up")
                fig = go.Figure(go.Bar(x=[x[1] * 100 for x in d][::-1], y=[x[0] for x in d][::-1], orientation="h", marker_color="#22d3ee"))
                fig.update_layout(height=210, margin=dict(t=5, b=0, l=0, r=0), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                  xaxis_title="risk points", font={"color": "#e6edf7"})
                st.plotly_chart(fig, use_container_width=True)
        st.caption("Decision support only. Not proof that a site is malicious or safe.")
        with st.expander("View extracted features"):
            st.dataframe(pd.DataFrame([extract_features(url)]).T.rename(columns={0: "value"}), use_container_width=True)
        c = db(); c.execute("INSERT INTO scans VALUES (?,?,?,?)", (dt.datetime.now().isoformat(timespec="seconds"), normalize(url), p, v)); c.commit(); c.close()

with tab2:
    st.markdown("Paste one URL per line, or upload a CSV with a `url` column. Results are ranked by risk.")
    txt = st.text_area("URLs", height=130)
    up = st.file_uploader("CSV file", type="csv")
    urls = [u.strip() for u in txt.splitlines() if u.strip()]
    if up is not None: urls += pd.read_csv(up)["url"].dropna().astype(str).tolist()
    if st.button("Scan all", type="primary") and urls:
        urls = urls[:500]
        res = pd.DataFrame({"url": urls, "risk": [score(u) for u in urls]})
        res["verdict"] = res.risk.map(verdict); res = res.sort_values("risk", ascending=False)
        a, b, c_ = st.columns(3)
        a.markdown(kpi("URLs scanned", len(res)), unsafe_allow_html=True)
        b.markdown(kpi("Flagged phishing", int((res.risk >= .7).sum())), unsafe_allow_html=True)
        c_.markdown(kpi("Likely legitimate", int((res.risk < .4).sum())), unsafe_allow_html=True)
        st.dataframe(res, use_container_width=True)
        st.download_button("Download results (CSV)", res.to_csv(index=False), "phishlens_results.csv")

with tab3:
    c = db(); h = pd.read_sql("SELECT * FROM scans ORDER BY ts DESC LIMIT 200", c); c.close()
    if len(h):
        a, b, c_ = st.columns(3)
        a.markdown(kpi("Total scans", len(h)), unsafe_allow_html=True)
        b.markdown(kpi("Flagged", int((h.score >= .7).sum())), unsafe_allow_html=True)
        c_.markdown(kpi("Average risk", f"{h.score.mean()*100:.0f}%"), unsafe_allow_html=True)
        st.dataframe(h, use_container_width=True)
    else: st.info("No scans yet. Analyze a link to start your history.")

with tab4:
    if not metrics: st.info("Run train.py first.")
    else:
        st.warning(f"Metrics come from `{metrics['dataset']}`. If this is the synthetic demo file, the numbers are not meaningful. Retrain on a real dataset.")
        k = st.columns(5)
        for col, (lbl, key) in zip(k, [("Accuracy", "accuracy"), ("Precision", "precision"), ("Recall", "recall"), ("F1-score", "f1"), ("ROC-AUC", "roc_auc")]):
            col.markdown(kpi(lbl, f"{metrics[key]:.3f}"), unsafe_allow_html=True)
        cm = metrics["confusion_matrix"]
        fig = go.Figure(go.Heatmap(z=cm, x=["Pred legit", "Pred phishing"], y=["Actual legit", "Actual phishing"], text=cm,
                                   texttemplate="%{text}", colorscale=[[0, "#0f1a30"], [1, "#22d3ee"]], showscale=False))
        fig.update_layout(height=330, title="Confusion matrix (test set)", paper_bgcolor="rgba(0,0,0,0)", font={"color": "#e6edf7"}); fig.update_yaxes(autorange="reversed")
        fi = pd.Series(metrics["feature_importance"]).head(10)[::-1]
        fig2 = go.Figure(go.Bar(x=fi.values, y=fi.index, orientation="h", marker_color="#818cf8"))
        fig2.update_layout(height=330, title="Top feature importances", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={"color": "#e6edf7"})
        l, r = st.columns(2); l.plotly_chart(fig, use_container_width=True); r.plotly_chart(fig2, use_container_width=True)
        st.caption(f"Training rows: {metrics['n_train']}  |  Test rows: {metrics['n_test']}")
        if metrics["sample_false_positives"]:
            st.markdown("**Legitimate URLs wrongly flagged (false positives)**"); st.code("\n".join(metrics["sample_false_positives"]))

with tab5:
    a, b = st.columns(2)
    a.markdown('<div class="card"><h4>How it works</h4><p>21 URL features (length, dots, digits, hyphens, IP host, HTTPS, sensitive keywords, entropy...) are fed to a Random Forest trained on labelled phishing and legitimate URLs. A what-if analysis shows which features raised the risk.</p></div>', unsafe_allow_html=True)
    b.markdown('<div class="card"><h4>Limitations</h4><p>Structure-only analysis. It does not check domain age, page content, ownership or threat-intelligence feeds, and can produce false positives and false negatives. It is an educational prototype, not a replacement for commercial email or endpoint security.</p></div>', unsafe_allow_html=True)

st.markdown('<div class="foot">PhishLens AI - educational phishing-URL triage prototype</div>', unsafe_allow_html=True)
