import os
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Malaysian economic sentiment", layout="wide")

DATA_PATH = "data/results/dashboard_data.csv"
MODEL_DIR = "final_model"          # optional: unzipped model folder, only for the "Try it" box
COLORS = {"negative": "#e5534b", "neutral": "#8b8b85", "positive": "#6aa84f"}
ORDER = ["negative", "neutral", "positive"]

st.markdown("""
<style>
.card{background:#242424;border:1px solid #333;border-radius:10px;padding:14px 18px}
.card .lbl{color:#bdbdbd;font-size:.85rem}
.card .val{font-size:1.9rem;font-weight:600;line-height:1.2}
.alert{background:#3a2a10;border:1px solid #6b4a12;border-radius:10px;padding:12px 18px;margin:10px 0}
.alert b{color:#f0a030}
.post{display:flex;justify-content:space-between;border-bottom:1px solid #333;padding:8px 0;gap:12px}
.post:last-child{border-bottom:0}
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load():
    d = pd.read_csv(DATA_PATH)
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    return d


try:
    df = load()
except FileNotFoundError:
    st.error(f"Cannot find {DATA_PATH}. Run the app from the repo folder (fyp-sentiment-analysis).")
    st.stop()

text_col = next((c for c in ["clean_text", "text"] if c in df.columns), None)

# ---------- header + filters ----------
h1, h2, h3 = st.columns([3, 2, 2])
h1.title("Malaysian economic sentiment dashboard")
rater = h2.radio("Show", ["Model predictions", "Human labels"], horizontal=True)
only_conflict = h3.checkbox("US-Iran conflict period only (from 28 Feb 2026)", value=True)
col = "model_label" if rater == "Model predictions" else "label"

f1, f2 = st.columns([2, 2])
src_opts = sorted(df["source_type"].dropna().unique())
src = f1.multiselect("Account type", src_opts, default=src_opts)
themes = sorted(df["theme"].dropna().unique())
thm = f2.multiselect("Theme", themes, default=themes)

d = df[df["source_type"].isin(src) & df["theme"].isin(thm)]
if only_conflict:
    d = d[d["in_2026_conflict"].astype(str).str.lower().isin(["true", "1"])]
if d.empty:
    st.warning("No tweets match these filters.")
    st.stop()

counts = d[col].value_counts().reindex(ORDER, fill_value=0)
share = counts / counts.sum()
dominant = share.idxmax()
overall = dominant.capitalize() if share.max() >= 0.45 else "Mixed"

# ---------- KPI row ----------
k1, k2, k3 = st.columns(3)
k1.markdown(f'<div class="card"><div class="lbl">Tweets analysed</div><div class="val">{len(d):,}</div></div>', unsafe_allow_html=True)
k2.markdown(f'<div class="card"><div class="lbl">Overall sentiment</div><div class="val">{overall}</div></div>', unsafe_allow_html=True)
k3.markdown(f'<div class="card" style="background:#3a1616;border-color:#6b2020"><div class="lbl" style="color:#f08080">Negative sentiment</div><div class="val" style="color:#f08080">{share["negative"]:.0%}</div></div>', unsafe_allow_html=True)

# ---------- charts ----------
c1, c2 = st.columns([1, 1.4])
with c1:
    st.subheader("Sentiment distribution")
    fig = go.Figure(go.Pie(labels=[s.capitalize() for s in ORDER], values=counts.values, hole=0.55,
                           marker_colors=[COLORS[s] for s in ORDER], sort=False, textinfo="percent"))
    fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=320, paper_bgcolor="rgba(0,0,0,0)",
                      legend=dict(orientation="h", y=-0.05))
    st.plotly_chart(fig, width="stretch")

monthly = None
if d["date"].notna().any():
    m = d.dropna(subset=["date"]).copy()
    m["month"] = m["date"].dt.to_period("M").dt.to_timestamp()
    g = m.groupby("month")[col].agg(n="size", neg=lambda s: (s == "negative").sum()).reset_index()
    g["neg_pct"] = g["neg"] / g["n"] * 100
    monthly = g
with c2:
    st.subheader("Negative share by month")
    if monthly is not None and len(monthly) > 1:
        fig = go.Figure(go.Scatter(x=monthly["month"], y=monthly["neg_pct"], mode="lines+markers",
                                   line=dict(color=COLORS["negative"], width=3),
                                   customdata=monthly["n"],
                                   hovertemplate="%{x|%b %Y}<br>%{y:.0f}% negative<br>n=%{customdata} tweets<extra></extra>"))
        fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=320, paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)", yaxis=dict(title="% negative", range=[0, 100]))
        st.plotly_chart(fig, width="stretch")
        st.caption("Each month has only a few dozen tweets (see n in the hover), so treat small moves as noise.")
    else:
        st.info("Not enough months in the current filter to draw a trend.")

# ---------- alert (computed, not hard-coded) ----------
if monthly is not None and len(monthly) > 1:
    big = monthly[monthly["n"] >= 20]
    if not big.empty:
        top = big.loc[big["neg_pct"].idxmax()]
        st.markdown(f'<div class="alert"><b>Highest negative share:</b> {top["month"]:%B %Y} at {top["neg_pct"]:.0f}% '
                    f'({int(top["neg"])} of {int(top["n"])} tweets). Only months with 20+ tweets are considered.</div>',
                    unsafe_allow_html=True)

# ---------- by theme ----------
st.subheader("Sentiment by theme")
t = pd.crosstab(d["theme"], d[col]).reindex(columns=ORDER, fill_value=0)
t = t.loc[t.sum(axis=1).sort_values().index]
fig = go.Figure()
for s in ORDER:
    fig.add_bar(y=t.index, x=t[s] / t.sum(axis=1) * 100, name=s.capitalize(), orientation="h",
                marker_color=COLORS[s], customdata=t[s], hovertemplate="%{y}: %{customdata} tweets<extra></extra>")
fig.update_layout(barmode="stack", height=320, margin=dict(t=10, b=10, l=10, r=10),
                  paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", xaxis=dict(title="% of tweets"))
st.plotly_chart(fig, width="stretch")

# ---------- recent posts ----------
if text_col:
    st.subheader("Sample of tweets")
    sample = d.sort_values("date", ascending=False).head(8)
    rows = ""
    for _, r in sample.iterrows():
        lab = str(r[col])
        txt = str(r[text_col]).replace("<", "&lt;")[:110]
        rows += f'<div class="post"><span>{txt}</span><span style="color:{COLORS.get(lab, "#aaa")}">{lab.capitalize()}</span></div>'
    st.markdown(f'<div class="card">{rows}</div>', unsafe_allow_html=True)

# ---------- optional: try the model ----------
st.subheader("Try the model")
if os.path.isdir(MODEL_DIR):
    @st.cache_resource
    def get_clf():
        from transformers import pipeline
        return pipeline("text-classification", model=MODEL_DIR, tokenizer=MODEL_DIR, top_k=None)
    txt = st.text_area("Type a tweet about the economy (Malay or English)")
    if st.button("Classify") and txt.strip():
        for r in get_clf()(txt)[0]:
            st.write(f"{r['label'].capitalize()}: {r['score']:.0%}")
else:
    st.caption("Optional: unzip final_model.zip into this folder (so a `final_model` folder sits next to app.py) and install "
               "`transformers` and `torch` to enable live predictions.")

# ---------- honest footer ----------
st.caption("Predictions are out-of-fold (each tweet was predicted by a model that never saw it in training). "
           "Macro F1 is about 0.58 and the neutral class is the weakest. About 70% of tweets come from news or official "
           "accounts. Themes are keyword-based. Data: dashboard_data.csv in the project's GitHub repo.")
