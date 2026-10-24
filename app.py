import hashlib
import os

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import analysis as A

st.set_page_config(page_title="Malaysian economic sentiment", layout="wide")

DATA_PATH = "data/results/dashboard_data.csv"
MODEL_DIR = "final_model"          # unzipped model folder (not stored in GitHub, too large)
COLORS = {"negative": "#e5534b", "neutral": "#8b8b85", "positive": "#6aa84f"}
ORDER = A.ORDER
PRIORITY_COLOR = {"High": "#e5534b", "Medium": "#f0a030", "Watch": "#6b9bd1", "Info": "#8b8b85"}

st.markdown("""
<style>
.card{background:#242424;color:#eaeaea;border:1px solid #333;border-radius:10px;padding:14px 18px}
.card .lbl{color:#bdbdbd;font-size:.85rem}
.card .val{font-size:1.9rem;font-weight:600;line-height:1.2}
.alert{background:#3a2a10;color:#eaeaea;border:1px solid #6b4a12;border-radius:10px;padding:12px 18px;margin:10px 0}
.alert b{color:#f0a030}
.post{display:flex;justify-content:space-between;border-bottom:1px solid #333;padding:8px 0;gap:12px}
.post:last-child{border-bottom:0}
.rec{background:#242424;color:#eaeaea;border:1px solid #333;border-left-width:6px;border-radius:10px;padding:12px 18px;margin:10px 0}
.rec .tag{font-weight:600}
.rec .ex{color:#bdbdbd;font-size:.88rem;margin:2px 0 0 12px}
</style>
""", unsafe_allow_html=True)


def card(label, value, bg=None, fg=None):
    style = f' style="background:{bg};border-color:{fg}"' if bg else ""
    vstyle = f' style="color:{fg}"' if fg else ""
    return f'<div class="card"{style}><div class="lbl">{label}</div><div class="val"{vstyle}>{value}</div></div>'


@st.cache_resource
def get_predictor():
    from transformers import pipeline
    clf = pipeline("text-classification", model=MODEL_DIR, tokenizer=MODEL_DIR, top_k=None)
    return A.make_predict_fn(clf)


def read_upload(file):
    for enc in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            file.seek(0)
            return pd.read_csv(file, encoding=enc)
        except UnicodeDecodeError:
            continue
        except pd.errors.EmptyDataError:
            return None
    return None


# =====================================================================
# Tab 1: policymaker uploads a CSV
# =====================================================================
def render_upload_tab():
    st.markdown("Upload a CSV of social media posts. The dashboard classifies each post as negative, neutral or positive, "
                "groups the posts by theme, and suggests what to look at first.")
    with st.expander("What should the CSV look like?"):
        st.markdown("- One row per post, with a column that holds the **text** (required).\n"
                    "- A **date** column is optional, but it enables the trend chart and the week-on-week check.\n"
                    f"- Up to {A.MAX_ROWS:,} posts are analysed per upload. Duplicate texts can be removed automatically.\n"
                    "- Works with the files this project already produces, for example `data/raw/raw_tweets.csv`.")

    if not os.path.isdir(MODEL_DIR):
        st.error("The model folder `final_model` was not found next to app.py. Unzip final_model.zip into the project folder "
                 "and install `transformers` and `torch`.")
        return

    file = st.file_uploader("Upload CSV", type=["csv"])
    if file is None:
        st.info("No file yet. Upload a CSV to begin.")
        show_results(None)
        return

    raw = read_upload(file)
    if raw is None or raw.empty:
        st.error("Could not read this file as a CSV, or it has no rows.")
        return

    cols = list(raw.columns)
    guess_t = A.guess_column(cols, A.TEXT_GUESSES)
    guess_d = A.guess_column(cols, A.DATE_GUESSES)
    c1, c2, c3 = st.columns([2, 2, 2])
    text_col = c1.selectbox("Column with the post text", cols, index=cols.index(guess_t) if guess_t else 0)
    date_opts = ["(none)"] + cols
    date_col = c2.selectbox("Column with the date (optional)", date_opts, index=date_opts.index(guess_d) if guess_d else 0)
    dedupe = c3.checkbox("Remove duplicate texts", value=True)
    st.caption(f"{len(raw):,} rows read. Preview of the text column:")
    st.dataframe(raw[[text_col]].head(5), hide_index=True, width="stretch")

    sig = hashlib.md5((file.name + str(file.size) + text_col + str(date_col) + str(dedupe)).encode()).hexdigest()
    if st.button("Analyse", type="primary"):
        frame, info = A.prepare(raw, text_col, None if date_col == "(none)" else date_col, dedupe)
        if frame.empty:
            st.error("No usable text found in that column.")
            return
        if info["capped"]:
            st.warning(f"Only the first {A.MAX_ROWS:,} posts were analysed ({info['capped']:,} more were skipped).")
        bar = st.progress(0.0, text="Classifying posts. The first run loads the model and can take about 30 seconds.")
        predict = get_predictor()
        res = A.analyse(frame, predict, progress=lambda f: bar.progress(f, text=f"Classifying posts… {f:.0%}"))
        bar.empty()
        st.session_state["results"] = {"res": res, "info": info, "sig": sig, "name": file.name}

    stored = st.session_state.get("results")
    if stored and stored["sig"] == sig:
        show_results(stored)
    elif stored:
        st.caption("Settings changed since the last analysis. Press Analyse to update the results.")


def show_results(stored):
    if not stored:
        return
    res, info = stored["res"], stored["info"]
    s = A.shares(res)
    unc = res["uncertain"].mean()

    st.divider()
    k = st.columns(5)
    k[0].markdown(card("Posts analysed", f"{len(res):,}"), unsafe_allow_html=True)
    k[1].markdown(card("Negative", f"{s['negative']:.0%}", "#3a1616", "#f08080"), unsafe_allow_html=True)
    k[2].markdown(card("Neutral", f"{s['neutral']:.0%}"), unsafe_allow_html=True)
    k[3].markdown(card("Positive", f"{s['positive']:.0%}"), unsafe_allow_html=True)
    k[4].markdown(card("Uncertain predictions", f"{unc:.0%}"), unsafe_allow_html=True)
    notes = []
    if info["duplicates"]:
        notes.append(f"{info['duplicates']:,} duplicate texts removed")
    if info["too_short"]:
        notes.append(f"{info['too_short']:,} rows too short to analyse")
    if notes:
        st.caption(" · ".join(notes))

    # ---- what needs attention ----
    recs = A.recommendations(res)
    st.subheader("What needs attention")
    for r in recs:
        color = PRIORITY_COLOR[r["priority"]]
        ex = "".join(f'<div class="ex">“{e.replace("<", "&lt;")}”</div>' for e in r["evidence"])
        st.markdown(f'<div class="rec" style="border-left-color:{color}"><span class="tag" style="color:{color}">'
                    f'{r["priority"]}</span> · <b>{r["topic"].replace("_", " ")}</b><br>{r["finding"]}<br>'
                    f'<i>Suggested next step:</i> {r["action"]}{ex}</div>', unsafe_allow_html=True)
    st.caption("These suggestions come from simple threshold rules on the model's output (see the settings at the top of analysis.py). "
               "They tell you where to look first. They are not forecasts and need human judgement.")

    # ---- charts ----
    c1, c2 = st.columns([1, 1.4])
    with c1:
        st.subheader("Sentiment distribution")
        fig = go.Figure(go.Pie(labels=[x.capitalize() for x in ORDER], values=[s[x] for x in ORDER], hole=0.55,
                               marker_colors=[COLORS[x] for x in ORDER], sort=False, textinfo="percent"))
        fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=320, paper_bgcolor="rgba(0,0,0,0)",
                          legend=dict(orientation="h", y=-0.05))
        st.plotly_chart(fig, width="stretch")
    with c2:
        st.subheader("Negative share over time")
        tr = A.trend(res)
        if tr is None:
            st.info("No usable dates in this file, so no trend chart.")
        else:
            g, unit = tr
            fig = go.Figure(go.Scatter(x=g["period"], y=g["neg_share"] * 100, mode="lines+markers",
                                       line=dict(color=COLORS["negative"], width=3), customdata=g["n"],
                                       hovertemplate="%{x|%d %b %Y}<br>%{y:.0f}% negative<br>n=%{customdata} posts<extra></extra>"))
            fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=320, paper_bgcolor="rgba(0,0,0,0)",
                              plot_bgcolor="rgba(0,0,0,0)", yaxis=dict(title="% negative", range=[0, 100]))
            st.plotly_chart(fig, width="stretch")
            st.caption(f"Grouped by {unit}. Periods with fewer than {A.TREND_POINT_MIN} posts are hidden because they are too noisy. Hover to see how many posts each point is based on.")

    st.subheader("Sentiment by theme")
    t = A.theme_table(res)
    tp = t.loc[t["n"].sort_values().index]
    fig = go.Figure()
    for x in ORDER:
        fig.add_bar(y=tp.index, x=tp[x] / tp["n"] * 100, name=x.capitalize(), orientation="h",
                    marker_color=COLORS[x], customdata=tp[x], hovertemplate="%{y}: %{customdata} posts<extra></extra>")
    fig.update_layout(barmode="stack", height=320, margin=dict(t=10, b=10, l=10, r=10),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", xaxis=dict(title="% of posts"))
    st.plotly_chart(fig, width="stretch")

    geo = res[res["geopolitical"]]
    if len(geo):
        st.subheader("Posts that mention a geopolitical event")
        g1, g2 = st.columns(2)
        rest = res[~res["geopolitical"]]
        g1.markdown(card(f"Mention a geopolitical event (n={len(geo)})", f"{(geo['label'] == 'negative').mean():.0%} negative"), unsafe_allow_html=True)
        g2.markdown(card(f"Other posts (n={len(rest)})", f"{(rest['label'] == 'negative').mean():.0%} negative" if len(rest) else "n/a"), unsafe_allow_html=True)
        st.caption("Detected by keywords such as war, Iran, Hormuz, Gaza, Ukraine, sanctions and tariffs.")

    with st.expander("Read example posts by theme"):
        pick = st.selectbox("Theme", list(t.index))
        sub = res[res["theme"] == pick].sort_values("confidence", ascending=False).head(8)
        rows = ""
        for _, r in sub.iterrows():
            rows += (f'<div class="post"><span>{str(r["text"])[:140].replace("<", "&lt;")}</span>'
                     f'<span style="color:{COLORS[r["label"]]}">{r["label"].capitalize()} {r["confidence"]:.0%}</span></div>')
        st.markdown(f'<div class="card">{rows}</div>', unsafe_allow_html=True)

    # ---- downloads ----
    d1, d2 = st.columns(2)
    d1.download_button("Download results (CSV)", res.to_csv(index=False).encode("utf-8-sig"),
                       "sentiment_results.csv", "text/csv")
    d2.download_button("Download summary (Markdown)", A.summary_markdown(res, recs, info, stored["name"]).encode("utf-8"),
                       "sentiment_summary.md", "text/markdown")

    st.caption(f"Sentiment comes from {A.MODEL_NOTE}. Themes are keyword-based. Predictions under "
               f"{A.UNCERTAIN_BELOW:.0%} confidence are flagged as uncertain.")


# =====================================================================
# Tab 2: the case study dashboard (labelled data from the project)
# =====================================================================
@st.cache_data
def load_case():
    d = pd.read_csv(DATA_PATH)
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    return d


def render_case_study():
    try:
        df = load_case()
    except FileNotFoundError:
        st.error(f"Cannot find {DATA_PATH}. Run the app from the repo folder (fyp-sentiment-analysis).")
        return
    text_col = next((c for c in ["clean_text", "text"] if c in df.columns), None)

    h2, h3 = st.columns([2, 2])
    rater = h2.radio("Show", ["Model predictions", "Human labels"], horizontal=True)
    only_conflict = h3.checkbox("2026 period only (since 28 Feb 2026)", value=True)
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
        return

    counts = d[col].value_counts().reindex(ORDER, fill_value=0)
    share = counts / counts.sum()
    overall = share.idxmax().capitalize() if share.max() >= 0.45 else "Mixed"

    k1, k2, k3 = st.columns(3)
    k1.markdown(card("Tweets analysed", f"{len(d):,}"), unsafe_allow_html=True)
    k2.markdown(card("Overall sentiment", overall), unsafe_allow_html=True)
    k3.markdown(card("Negative sentiment", f"{share['negative']:.0%}", "#3a1616", "#f08080"), unsafe_allow_html=True)

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
                                       line=dict(color=COLORS["negative"], width=3), customdata=monthly["n"],
                                       hovertemplate="%{x|%b %Y}<br>%{y:.0f}% negative<br>n=%{customdata} tweets<extra></extra>"))
            fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=320, paper_bgcolor="rgba(0,0,0,0)",
                              plot_bgcolor="rgba(0,0,0,0)", yaxis=dict(title="% negative", range=[0, 100]))
            st.plotly_chart(fig, width="stretch")
            st.caption("Each month has only a few dozen tweets (see n in the hover), so treat small moves as noise.")
        else:
            st.info("Not enough months in the current filter to draw a trend.")

    if monthly is not None and len(monthly) > 1:
        big = monthly[monthly["n"] >= 20]
        if not big.empty:
            top = big.loc[big["neg_pct"].idxmax()]
            st.markdown(f'<div class="alert"><b>Highest negative share:</b> {top["month"]:%B %Y} at {top["neg_pct"]:.0f}% '
                        f'({int(top["neg"])} of {int(top["n"])} tweets). Only months with 20+ tweets are considered.</div>',
                        unsafe_allow_html=True)

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

    if text_col:
        st.subheader("Sample of tweets")
        sample = d.sort_values("date", ascending=False).head(8)
        rows = ""
        for _, r in sample.iterrows():
            lab = str(r[col])
            txt = str(r[text_col]).replace("<", "&lt;")[:110]
            rows += f'<div class="post"><span>{txt}</span><span style="color:{COLORS.get(lab, "#aaa")}">{lab.capitalize()}</span></div>'
        st.markdown(f'<div class="card">{rows}</div>', unsafe_allow_html=True)

    st.caption("Predictions are out-of-fold (each tweet was predicted by a model that never saw it in training). "
               f"Macro F1 is about 0.68 and the neutral class is still weaker than the others. About "
               f"{100 * (d['source_type'] == 'news_official').mean():.0f}% of tweets come from news or official accounts. "
               "Themes are keyword-based. Data: dashboard_data.csv in the project's GitHub repo.")


# =====================================================================
st.title("Malaysian economic sentiment dashboard")
tab_up, tab_case = st.tabs(["Analyse a CSV", "Case study: labelled tweets"])
with tab_up:
    render_upload_tab()
    if os.path.isdir(MODEL_DIR):
        with st.expander("Quick test: one post at a time"):
            txt = st.text_area("Type a post about the economy (Malay or English)")
            if st.button("Classify") and txt.strip():
                P = get_predictor()([A.clean_text(txt)])[0]
                if P.max() < A.UNCERTAIN_BELOW:
                    st.warning(f"Uncertain: the top class is only {P.max():.0%}. Treat this prediction with caution.")
                for i in P.argsort()[::-1]:
                    st.write(f"{ORDER[i].capitalize()}: {P[i]:.0%}")
with tab_case:
    render_case_study()
