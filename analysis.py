"""Logic for the policymaker upload dashboard.

No Streamlit code lives here, so every rule can be read and tested on its own.
All thresholds are constants at the top: change them here, not in app.py.
"""
import re
import numpy as np
import pandas as pd

ORDER = ["negative", "neutral", "positive"]

# ---- settings you can explain to an examiner ----
UNCERTAIN_BELOW = 0.60   # a prediction under 60% confidence is flagged "uncertain"
MIN_THEME_N = 10         # a theme needs at least this many tweets before we comment on it
HIGH_NEG = 0.60          # negative share >= 60%  -> High priority
MEDIUM_NEG = 0.45        # negative share >= 45%  -> Medium priority
TREND_MIN_N = 15         # each of the two weeks compared needs at least this many tweets
TREND_POINT_MIN = 5      # trend chart hides periods with fewer posts than this (too noisy)
TREND_JUMP = 0.10        # +10 percentage points of negative share counts as "rising"
GEO_GAP = 0.10           # geopolitical tweets this much more negative than the rest -> flag
UNCERTAIN_WARN = 0.30    # warn if more than 30% of predictions are uncertain
MAX_ROWS = 5000          # keeps CPU runtime reasonable
# Update this line whenever you re-run the evaluation:
MODEL_NOTE = "a fine-tuned XLM-RoBERTa model (about 0.68 macro F1 in cross-validation on 1,839 labelled tweets)"

TEXT_GUESSES = ["clean_text", "text", "full_text", "tweet", "content", "message", "comment", "post", "body"]
DATE_GUESSES = ["created_at", "date", "datetime", "timestamp", "time", "posted_at", "createdat"]

THEME_RULES = [
    ("cost_of_living", r"kos sara hidup|sara hidup|cost of living"),
    ("inflation", r"inflasi|inflation"),
    ("jobs", r"gaji|pekerjaan|upah|pengangguran|pekerja|jobs|salary"),
    ("food", r"makanan|beras|ayam|telur|tepung|gandum|baja|food"),
    ("ringgit", r"ringgit|dolar|mata wang"),
    ("fuel", r"minyak|petrol|diesel|ron95|ron97|budi95|bahan api|subsidi|brent|hormuz"),
]

GEO_PATTERN = (r"\b(?:perang|war|iran|israel|hormuz|gaza|palestin\w*|ukrain\w*|rusia|russia|konflik|conflict|"
               r"sekatan|sanctions?|tarif|tariffs?|asia barat|middle east|timur tengah|laut merah|red sea|"
               r"houthi|opec|geopolitik|geopolitical)\b")

ACTIONS = {
    "fuel": "Review fuel and subsidy messaging: explain who is affected, by how much and until when. Check whether subsidy targeting is understood.",
    "food": "Check retail price monitoring for staples (chicken, eggs, rice, flour) and consider targeted enforcement or aid for the items people mention.",
    "cost_of_living": "Communicate relief measures (cash aid, rebates) and who qualifies. Look at how lower-income households are affected.",
    "inflation": "Publish a short, plain-language explanation of the latest inflation data and which items drive it.",
    "jobs": "Review employment signals: job vacancies, retrenchment reports, wage concerns. Consider visible job-support measures.",
    "ringgit": "Address ringgit and import-cost concerns with clear communication from the relevant ministry or central bank.",
    "other": "These tweets did not match a known theme. Read the example tweets to find the underlying issue.",
}
PRIORITY_RANK = {"High": 0, "Medium": 1, "Watch": 2, "Info": 3}


# ---------- reading and cleaning ----------
def guess_column(columns, guesses):
    lower = {c.lower(): c for c in columns}
    for g in guesses:
        if g in lower:
            return lower[g]
    return None


def clean_text(t):
    t = re.sub(r"https?://\S+", "", str(t))
    return re.sub(r"\s+", " ", t).strip()


def parse_dates(series):
    s = series.astype(str)
    d = pd.to_datetime(s, format="%a %b %d %H:%M:%S %z %Y", errors="coerce", utc=True)
    if d.isna().mean() > 0.5:
        d = pd.to_datetime(s, errors="coerce", utc=True)
    return d.dt.tz_convert(None)


def prepare(raw, text_col, date_col=None, drop_dupes=True, max_rows=MAX_ROWS):
    """Return (frame with 'text' and 'date', info about what was dropped)."""
    info = {"rows_in": len(raw), "too_short": 0, "duplicates": 0, "capped": 0}
    f = pd.DataFrame({"text": raw[text_col].map(clean_text)})
    f["date"] = parse_dates(raw[date_col]) if date_col else pd.NaT
    short = f["text"].str.len() < 3
    info["too_short"] = int(short.sum())
    f = f[~short]
    if drop_dupes:
        dup = f["text"].duplicated()
        info["duplicates"] = int(dup.sum())
        f = f[~dup]
    if len(f) > max_rows:
        info["capped"] = len(f) - max_rows
        f = f.iloc[:max_rows]
    return f.reset_index(drop=True), info


# ---------- classification ----------
_ALIAS = {"label_0": "negative", "label_1": "neutral", "label_2": "positive"}


def make_predict_fn(clf):
    """Wrap a Hugging Face text-classification pipeline (top_k=None) as text list -> probabilities."""
    def predict(batch):
        out = clf(list(batch), truncation=True, max_length=128, batch_size=len(batch))
        P = np.zeros((len(batch), 3))
        for i, scores in enumerate(out):
            d = {}
            for s in scores:
                k = str(s["label"]).lower()
                d[_ALIAS.get(k, k)] = float(s["score"])
            P[i] = [d.get(k, 0.0) for k in ORDER]
        return P
    return predict


def classify(texts, predict_fn, batch_size=32, progress=None):
    probs = []
    for i in range(0, len(texts), batch_size):
        probs.append(predict_fn(texts[i:i + batch_size]))
        if progress:
            progress(min(i + batch_size, len(texts)) / len(texts))
    P = np.vstack(probs) if probs else np.zeros((0, 3))
    return np.array(ORDER)[P.argmax(1)], P.max(1)


def assign_theme(t):
    t = str(t).lower()
    for name, pat in THEME_RULES:
        if re.search(pat, t):
            return name
    return "other"


def analyse(frame, predict_fn, progress=None):
    res = frame.copy()
    res["label"], res["confidence"] = classify(res["text"].tolist(), predict_fn, progress=progress)
    res["uncertain"] = res["confidence"] < UNCERTAIN_BELOW
    res["theme"] = res["text"].map(assign_theme)
    res["geopolitical"] = res["text"].str.lower().str.contains(GEO_PATTERN, regex=True)
    return res


# ---------- summaries ----------
def shares(res):
    c = res["label"].value_counts().reindex(ORDER, fill_value=0)
    return c / c.sum() if c.sum() else c


def theme_table(res):
    t = pd.crosstab(res["theme"], res["label"]).reindex(columns=ORDER, fill_value=0)
    t["n"] = t.sum(axis=1)
    t["neg_share"] = t["negative"] / t["n"]
    return t.sort_values("n", ascending=False)


def trend(res):
    d = res.dropna(subset=["date"])
    if d.empty or d["date"].nunique() < 2:
        return None
    span = (d["date"].max() - d["date"].min()).days
    freq, name = ("D", "day") if span <= 31 else (("W", "week") if span <= 180 else ("M", "month"))
    g = d.groupby(d["date"].dt.to_period(freq).dt.to_timestamp())["label"].agg(
        n="size", neg=lambda s: (s == "negative").sum()).reset_index()
    g.columns = ["period", "n", "neg"]
    g["neg_share"] = g["neg"] / g["n"]
    g = g[g["n"] >= TREND_POINT_MIN]
    if len(g) < 2:
        return None
    return g, name


def recent_vs_previous(res):
    d = res.dropna(subset=["date"])
    if d.empty:
        return None
    end = d["date"].max()
    if (end - d["date"].min()).days < 13:
        return None
    rec = d[d["date"] > end - pd.Timedelta(days=7)]
    prev = d[(d["date"] <= end - pd.Timedelta(days=7)) & (d["date"] > end - pd.Timedelta(days=14))]
    if len(rec) < TREND_MIN_N or len(prev) < TREND_MIN_N:
        return None
    return float((prev["label"] == "negative").mean()), float((rec["label"] == "negative").mean()), len(prev), len(rec)


def evidence(res, theme, k=3):
    d = res[(res["theme"] == theme) & (res["label"] == "negative")].sort_values("confidence", ascending=False)
    return d.head(k)["text"].str.slice(0, 200).tolist()


def recommendations(res):
    """Plain if-then rules on the model's output. These are prompts for a human to look closer, not forecasts."""
    recs = []
    tt = theme_table(res)
    for theme, r in tt.iterrows():
        if r["n"] < MIN_THEME_N:
            continue
        ns = r["neg_share"]
        if ns >= HIGH_NEG:
            pr = "High"
        elif ns >= MEDIUM_NEG:
            pr = "Medium"
        else:
            continue
        recs.append({"priority": pr, "topic": theme,
                     "finding": f"{ns:.0%} of {int(r['n'])} tweets about '{theme}' are negative.",
                     "action": ACTIONS.get(theme, ACTIONS["other"]), "evidence": evidence(res, theme)})

    geo, rest = res[res["geopolitical"]], res[~res["geopolitical"]]
    if len(geo) >= MIN_THEME_N and len(rest) >= MIN_THEME_N:
        g, o = (geo["label"] == "negative").mean(), (rest["label"] == "negative").mean()
        if g - o >= GEO_GAP:
            recs.append({"priority": "Medium", "topic": "geopolitical events",
                         "finding": f"Tweets mentioning a geopolitical event are more negative ({g:.0%} of {len(geo)}) than the rest ({o:.0%} of {len(rest)}).",
                         "action": "Geopolitical events appear to be driving concern. Prioritise clear communication on fuel and price stability and on supply plans.",
                         "evidence": geo[geo["label"] == "negative"].sort_values("confidence", ascending=False).head(3)["text"].str.slice(0, 200).tolist()})

    rv = recent_vs_previous(res)
    if rv:
        prev, rec, n1, n2 = rv
        if rec - prev >= TREND_JUMP:
            recs.append({"priority": "High", "topic": "trend",
                         "finding": f"Negative share rose from {prev:.0%} (previous 7 days, n={n1}) to {rec:.0%} (latest 7 days, n={n2}).",
                         "action": "Negative sentiment is rising. Check what happened in the latest week and whether a response is needed.",
                         "evidence": []})

    if res["uncertain"].mean() > UNCERTAIN_WARN:
        recs.append({"priority": "Watch", "topic": "data quality",
                     "finding": f"{res['uncertain'].mean():.0%} of predictions are uncertain (confidence below {UNCERTAIN_BELOW:.0%}).",
                     "action": "Treat the percentages as rough. Read the example tweets or review a sample by hand before acting.",
                     "evidence": []})
    if not recs:
        recs.append({"priority": "Info", "topic": "overall",
                     "finding": "No theme or trend passed the attention thresholds.",
                     "action": "Keep monitoring. Upload newer data regularly to see changes.",
                     "evidence": []})
    return sorted(recs, key=lambda r: PRIORITY_RANK[r["priority"]])


def summary_markdown(res, recs, info, source_name="uploaded file"):
    s = shares(res)
    lines = [f"# Sentiment summary: {source_name}", "",
             f"Tweets analysed: {len(res)} (rows in file: {info['rows_in']}, too short: {info['too_short']}, "
             f"duplicates removed: {info['duplicates']}, not analysed because of the {MAX_ROWS}-row limit: {info['capped']})", "",
             f"Negative {s['negative']:.0%} | Neutral {s['neutral']:.0%} | Positive {s['positive']:.0%} | "
             f"Uncertain predictions {res['uncertain'].mean():.0%}", "", "## What needs attention", ""]
    for r in recs:
        lines += [f"**[{r['priority']}] {r['topic']}**: {r['finding']}", f"- Suggested next step: {r['action']}"]
        for e in r["evidence"]:
            lines.append(f"- Example: \"{e}\"")
        lines.append("")
    lines += ["## Notes", "",
              f"Sentiment comes from {MODEL_NOTE}. "
              "Themes are keyword-based. Suggested steps come from simple threshold rules and are prompts for review, not forecasts."]
    return "\n".join(lines)
