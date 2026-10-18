# AI-Based Sentiment Analysis System for Malaysian Public Opinion on the Current Economy

Final Year Project (FYP II), BIT (Hons) Big Data Analytics, Universiti Teknologi PETRONAS.

The system collects tweets about the Malaysian economy (fuel prices, food prices, inflation, cost of living, jobs, ringgit),
classifies their sentiment (negative, neutral, positive) with a fine-tuned XLM-RoBERTa model, and shows the results on a
dashboard for policymakers. The US-Iran conflict (from 28 Feb 2026) is the case study.

## Pipeline

1. **Collect** tweets with GetXAPI -> `data/raw/`
2. **Clean** (Malaysian tweets only, URLs removed, short and duplicate tweets dropped) -> `data/cleaned/`
3. **Label** (negative / neutral / positive, AI-assisted and reviewed by one person) -> `data/labeling/`
4. **Fine-tune and evaluate** XLM-RoBERTa (`cardiffnlp/twitter-xlm-roberta-base-sentiment`) in Google Colab -> `data/results/`
5. **Dashboard** (Streamlit) -> `app.py`

## Folder guide

| Path | What it is |
|---|---|
| `app.py`, `requirements.txt`, `.streamlit/` | The Streamlit dashboard |
| `data/raw/raw_tweets.csv` | Tweets exactly as collected |
| `data/cleaned/cleaned_tweets.csv` | Tweets after cleaning |
| `data/labeling/labeled_tweets.csv` | Final labeled dataset (404 tweets) |
| `data/labeling/new_tweets_to_label*.csv` | Batches of tweets that were labeled in rounds |
| `data/results/cv_predictions.csv` | Out-of-fold model predictions from cross-validation |
| `data/results/dashboard_data.csv` | Data the dashboard reads (predictions, human labels, date, theme, account type) |

## Results

Evaluated with 5-fold cross-validation grouped by author, repeated over 5 different splits.

- Macro F1: about 0.58 (+/- 0.02), against about 0.50 for the untrained model
- Agreement with human labels: about 61%
- Weakest class: neutral (event and political news that the model gives a polarity to)

## Limitations

- About 70% of labeled tweets come from news or official accounts, not individuals.
- Labels were AI-assisted and reviewed by a single person.
- The language filter is approximate, so some Indonesian tweets may remain.
- Dashboard themes are keyword-based.

## Run the dashboard

```
pip install -r requirements.txt
python -m streamlit run app.py
```

The trained model is not stored in this repository because of its size (about 1 GB).
To enable the "Try the model" box, place the model in a folder named `final_model` next to `app.py`.
