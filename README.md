# SMS Spam Detector

End-to-end ML classifier for the [UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) dataset. Predicts whether a text message is spam or legitimate using TF-IDF + hand-crafted features with Logistic Regression. Ships as a Streamlit web app with threshold tuning and feature explainability.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download data and train model (saves model.joblib)
python train.py

# 3. Launch the app
streamlit run app.py
```

Then open http://localhost:8501.

## Project Structure

| File | Purpose |
|------|---------|
| `features.py` | Shared `MessageFeatureExtractor`, `to_dataframe`, `build_preprocessor` |
| `train.py` | CLI training script: downloads data, trains 3 models, evaluates, saves best |
| `app.py` | Streamlit web app |
| `notebook.ipynb` | Full EDA → training → evaluation → error analysis walkthrough |
| `model.joblib` | Pre-trained Logistic Regression pipeline (committed to repo) |
| `requirements.txt` | Python dependencies |
| `model_card.md` | Limitations, risks, evaluation metrics |
| `data/SMSSpamCollection` | Raw dataset (auto-downloaded by `train.py`) |

## Pipeline Architecture

```
Raw text → FunctionTransformer(to_dataframe) → ColumnTransformer
                                                 ├── TfidfVectorizer (unigrams+bigrams, ≤10000 features)
                                                 └── MessageFeatureExtractor → MaxAbsScaler (7 numeric features)
                                               → LogisticRegression
```

**Numeric features:** character length, word count, uppercase count, uppercase ratio, punctuation count, has currency symbol, digit count.

## Model Performance

Run `python train.py` to see actual metrics. Expected results (Logistic Regression on 20% stratified test set):

| Metric | Spam | Ham | Weighted |
|--------|------|-----|---------|
| Precision | ~0.99 | ~0.98 | ~0.98 |
| Recall | ~0.91 | ~1.00 | ~0.98 |
| F1 | ~0.95 | ~0.99 | ~0.98 |
| ROC-AUC | ~0.99 | | |

## Training Options

```bash
python train.py --help

optional arguments:
  --data-dir DATA_DIR       Dataset directory (default: data)
  --model-output PATH       Output path for model.joblib (default: model.joblib)
  --test-size FLOAT         Test split fraction (default: 0.2)
  --random-state INT        Random seed (default: 42)
  --no-plots                Skip saving plot PNGs
```

## Running the Notebook

```bash
pip install jupyter
jupyter notebook notebook.ipynb
```

## Deploy to Streamlit Community Cloud

1. Push this repo to GitHub (include `model.joblib`)
2. Go to https://share.streamlit.io and sign in with GitHub
3. New app → select repo → main file: `app.py` → Deploy

No training step is needed in the cloud — `model.joblib` is pre-built and committed.

## Key Design Decisions

- **`features.py` as shared module:** Required for joblib deserialization. If `to_dataframe` or `MessageFeatureExtractor` were defined as lambdas or inside `__main__`, loading `model.joblib` in `app.py` would fail with a `ModuleNotFoundError`.
- **`MaxAbsScaler` not `StandardScaler`:** Keeps all numeric features ≥ 0, satisfying MultinomialNB's strict non-negativity requirement.
- **`loss='modified_huber'` for SGDClassifier:** Enables `predict_proba`, required for threshold tuning and PR curves. `loss='hinge'` does not support this.
- **`class_weight='balanced'`:** Compensates for the 87%/13% class imbalance without undersampling.

## License

Dataset: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
Code: MIT
