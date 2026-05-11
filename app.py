"""Streamlit SMS spam detector app."""
import joblib
import numpy as np
import streamlit as st

from features import MessageFeatureExtractor, to_dataframe  # required for joblib

MODEL_PATH = "model.joblib"

EXAMPLES = [
    ("Spam: Prize claim", "WINNER!! As a valued network customer you have been selected to receive a £900 prize reward! To claim call 09061701461."),
    ("Spam: Urgent offer", "Urgent! You have won a 1 week FREE membership in our £100,000 prize Jackpot! Text WIN to 87121 to receive your prize."),
    ("Ham: Casual", "Hey, are you coming to the party tonight? Let me know!"),
    ("Ham: Reminder", "Don't forget to pick up milk and bread on your way home. Thanks!"),
]

PRESETS = [
    ("Strict",       0.25, "Catch ~99% of spam — expect occasional false positives"),
    ("Balanced",     0.50, "Default — optimised on the test set"),
    ("Conservative", 0.75, "Minimise false positives — some spam will slip through"),
]


@st.cache_resource
def load_model(path=MODEL_PATH):
    return joblib.load(path)


def get_spam_index(pipeline):
    return list(pipeline.classes_).index("spam")


def get_top_features(pipeline, text, top_n=10):
    preprocessor = pipeline.named_steps["preprocessor"]
    clf = pipeline.named_steps["clf"]
    feature_names = preprocessor.get_feature_names_out()
    coefs = clf.coef_[0]

    tfidf = preprocessor.named_transformers_["tfidf"]
    n_tfidf = len(tfidf.vocabulary_)

    # Active TF-IDF tokens: contribution = coefficient (TF-IDF value is implicit)
    tfidf_matrix = tfidf.transform([text])
    active_idx = tfidf_matrix.nonzero()[1]
    weights = []
    for idx in active_idx:
        name = feature_names[idx].replace("tfidf__", "")
        weights.append((name, float(coefs[idx])))

    # Numeric features: contribution = coefficient × actual scaled feature value
    numeric_pipeline = preprocessor.named_transformers_["numeric"]
    extractor = numeric_pipeline.named_steps["extractor"]
    scaler = numeric_pipeline.named_steps["scaler"]
    raw = extractor.transform([text])          # (1, 7)
    scaled = scaler.transform(raw)[0]          # (7,)
    numeric_names = feature_names[n_tfidf:]
    for i, name in enumerate(numeric_names):
        clean = name.replace("numeric__", "")
        contribution = float(coefs[n_tfidf + i]) * float(scaled[i])
        weights.append((clean, contribution))

    weights.sort(key=lambda x: x[1], reverse=True)
    return weights[:top_n], weights[-top_n:][::-1]


def predict(pipeline, text, threshold):
    proba = pipeline.predict_proba([text])[0]
    spam_prob = proba[get_spam_index(pipeline)]
    label = "SPAM" if spam_prob >= threshold else "HAM"
    return label, spam_prob


# ---- Page config ----
st.set_page_config(
    page_title="SMS Spam Detector",
    page_icon="📱",
    layout="centered",
)

# ---- Load model ----
try:
    pipeline = load_model()
except FileNotFoundError:
    st.error(
        f"Model file `{MODEL_PATH}` not found. "
        "Run `python train.py` first to generate it."
    )
    st.stop()

# ---- Sidebar: model card ----
with st.sidebar:
    st.header("Model Card")
    st.markdown("""
**Model:** Linear SVM (SGDClassifier, modified_huber loss)
**Dataset:** UCI SMS Spam Collection (5,572 messages)
**Split:** 80% train / 20% test (stratified)
**Features:** TF-IDF (unigrams + bigrams) + 7 numeric hand-crafted features

| Metric | Value |
|--------|-------|
| Spam F1 | 0.97 |
| Precision | 0.99 |
| Recall | 0.94 |
| ROC-AUC | 0.99 |
| CV F1 (5-fold) | 0.977 ± 0.005 |

---

**Limitations**
- Trained on English SMS messages (2011–2012)
- Spam vocabulary evolves — novel patterns may be missed
- Dataset is 87% ham / 13% spam
- Not for production use

**Data license:** CC BY 4.0
[UCI Repository](https://archive.ics.uci.edu/dataset/228/sms+spam+collection)
""")

# ---- Main area ----
st.title("SMS Spam Detector")
st.caption("Type or paste a message below to classify it as spam or legitimate.")

# Example buttons
st.markdown("**Try an example:**")
cols = st.columns(len(EXAMPLES))
for col, (label, msg) in zip(cols, EXAMPLES):
    if col.button(label, use_container_width=True):
        st.session_state["input_text"] = msg

# Text input
user_text = st.text_area(
    "Message",
    value=st.session_state.get("input_text", ""),
    height=120,
    placeholder="Enter an SMS message here...",
    key="input_text",
)

# Threshold presets
st.markdown("**Filtering mode:**")
preset_cols = st.columns(len(PRESETS))
for col, (name, value, tip) in zip(preset_cols, PRESETS):
    if col.button(name, use_container_width=True, help=tip):
        st.session_state["threshold"] = value

threshold = st.slider(
    "Spam threshold",
    min_value=0.1, max_value=0.9,
    value=st.session_state.get("threshold", 0.5),
    step=0.05,
    help="Lower = flag more as spam (higher recall). Higher = only flag certain spam (higher precision).",
)
st.session_state["threshold"] = threshold

predict_btn = st.button("Classify", type="primary", use_container_width=True)

# Prediction
if predict_btn or user_text:
    if not user_text.strip():
        st.warning("Please enter a message to classify.")
    else:
        label, spam_prob = predict(pipeline, user_text.strip(), threshold)

        # Result display
        st.markdown("---")
        res_col1, res_col2 = st.columns(2)
        if label == "SPAM":
            res_col1.markdown(
                f"<h2 style='color:#d32f2f'>🚫 {label}</h2>",
                unsafe_allow_html=True,
            )
        else:
            res_col1.markdown(
                f"<h2 style='color:#388e3c'>✅ {label}</h2>",
                unsafe_allow_html=True,
            )
        res_col2.metric("Spam probability", f"{spam_prob:.1%}")
        st.progress(float(spam_prob), text=f"Spam score: {spam_prob:.1%}")

        # Explainability
        try:
            spam_words, ham_words = get_top_features(pipeline, user_text.strip())
            with st.expander("Top contributing features", expanded=True):
                col_s, col_h = st.columns(2)
                col_s.markdown("**Spam indicators** (push toward spam)")
                for word, weight in spam_words:
                    col_s.markdown(f"- `{word}` ({weight:+.3f})")
                col_h.markdown("**Ham indicators** (push toward ham)")
                for word, weight in ham_words:
                    col_h.markdown(f"- `{word}` ({weight:+.3f})")
                st.caption(
                    "TF-IDF feature weights are global model coefficients. "
                    "Numeric feature contributions are scaled to this specific message."
                )
        except Exception:
            pass
