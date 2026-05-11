import re
import string
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer


class MessageFeatureExtractor(BaseEstimator, TransformerMixin):
    CURRENCY_RE = re.compile(r'[$£€]')

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        rows = []
        for msg in X:
            msg = str(msg)
            char_len = len(msg)
            word_count = len(msg.split())
            upper_count = sum(c.isupper() for c in msg)
            upper_ratio = upper_count / max(char_len, 1)
            punct_count = sum(c in string.punctuation for c in msg)
            has_currency = float(bool(self.CURRENCY_RE.search(msg)))
            digit_count = sum(c.isdigit() for c in msg)
            rows.append([char_len, word_count, upper_count, upper_ratio,
                         punct_count, has_currency, digit_count])
        return np.array(rows, dtype=float)

    def get_feature_names_out(self, input_features=None):
        return np.array(['char_length', 'word_count', 'uppercase_count',
                         'uppercase_ratio', 'punctuation_count',
                         'has_currency', 'digit_count'])


def to_dataframe(X):
    return pd.DataFrame({'message': list(X)})


def build_preprocessor():
    tfidf = TfidfVectorizer(
        analyzer='word',
        ngram_range=(1, 2),
        min_df=2,
        max_features=10000,
        sublinear_tf=True,
        strip_accents='unicode',
        token_pattern=r'\b[a-zA-Z][a-zA-Z]+\b',
    )
    numeric = Pipeline([
        ('extractor', MessageFeatureExtractor()),
        ('scaler', MaxAbsScaler()),
    ])
    return ColumnTransformer(
        transformers=[
            ('tfidf', tfidf, 'message'),
            ('numeric', numeric, 'message'),
        ],
        sparse_threshold=0,
    )
