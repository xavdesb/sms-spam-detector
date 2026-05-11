"""Reproducible training script for the SMS spam detector."""
import argparse
import os
import sys
import urllib.request
import zipfile

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from features import MessageFeatureExtractor, build_preprocessor, to_dataframe

DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
RANDOM_STATE = 42
TEST_SIZE = 0.2


def parse_args():
    p = argparse.ArgumentParser(description="Train SMS spam classifier")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--model-output", default="model.joblib")
    p.add_argument("--test-size", type=float, default=TEST_SIZE)
    p.add_argument("--random-state", type=int, default=RANDOM_STATE)
    p.add_argument("--no-plots", action="store_true")
    return p.parse_args()


def download_dataset(data_dir, url=DATA_URL):
    os.makedirs(data_dir, exist_ok=True)
    target = os.path.join(data_dir, "SMSSpamCollection")
    if os.path.exists(target):
        print(f"Dataset found at {target}")
        return target
    print(f"Downloading dataset from {url} ...")
    zip_path = os.path.join(data_dir, "smsspamcollection.zip")
    urllib.request.urlretrieve(url, zip_path)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(data_dir)
    os.remove(zip_path)
    print(f"Saved to {target}")
    return target


def load_data(data_file):
    df = pd.read_csv(
        data_file, sep="\t", header=None,
        names=["label", "message"], encoding="latin-1"
    )
    assert df["label"].isin(["ham", "spam"]).all(), "Unexpected labels"
    assert df["message"].notna().all(), "Null messages found"
    return df


def make_pipeline(classifier):
    return Pipeline([
        ("to_df", FunctionTransformer(to_dataframe, validate=False)),
        ("preprocessor", build_preprocessor()),
        ("clf", classifier),
    ])


def spam_idx(pipeline):
    return list(pipeline.classes_).index("spam")


def evaluate(pipeline, X_test, y_test, name):
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, spam_idx(pipeline)]
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, pos_label="spam")
    rec = recall_score(y_test, y_pred, pos_label="spam")
    f1 = f1_score(y_test, y_pred, pos_label="spam")
    auc = roc_auc_score((y_test == "spam").astype(int), y_proba)
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(classification_report(y_test, y_pred))
    return {"Model": name, "Accuracy": acc, "Precision": prec,
            "Recall": rec, "F1": f1, "ROC-AUC": auc,
            "y_pred": y_pred, "y_proba": y_proba}


def plot_confusion_matrices(models_results, X_test, y_test, save_plots):
    n = len(models_results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, result in zip(axes, models_results):
        cm = confusion_matrix(y_test, result["y_pred"], labels=["ham", "spam"])
        disp = ConfusionMatrixDisplay(cm, display_labels=["ham", "spam"])
        disp.plot(ax=ax, colorbar=False)
        ax.set_title(result["Model"])
    plt.tight_layout()
    if save_plots:
        plt.savefig("confusion_matrices.png", dpi=100)
        print("Saved confusion_matrices.png")
    plt.close()


def plot_pr_curves(models_results, y_test, save_plots):
    y_bin = (y_test == "spam").astype(int)
    fig, ax = plt.subplots(figsize=(7, 5))
    for result in models_results:
        prec, rec, _ = precision_recall_curve(y_bin, result["y_proba"])
        ax.plot(rec, prec, label=f"{result['Model']} (F1={result['F1']:.3f})")
    baseline_prec = y_bin.mean()
    ax.axhline(baseline_prec, linestyle="--", color="gray", label="Baseline (always-ham)")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves")
    ax.legend()
    plt.tight_layout()
    if save_plots:
        plt.savefig("pr_curves.png", dpi=100)
        print("Saved pr_curves.png")
    plt.close()


def plot_roc_curves(models_results, y_test, save_plots):
    y_bin = (y_test == "spam").astype(int)
    fig, ax = plt.subplots(figsize=(7, 5))
    for result in models_results:
        fpr, tpr, _ = roc_curve(y_bin, result["y_proba"])
        ax.plot(fpr, tpr, label=f"{result['Model']} (AUC={result['ROC-AUC']:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves")
    ax.legend()
    plt.tight_layout()
    if save_plots:
        plt.savefig("roc_curves.png", dpi=100)
        print("Saved roc_curves.png")
    plt.close()


def error_analysis(pipeline, X_test, y_test, name="best model"):
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, spam_idx(pipeline)]

    fp_mask = (y_test == "ham") & (y_pred == "spam")
    fn_mask = (y_test == "spam") & (y_pred == "ham")

    fp_df = pd.DataFrame({"message": X_test[fp_mask], "spam_proba": y_proba[fp_mask]})
    fp_df = fp_df.sort_values("spam_proba", ascending=False).head(10)

    fn_df = pd.DataFrame({"message": X_test[fn_mask], "spam_proba": y_proba[fn_mask]})
    fn_df = fn_df.sort_values("spam_proba", ascending=True).head(10)

    print(f"\n--- Error Analysis: {name} ---")
    print(f"\nFalse Positives (ham flagged as spam): {fp_mask.sum()}")
    for _, row in fp_df.iterrows():
        print(f"  [{row['spam_proba']:.3f}] {row['message'][:80]}")

    print(f"\nFalse Negatives (spam missed): {fn_mask.sum()}")
    for _, row in fn_df.iterrows():
        print(f"  [{row['spam_proba']:.3f}] {row['message'][:80]}")

    return fp_df, fn_df


def lr_top_features(pipeline, top_n=20):
    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    coefs = pipeline.named_steps["clf"].coef_[0]
    indices = np.argsort(coefs)

    print(f"\n--- Top {top_n} spam indicators ---")
    for i in indices[-top_n:][::-1]:
        clean = feature_names[i].replace("tfidf__", "").replace("numeric__", "")
        print(f"  {clean:40s}  {coefs[i]:+.4f}")

    print(f"\n--- Top {top_n} ham indicators ---")
    for i in indices[:top_n]:
        clean = feature_names[i].replace("tfidf__", "").replace("numeric__", "")
        print(f"  {clean:40s}  {coefs[i]:+.4f}")


def main():
    args = parse_args()
    save_plots = not args.no_plots

    data_file = download_dataset(args.data_dir)
    df = load_data(data_file)
    print(f"\nLoaded {len(df)} messages: {df['label'].value_counts().to_dict()}")

    X = df["message"].values
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size,
        random_state=args.random_state, stratify=y
    )
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")

    # Baseline: always predict ham
    y_baseline = np.full(len(y_test), "ham")
    baseline_acc = accuracy_score(y_test, y_baseline)
    print(f"\nBaseline (always-ham) accuracy: {baseline_acc:.4f}")
    print(classification_report(y_test, y_baseline))

    models = {
        "MultinomialNB": make_pipeline(MultinomialNB(alpha=0.1)),
        "LogisticRegression": make_pipeline(
            LogisticRegression(C=1.0, max_iter=1000,
                               random_state=args.random_state,
                               class_weight="balanced")
        ),
        "LinearSVM": make_pipeline(
            SGDClassifier(loss="modified_huber", random_state=args.random_state,
                          class_weight="balanced", max_iter=1000)
        ),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.random_state)
    results = []
    for name, pipeline in models.items():
        print(f"\nTraining {name}...")
        pipeline.fit(X_train, y_train)
        cv_scores = cross_val_score(pipeline, X_train, y_train,
                                    cv=cv, scoring="f1_macro", n_jobs=-1)
        result = evaluate(pipeline, X_test, y_test, name)
        result["CV_F1_mean"] = cv_scores.mean()
        result["CV_F1_std"] = cv_scores.std()
        results.append(result)

    # Comparison table
    metrics_df = pd.DataFrame([
        {k: v for k, v in r.items() if k not in ("y_pred", "y_proba")}
        for r in results
    ])
    metrics_df = metrics_df.set_index("Model")
    print("\n\n=== Model Comparison ===")
    print(metrics_df.round(4).to_string())

    # Plots
    if save_plots:
        plot_confusion_matrices(results, X_test, y_test, save_plots)
        plot_pr_curves(results, y_test, save_plots)
        plot_roc_curves(results, y_test, save_plots)

    # Best model by F1
    best_idx = max(range(len(results)), key=lambda i: results[i]["F1"])
    best_name = results[best_idx]["Model"]
    best_pipeline = models[best_name]
    print(f"\nBest model by spam F1: {best_name} (F1={results[best_idx]['F1']:.4f})")

    # Error analysis on best model
    error_analysis(best_pipeline, X_test, y_test, best_name)

    # Feature importance (LR always available)
    lr_pipeline = models["LogisticRegression"]
    lr_top_features(lr_pipeline)

    # Save best model
    joblib.dump(best_pipeline, args.model_output)
    print(f"\nSaved {best_name} pipeline to {args.model_output}")

    # Print metrics for model_card.md
    best_result = results[best_idx]
    print(f"\n--- Fill these into model_card.md ---")
    print(f"Precision (spam): {best_result['Precision']:.4f}")
    print(f"Recall    (spam): {best_result['Recall']:.4f}")
    print(f"F1        (spam): {best_result['F1']:.4f}")
    print(f"ROC-AUC:          {best_result['ROC-AUC']:.4f}")


if __name__ == "__main__":
    main()
