# Model Card — SMS Spam Classifier

## Model Details

| | |
|---|---|
| **Type** | LinearSVM / SGDClassifier(loss='modified_huber') (sklearn Pipeline) |
| **Version** | 1.0 |
| **Framework** | scikit-learn ≥ 1.3 |
| **Training data** | UCI SMS Spam Collection (5,572 messages) |
| **Date** | 2026-05 |

**Pipeline:** TF-IDF (unigrams + bigrams, ≤10,000 features) + 7 hand-crafted numeric features, combined via `ColumnTransformer`, fed into `SGDClassifier(loss='modified_huber', class_weight='balanced')` (Linear SVM with probability estimates).

## Intended Use

- Educational / portfolio demonstration of end-to-end ML workflow
- Illustrates honest model evaluation (precision, recall, error analysis)
- Not designed for production spam filtering

## Evaluation Metrics

Held-out test set: 20% stratified split, `random_state=42`. Run `python train.py` to reproduce.

| Metric | Ham | Spam | Weighted avg |
|--------|-----|------|-------------|
| Precision | 0.99 | 0.99 | 0.99 |
| Recall | 1.00 | 0.94 | 0.99 |
| F1-score | 0.99 | 0.97 | 0.99 |
| ROC-AUC | | 0.99 | |

**Cross-validation:** 5-fold stratified CV F1 macro on training set: 0.9766 ± 0.0051

## Known Limitations

- **Temporal drift:** Trained on SMS messages from approximately 2011–2012 (UK/Singapore). Spam vocabulary evolves; novel patterns may be missed.
- **Language:** Vocabulary is English-only. Messages in other scripts or mixed-language texts are handled poorly.
- **Class imbalance:** Dataset is 87% ham / 13% spam. Even with balanced weighting, the model is calibrated on this distribution and may not transfer to inboxes with different spam rates.
- **Short messages:** Very short messages (< 5 words) have few TF-IDF features active; predictions rely heavily on numeric features in those cases.
- **No adversarial testing:** The model has not been tested against deliberate obfuscation (e.g., l33tspeak, zero-width characters).

## Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| False positives (legitimate messages flagged as spam) | Raise the threshold slider in the app to reduce sensitivity |
| False negatives (spam reaching inbox) | Lower the threshold slider for higher recall |
| Bias against certain message styles | Review false positive analysis in `notebook.ipynb` |
| Over-reliance on model output | Model probability is shown alongside the label; users should apply judgment |

## Training Data

**Source:** UCI Machine Learning Repository — SMS Spam Collection  
**URL:** https://archive.ics.uci.edu/dataset/228/sms+spam+collection  
**License:** Creative Commons CC BY 4.0  
**Size:** 5,572 labeled SMS messages (4,825 ham, 747 spam)  
**Collection:** Messages were compiled from the Grumbletext website, the NUS SMS Corpus, and the British Broadcasting Corporation (UK mobile operators).

## Ethical Notes

This model is built for educational purposes. Deploying any automated spam filter in a real messaging system requires:

1. Ongoing retraining as spam patterns evolve
2. User-visible controls to adjust sensitivity
3. A human review process for borderline cases
4. Transparency about false positive rates (legitimate messages blocked)
5. Compliance with applicable messaging and data protection regulations
