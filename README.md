# Mortality risk prediction on hospitalization level using radiology reports with statistical, machine and deep learning techniques
To predict in-hospital all-cause mortality during a single hospitalization, we developed three models — Logistic Regression (LR), Gradient Boosting (XGB), and a Feedforward Neural Network (FNN) — based on a large clinical dataset. These models use both clinical characteristics and radiology reports, the latter of which were transformed into numerical feature vectors using Bidirectional Encoder Representations from Transformers (BERT).

We evaluated model performance across three different feature sets:
1. Clinical characteristics only
2. Radiology reports only
3. Combined clinical characteristics and radiology reports

Performance was assessed using several metrics: ROC AUC, PR AUC, Accuracy, Sensitivity, Specificity, and F1 Score.


Scripts:
1. Embedding: Generates 768-dimensional BERT embeddings for radiology reports associated with each hospitalization.
2. ML_models: Produces predictions using the three models (LR, XGB, FNN).
