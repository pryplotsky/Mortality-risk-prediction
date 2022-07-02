# Mortality risk prediction on hospitalization level using radiology reports with statistical, machine and deep learning techniques
To predict predict an in-hospital all-cause mortality during one hospitalization based on a huge dataset, we built logistic regression (LR), gradient boosting (XGB) and feedforward neural network (FNN) using clinical characteristics and radiology reports, which were converted into numerical feature vectors utilizing Bidirectional Encoder Representations from Transformers (BERT). These models were compared on three different sets of features: a) clinical characteristics of patients, b) radiology reports and c) clinical characteristics of patients with radiology reports. To evaluate our models we used ROC AUC, PR AUC, Accuracy, Sensitivity, Specificity, Precision and F1 score.
Scripts:
1. Embedding - to get 768 BERT embedding of radiology reports for one hospitalization.
2. ML_models - to get predictions from 3 ML models (LG, XGB, FNN)
