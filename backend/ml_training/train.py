import pandas as pd
import numpy as np
import joblib
import shap

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# 1️⃣ Load Data
df = pd.read_csv("../../data/raw/creditcard.csv")

X = df.drop("Class", axis=1)
y = df["Class"]

# 2️⃣ Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 3️⃣ Handle Imbalance with SMOTE
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# 4️⃣ Model (Advanced)
model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    scale_pos_weight=10,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42
)

model.fit(X_train_res, y_train_res)

# 5️⃣ Evaluation
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))

# 6️⃣ Save Model
joblib.dump(model, "../../data/models/xgb_fraud_model.pkl")

# 7️⃣ SHAP Explainer
explainer = shap.Explainer(model)
joblib.dump(explainer, "../../data/models/shap_explainer.pkl")

print("Model & Explainer Saved Successfully 🚀")
