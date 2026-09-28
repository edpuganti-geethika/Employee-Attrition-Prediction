import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# Load dataset
df = pd.read_csv("data/employee_data.csv")

# Encode categorical columns
department_encoder = LabelEncoder()
overtime_encoder = LabelEncoder()
attrition_encoder = LabelEncoder()

df["Department"] = department_encoder.fit_transform(df["Department"])
df["OverTime"] = overtime_encoder.fit_transform(df["OverTime"])
df["Attrition"] = attrition_encoder.fit_transform(df["Attrition"])

# Features and target
X = df.drop("Attrition", axis=1)
y = df["Attrition"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------- Random Forest ----------------
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)
rf_prob = rf_model.predict_proba(X_test)[:, 1]

# ---------------- Logistic Regression ----------------
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lr_model = LogisticRegression(random_state=42)

lr_model.fit(X_train_scaled, y_train)

lr_pred = lr_model.predict(X_test_scaled)
lr_prob = lr_model.predict_proba(X_test_scaled)[:, 1]

# ---------------- Results ----------------
print("\n========== MODEL COMPARISON ==========")

print("\nRandom Forest")
print("Accuracy :", round(accuracy_score(y_test, rf_pred) * 100, 2), "%")
print("Precision:", round(precision_score(y_test, rf_pred) * 100, 2), "%")
print("Recall   :", round(recall_score(y_test, rf_pred) * 100, 2), "%")
print("F1 Score :", round(f1_score(y_test, rf_pred) * 100, 2), "%")
print("ROC-AUC  :", round(roc_auc_score(y_test, rf_prob), 2))

print("\nLogistic Regression")
print("Accuracy :", round(accuracy_score(y_test, lr_pred) * 100, 2), "%")
print("Precision:", round(precision_score(y_test, lr_pred) * 100, 2), "%")
print("Recall   :", round(recall_score(y_test, lr_pred) * 100, 2), "%")
print("F1 Score :", round(f1_score(y_test, lr_pred) * 100, 2), "%")
print("ROC-AUC  :", round(roc_auc_score(y_test, lr_prob), 2))

print("\n======================================")
print("Model comparison completed successfully!")