import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("data/employee_data.csv")

# Convert text columns into numbers
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

# Train model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Check accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Model trained successfully!")
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Save model
joblib.dump(model, "employee_attrition_model.pkl")

print("Model saved successfully!")
