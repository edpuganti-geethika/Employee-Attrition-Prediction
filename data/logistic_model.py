import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("data/employee_data.csv")

# Encode categorical columns
department_encoder = LabelEncoder()
overtime_encoder = LabelEncoder()
attrition_encoder = LabelEncoder()

df["Department"] = department_encoder.fit_transform(df["Department"])
df["OverTime"] = overtime_encoder.fit_transform(df["OverTime"])
df["Attrition"] = attrition_encoder.fit_transform(df["Attrition"])

# Separate features and target
X = df.drop("Attrition", axis=1)
y = df["Attrition"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# Create Logistic Regression model
model = LogisticRegression(random_state=42)

# Train model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Logistic Regression trained successfully!")
print(f"Model Accuracy: {accuracy * 100:.2f}%")

