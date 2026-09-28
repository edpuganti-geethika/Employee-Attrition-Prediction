import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("data/employee_data.csv")

print(df.head())
print(df.info())

# 1. Attrition by Department
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="Department", hue="Attrition")
plt.title("Attrition by Department")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 2. Monthly Income vs Attrition
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="Attrition", y="MonthlyIncome")
plt.title("Monthly Income vs Attrition")
plt.tight_layout()
plt.show()

# 3. Correlation Heatmap
plt.figure(figsize=(10, 7))
numeric_df = df.select_dtypes(include="number")
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

# 4. Feature Importance
import joblib

model = joblib.load("employee_attrition_model.pkl")

features = [
    "EmployeeID",
    "Age",
    "Department",
    "MonthlyIncome",
    "YearsAtCompany",
    "JobSatisfaction",
    "WorkLifeBalance",
    "PerformanceRating",
    "OverTime",
    "DistanceFromHome"
]

importance = model.feature_importances_

plt.figure(figsize=(10, 6))
plt.barh(features, importance)
plt.xlabel("Importance")
plt.ylabel("Features")
plt.title("Feature Importance")
plt.tight_layout()
plt.show()

print("EDA completed successfully!")