import pandas as pd
import numpy as np

# For reproducible results
np.random.seed(42)

n = 500

departments = ["Sales", "Engineering", "HR", "Marketing", "Finance"]

data = {
    "EmployeeID": range(1001, 1001 + n),
    "Age": np.random.randint(21, 61, n),
    "Department": np.random.choice(departments, n),
    "MonthlyIncome": np.random.randint(18000, 120000, n),
    "YearsAtCompany": np.random.randint(0, 21, n),
    "JobSatisfaction": np.random.randint(1, 6, n),
    "WorkLifeBalance": np.random.randint(1, 6, n),
    "PerformanceRating": np.random.randint(1, 6, n),
    "OverTime": np.random.choice(["Yes", "No"], n, p=[0.3, 0.7]),
    "DistanceFromHome": np.random.randint(1, 51, n)
}

df = pd.DataFrame(data)

# Create a realistic attrition probability
risk = (
    0.10
    + (df["OverTime"] == "Yes") * 0.25
    + (df["JobSatisfaction"] <= 2) * 0.15
    + (df["WorkLifeBalance"] <= 2) * 0.10
    + (df["DistanceFromHome"] > 30) * 0.10
    + (df["YearsAtCompany"] <= 2) * 0.10
    - (df["JobSatisfaction"] >= 4) * 0.08
    - (df["MonthlyIncome"] > 80000) * 0.05
)

risk = np.clip(risk, 0.03, 0.80)

df["Attrition"] = np.where(
    np.random.random(n) < risk,
    "Yes",
    "No"
)

# Save the dataset
df.to_csv("data/employee_data.csv", index=False)

print("Dataset created successfully!")
print(f"Total employees: {len(df)}")
print("\nFirst 5 rows:")
print(df.head())
print("\nAttrition counts:")
print(df["Attrition"].value_counts())