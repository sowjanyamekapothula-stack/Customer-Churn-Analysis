import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("customer_churn_sample.csv")

print("DATASET LOADED SUCCESSFULLY")
print("Shape:", df.shape)

print("\nFirst 5 rows:")
print(df.head())

print("\nColumn names:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:")
print(df.duplicated().sum())

print("\nSummary statistics:")
print(df.describe(include="all").T)
# Create a copy of the original dataset
clean_df = df.copy()

# 1. Remove duplicate rows
print("Duplicates before cleaning:", clean_df.duplicated().sum())

clean_df = clean_df.drop_duplicates()

print("Duplicates after cleaning:", clean_df.duplicated().sum())


# 2. Check missing values
print("\nMissing values before cleaning:")
print(clean_df.isnull().sum())


# 3. Standardize column names
clean_df.columns = (
    clean_df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
)

print("\nUpdated column names:")
print(clean_df.columns.tolist())


# 4. Remove extra spaces from text columns
for col in clean_df.select_dtypes(include="object").columns:
    clean_df[col] = clean_df[col].str.strip()


# 5. Validate after cleaning
print("\nFinal dataset shape:", clean_df.shape)
print("Remaining duplicates:", clean_df.duplicated().sum())
print("\nRemaining missing values:")
print(clean_df.isnull().sum())
# Select numeric columns
numeric_cols = clean_df.select_dtypes(
    include="number"
).columns

# Detect potential outliers using IQR
for col in numeric_cols:
    Q1 = clean_df[col].quantile(0.25)
    Q3 = clean_df[col].quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outliers = clean_df[
        (clean_df[col] < lower) |
        (clean_df[col] > upper)
    ]

    print(f"\nColumn: {col}")
    print("Potential outliers:", len(outliers))
    print("Lower limit:", lower)
    print("Upper limit:", upper)
    # 1. Churn distribution
print("\nChurn Distribution:")
print(clean_df["churn"].value_counts())

# 2. Average age and charges
print("\nSummary Statistics:")
print(clean_df[
    ["age", "tenuremonths",
     "monthlycharges", "totalcharges",
     "supporttickets"]
].describe())

# 3. Churn by contract type
print("\nChurn by Contract Type:")
print(pd.crosstab(
    clean_df["contracttype"],
    clean_df["churn"]
))

# 4. Churn by subscription type
print("\nChurn by Subscription Type:")
print(pd.crosstab(
    clean_df["subscriptiontype"],
    clean_df["churn"]
))
# Customer Churn Distribution Chart

plt.figure(figsize=(6, 4))

sns.countplot(
    data=clean_df,
    x="churn"
)

plt.title("Customer Churn Distribution")
plt.xlabel("Churn Status")
plt.ylabel("Number of Customers")

plt.tight_layout()
plt.show()
# Churn by Contract Type

plt.figure(figsize=(8, 5))

sns.countplot(
    data=clean_df,
    x="contracttype",
    hue="churn"
)

plt.title("Customer Churn by Contract Type")
plt.xlabel("Contract Type")
plt.ylabel("Number of Customers")

plt.tight_layout()
plt.show()
# Churn by Subscription Type

plt.figure(figsize=(8, 5))

sns.countplot(
    data=clean_df,
    x="subscriptiontype",
    hue="churn"
)

plt.title("Customer Churn by Subscription Type")
plt.xlabel("Subscription Type")
plt.ylabel("Number of Customers")

plt.tight_layout()
plt.show()
clean_df.to_csv(
    "customer_churn_cleaned.csv",
    index=False
)

print("Cleaned dataset exported successfully!")
# STEP 7: Export final project files

from pathlib import Path

output_folder = Path(".")

# 1. Save cleaned dataset
clean_df.to_csv(
    output_folder / "customer_churn_cleaned.csv",
    index=False
)

# 2. Create data dictionary
data_dictionary = pd.DataFrame({
    "Column": [
        "customerid",
        "gender",
        "age",
        "tenuremonths",
        "subscriptiontype",
        "monthlycharges",
        "totalcharges",
        "contracttype",
        "supporttickets",
        "paymentmethod",
        "churn"
    ],
    "Description": [
        "Unique customer identifier",
        "Customer gender",
        "Customer age in years",
        "Customer tenure in months",
        "Customer subscription plan",
        "Monthly subscription charges",
        "Total charges paid by customer",
        "Customer contract duration",
        "Number of support tickets",
        "Customer payment method",
        "Whether the customer churned"
    ]
})

data_dictionary.to_csv(
    output_folder / "data_dictionary.csv",
    index=False
)

# 3. Generate actual summary report
total_customers = len(clean_df)

churn_count = (
    clean_df["churn"].astype(str).str.strip().str.lower()
    == "yes"
).sum()

churn_rate = (
    churn_count / total_customers * 100
    if total_customers > 0 else 0
)

report = f"""
CUSTOMER CHURN - DATA QUALITY AND EDA REPORT

1. DATASET OVERVIEW
Total customers: {total_customers}
Total columns: {clean_df.shape[1]}

2. DATA QUALITY
Duplicate rows remaining: {clean_df.duplicated().sum()}
Missing values remaining: {clean_df.isnull().sum().sum()}

3. CHURN ANALYSIS
Churned customers: {churn_count}
Customers retained: {total_customers - churn_count}
Churn rate: {churn_rate:.2f}%

4. CONTRACT ANALYSIS
{pd.crosstab(clean_df['contracttype'], clean_df['churn']).to_string()}

5. SUBSCRIPTION ANALYSIS
{pd.crosstab(clean_df['subscriptiontype'], clean_df['churn']).to_string()}

6. CLEANING DECISIONS
- Removed exact duplicate rows; none were found.
- Standardized column names to lowercase.
- Removed leading and trailing spaces from text fields.
- No missing values were found.
- No potential outliers were flagged by the IQR method.
- No outliers were removed.

7. LIMITATION
The dataset contains only 15 customer records.
Findings are descriptive and should not be generalized
to a larger customer population.
"""

with open(
    output_folder / "analysis_report.txt",
    "w",
    encoding="utf-8"
) as file:
    file.write(report)

print("\nAll project files exported successfully!")
print("1. customer_churn_cleaned.csv")
print("2. data_dictionary.csv")
print("3. analysis_report.txt")