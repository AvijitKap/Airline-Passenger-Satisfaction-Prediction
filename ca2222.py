# ============================================
# Airplane Passenger Tracking ML Project
# Regression: Satisfaction Score Prediction
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

import joblib

# --------------------------------------------
# 1. Load Dataset
# --------------------------------------------
# Make sure the CSV is in the same directory or update the path
df = pd.read_csv(r"F:\gfhgjhkjl\synthetic_flight_passenger_data.csv")


print("Dataset Shape:", df.shape)
print(df.head())

# --------------------------------------------
# 2. Data Cleaning
# --------------------------------------------
df.drop_duplicates(inplace=True)

# Handle missing values
for col in df.columns:
    if df[col].dtype == "object":
        df[col].fillna(df[col].mode()[0], inplace=True)
    else:
        df[col].fillna(df[col].median(), inplace=True)

# --------------------------------------------
# 3. Feature Engineering
# --------------------------------------------

# Delay severity category
df["Delay_Severity"] = pd.cut(
    df["Delay_Minutes"],
    bins=[-1, 0, 15, 60, np.inf],
    labels=["No_Delay", "Minor", "Moderate", "Severe"]
)

# Price per mile
df["Price_per_Mile"] = df["Price_USD"] / (df["Distance_Miles"] + 1)

# Booking behavior
df["Booking_Type"] = np.where(
    df["Booking_Days_In_Advance"] <= 3,
    "Last_Minute",
    "Planned"
)

# --------------------------------------------
# 4. Encode Categorical Features (FIXED)
# --------------------------------------------
label_encoders = {}

categorical_cols = df.select_dtypes(include=["object", "category"]).columns
print("\nEncoding categorical columns:\n", categorical_cols)

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# --------------------------------------------
# 5. Define Features & Target
# --------------------------------------------
TARGET = "Flight_Satisfaction_Score"

X = df.drop(columns=[
    "Passenger_ID",
    "Flight_ID",
    TARGET
])

y = df[TARGET]

# Sanity check: no object columns
print("\nFeature dtypes:\n", X.dtypes.value_counts())

# --------------------------------------------
# 6. Train-Test Split
# --------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# --------------------------------------------
# 7. Define Models
# --------------------------------------------
models = {
    "Linear Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ]),
    "Decision Tree": DecisionTreeRegressor(
        max_depth=10,
        random_state=42
    ),
    "Random Forest": RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        random_state=42,
        n_jobs=-1
    )
}

# --------------------------------------------
# 8. Train & Evaluate Models
# --------------------------------------------
results = {}
best_model = None
best_r2 = -np.inf

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    results[name] = r2

    print(f"\n{name}")
    print(f"MAE : {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²  : {r2:.4f}")

    if r2 > best_r2:
        best_r2 = r2
        best_model = model
        best_model_name = name

# --------------------------------------------
# 9. Save Best Model
# --------------------------------------------
joblib.dump(best_model, "best_airline_satisfaction_model.pkl")
print(f"\n✅ Best Model Saved: {best_model_name}")

# --------------------------------------------
# 10. Make Predictions
# --------------------------------------------
sample_predictions = best_model.predict(X_test.iloc[:5])
print("\nSample Predictions:", sample_predictions)

# --------------------------------------------
# 11. Visualization
# --------------------------------------------

# Model comparison plot
plt.figure(figsize=(8,5))
sns.barplot(x=list(results.keys()), y=list(results.values()))
plt.title("Model Comparison (R² Score)")
plt.ylabel("R² Score")
plt.xticks(rotation=20)
plt.show()

# Actual vs Predicted plot
best_preds = best_model.predict(X_test)

plt.figure(figsize=(6,6))
plt.scatter(y_test, best_preds, alpha=0.5)
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    "r--"
)
plt.xlabel("Actual Satisfaction Score")
plt.ylabel("Predicted Satisfaction Score")
plt.title(f"Actual vs Predicted ({best_model_name})")
plt.show()

#asdfvbnj
plt.figure(figsize=(8,5))
sns.histplot(y, bins=20, kde=True)
plt.title("Distribution of Flight Satisfaction Score")
plt.xlabel("Satisfaction Score")
plt.ylabel("Frequency")
plt.show()

#drxtfyguhijokpl,;.
if best_model_name in ["Decision Tree", "Random Forest"]:

    importances = best_model.feature_importances_
    feature_names = X.columns

    feat_imp = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)

    plt.figure(figsize=(10,6))
    sns.barplot(
        x="Importance",
        y="Feature",
        data=feat_imp.head(10)
    )
    plt.title(f"Top 10 Feature Importances ({best_model_name})")
    plt.show()